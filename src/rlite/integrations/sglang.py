"""SGLang-facing helpers built on top of the generic resharding core."""

from __future__ import annotations

import os
import socket
from dataclasses import replace
from typing import Iterable, Mapping, Optional

from rlite.transport import MemoryKind, RankDescriptor, TransportSession
from rlite.weight_mapping import Framework, resolve_rule, translate_tensor

from .remote import RemoteTopology, decode_payload, encode_payload
from ..resharding import (
    FrameworkRole,
    FrameworkSnapshot,
    ParameterRecord,
    PendingReceive,
    WorkerEndpoint,
    abort_receive,
    commit_receive,
    execute_exchange_plan,
    frozen_coordinator_from_payload,
    normalize_expert_canonical_names,
    prepare_receive,
    split_grouped_expert_record,
)

_PENDING_SGLANG_RECEIVES: dict[str, PendingReceive] = {}


class _SyntheticTensor:
    """Planner-only tensor placeholder that preserves first-axis indexing."""

    def __init__(self, shape: tuple[int, ...], item_size: int) -> None:
        self.shape = tuple(int(dim) for dim in shape)
        self.item_size = int(item_size)

    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise TypeError("Synthetic tensors only support integer first-axis indexing.")
        if not self.shape:
            raise IndexError("Cannot index a scalar synthetic tensor.")
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)
        return _SyntheticTensor(self.shape[1:], self.item_size)


def _endpoint_from_model_runner(
    model_runner,
    *,
    role: FrameworkRole | str,
    rank_offset: int = 0,
    host: str | None = None,
    nic_names: Iterable[str] = (),
    provider_names: Iterable[str] = (),
) -> WorkerEndpoint:
    tp_size = max(1, int(getattr(model_runner, "tp_size", 1)))
    pp_size = max(1, int(getattr(model_runner, "pp_size", 1)))
    ep_size = max(1, int(getattr(model_runner, "moe_ep_size", 1)))
    dp_rank = int(getattr(model_runner, "dp_rank", 0) or 0)
    dp_size = int(getattr(getattr(model_runner, "server_args", None), "dp_size", 1) or 1)
    tp_rank = int(getattr(model_runner, "tp_rank", 0))
    pp_rank = int(getattr(model_runner, "pp_rank", 0))
    moe_ep_rank = int(getattr(model_runner, "moe_ep_rank", 0))
    global_rank = int(rank_offset) + dp_rank * (tp_size * pp_size) + pp_rank * tp_size + tp_rank
    return WorkerEndpoint(
        rank=global_rank,
        framework=Framework.SGLANG.value,
        role=role,
        host=host or socket.gethostname(),
        process_id=os.getpid(),
        device_id=getattr(model_runner, "gpu_id", None),
        nic_names=tuple(nic_names),
        provider_names=tuple(provider_names),
        tensor_parallel_rank=tp_rank,
        tensor_parallel_size=tp_size,
        expert_parallel_rank=moe_ep_rank,
        expert_parallel_size=ep_size,
        moe_tensor_parallel_rank=tp_rank % max(1, tp_size // ep_size),
        moe_tensor_parallel_size=max(1, tp_size // ep_size),
        pipeline_parallel_rank=pp_rank,
        pipeline_parallel_size=pp_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=max(1, dp_size),
    )


def collect_sglang_snapshot(
    model_runner,
    profile,
    role: FrameworkRole | str = FrameworkRole.TARGET,
    *,
    rank_offset: int = 0,
    host: str | None = None,
    nic_names: Iterable[str] = (),
    provider_names: Iterable[str] = (),
) -> FrameworkSnapshot:
    """Collect a snapshot from a live SGLang ModelRunner."""

    endpoint = _endpoint_from_model_runner(
        model_runner,
        role=role,
        rank_offset=rank_offset,
        host=host,
        nic_names=nic_names,
        provider_names=provider_names,
    )
    experts_per_rank = 0
    if endpoint.expert_parallel_size > 0 and profile.num_experts is not None:
        experts_per_rank = profile.num_experts // endpoint.expert_parallel_size
    endpoint = replace(endpoint, metadata={"experts_per_rank": str(experts_per_rank)})

    records = []
    for name, parameter in model_runner.model.named_parameters(recurse=True):
        mapped = translate_tensor(name, Framework.SGLANG, Framework.SGLANG, profile, view="local_shard")
        rule = resolve_rule(name, Framework.SGLANG, profile)
        component_sizes = ()
        local_component_sizes = ()
        if mapped.packing.components and mapped.packing.axis is not None:
            logical_sizes = mapped.packing.component_sizes(profile)
            component_sizes = tuple(logical_sizes[component] for component in mapped.packing.components)
            total_logical = sum(component_sizes)
            local_total = mapped.source_local_shape[mapped.packing.axis]
            local_component_sizes = tuple(
                (size * local_total) // total_logical if total_logical else size
                for size in component_sizes
            )
        record = ParameterRecord(
            record_id=name,
            framework_name=name,
            tensor=parameter.data,
            dtype=str(parameter.dtype),
            logical_shape=mapped.source_logical_shape or tuple(parameter.shape),
            local_shape=mapped.source_local_shape or tuple(parameter.shape),
            actual_shape=tuple(parameter.shape),
            canonical_names=mapped.canonical_names,
            packing=mapped.packing,
            parallel=mapped.parallel,
            tensor_role=mapped.tensor_role,
            memory_kind=MemoryKind.CUDA if str(parameter.device).startswith("cuda") else MemoryKind.CPU,
            component_logical_sizes=component_sizes,
            component_local_sizes=local_component_sizes,
            transpose=rule.transpose,
            reshape=rule.reshape,
            match_groups=mapped.match_groups,
            metadata={"experts_per_rank": str(experts_per_rank)},
        )
        normalized = normalize_expert_canonical_names(record, endpoint)
        records.extend(split_grouped_expert_record(normalized, endpoint))

    return FrameworkSnapshot(
        endpoint=endpoint,
        records=tuple(records),
        metadata={"experts_per_rank": str(experts_per_rank)},
    )


def build_sglang_update_payload(
    profile,
    execution_slice,
    *,
    peer_descriptors: Mapping[int, RankDescriptor],
    completed_source_ranks: Iterable[int] = (),
    session_host: str = "",
    session_nic_name: str = "",
    session_provider_name: str = "",
    rank_offset: int = 0,
) -> dict[str, object]:
    """Build the payload consumed by `load_format='rlite'` in SGLang."""

    return {
        "profile": profile,
        "execution_slice": execution_slice,
        "peer_descriptors": dict(peer_descriptors),
        "completed_source_ranks": tuple(int(value) for value in completed_source_ranks),
        "session_host": session_host,
        "session_nic_name": session_nic_name or execution_slice.selected_nic_name,
        "session_provider_name": session_provider_name or execution_slice.selected_provider_name,
        "rank_offset": int(rank_offset),
    }


def build_sglang_receive_payload(
    request_id: str,
    profile,
    execution_slices_by_rank: Mapping[int, object],
    *,
    world_size: int,
    session_host: str = "",
    session_nic_name: str = "",
    session_provider_name: str = "",
    rank_offset: int = 0,
) -> dict[str, object]:
    """Build the payload consumed by the explicit prepare/commit/abort receive lifecycle."""

    return {
        "request_id": str(request_id),
        "profile": profile,
        "execution_slices_by_rank": {
            int(rank): execution_slice for rank, execution_slice in execution_slices_by_rank.items()
        },
        "world_size": int(world_size),
        "session_host": session_host,
        "session_nic_name": session_nic_name,
        "session_provider_name": session_provider_name,
        "rank_offset": int(rank_offset),
    }


def _extract_rlite_payload(named_tensors):
    if isinstance(named_tensors, dict):
        return named_tensors
    if isinstance(named_tensors, (list, tuple)):
        if len(named_tensors) == 1 and isinstance(named_tensors[0], tuple):
            name, value = named_tensors[0]
            if name == "__rlite_payload__":
                return value
        if len(named_tensors) == 1 and isinstance(named_tensors[0], dict):
            return named_tensors[0]
    raise TypeError(
        "RLite SGLang updates expect a single payload dict or "
        "[('__rlite_payload__', payload)]."
    )


def apply_sglang_rlite_update(model_runner, named_tensors) -> tuple[bool, str]:
    """Entry point called from SGLang's ModelRunner patch."""

    payload = _extract_rlite_payload(named_tensors)
    execution_slice = payload["execution_slice"]
    profile = payload.get("profile")
    if profile is None:
        raise ValueError("RLite SGLang payload is missing the required architecture profile.")
    snapshot = collect_sglang_snapshot(
        model_runner,
        profile=profile,
        role=FrameworkRole.TARGET,
        rank_offset=int(payload.get("rank_offset", 0)),
        host=payload.get("session_host") or None,
        nic_names=(payload.get("session_nic_name", ""),),
        provider_names=(payload.get("session_provider_name", ""),),
    )
    peer_descriptors = {
        int(rank): descriptor for rank, descriptor in payload.get("peer_descriptors", {}).items()
    }
    ranks = {snapshot.endpoint.rank, *peer_descriptors.keys()}
    for task in execution_slice.send_tasks:
        ranks.add(int(task.src_rank))
        ranks.add(int(task.dst_rank))
    session = TransportSession(
        rank=snapshot.endpoint.rank,
        world_size=max(ranks) + 1 if ranks else snapshot.endpoint.rank + 1,
        host=payload.get("session_host") or snapshot.endpoint.host,
        nic_name=payload.get("session_nic_name", ""),
        provider_name=payload.get("session_provider_name", ""),
    )
    coordinator = frozen_coordinator_from_payload(
        peer_descriptors,
        tuple(payload.get("completed_source_ranks", ())),
    )
    execute_exchange_plan(snapshot, execution_slice, coordinator, session)
    return True, "Success"


def prepare_sglang_receive(model_runner, payload) -> dict[str, object]:
    """Prepare a receive on a live SGLang worker and return its descriptor summary."""

    request_id = str(payload["request_id"])
    profile = payload["profile"]
    snapshot = collect_sglang_snapshot(
        model_runner,
        profile=profile,
        role=FrameworkRole.TARGET,
        rank_offset=int(payload.get("rank_offset", 0)),
        host=payload.get("session_host") or None,
        nic_names=(payload.get("session_nic_name", ""),),
        provider_names=(payload.get("session_provider_name", ""),),
    )
    execution_slices_by_rank = {
        int(rank): slice_ for rank, slice_ in payload["execution_slices_by_rank"].items()
    }
    if snapshot.endpoint.rank not in execution_slices_by_rank:
        raise KeyError(
            f"RLite prepare payload has no execution slice for remote SGLang rank {snapshot.endpoint.rank}."
        )
    execution_slice = execution_slices_by_rank[snapshot.endpoint.rank]
    session = TransportSession(
        rank=snapshot.endpoint.rank,
        world_size=int(payload.get("world_size", snapshot.endpoint.rank + 1)),
        host=payload.get("session_host") or snapshot.endpoint.host,
        nic_name=payload.get("session_nic_name", "") or execution_slice.selected_nic_name,
        provider_name=payload.get("session_provider_name", "") or execution_slice.selected_provider_name,
    )
    existing = _PENDING_SGLANG_RECEIVES.pop(request_id, None)
    if existing is not None and not existing._finished:
        abort_receive(existing)
    pending = prepare_receive(snapshot, execution_slice, session)
    _PENDING_SGLANG_RECEIVES[request_id] = pending
    return {
        "rank": pending.rank,
        "rank_descriptor": pending.rank_descriptor,
        "requires_staging": pending.requires_staging,
        "fallback_bytes": pending.fallback_bytes,
    }


def commit_sglang_receive(request_id: str):
    pending = _PENDING_SGLANG_RECEIVES.pop(str(request_id))
    return commit_receive(pending)


def abort_sglang_receive(request_id: str):
    pending = _PENDING_SGLANG_RECEIVES.pop(str(request_id))
    return abort_receive(pending)


def prepare_sglang_rlite_receive(model_runner, named_tensors) -> tuple[bool, str]:
    payload = _extract_rlite_payload(named_tensors)
    try:
        return True, encode_payload(prepare_sglang_receive(model_runner, payload))
    except Exception as exc:
        return False, str(exc)


def commit_sglang_rlite_receive(named_tensors) -> tuple[bool, str]:
    payload = _extract_rlite_payload(named_tensors)
    try:
        result = commit_sglang_receive(str(payload["request_id"]))
        return True, encode_payload(
            {
                "rank": result.rank,
                "applied_binding_ids": result.applied_binding_ids,
            }
        )
    except Exception as exc:
        return False, str(exc)


def abort_sglang_rlite_receive(named_tensors) -> tuple[bool, str]:
    payload = _extract_rlite_payload(named_tensors)
    try:
        result = abort_sglang_receive(str(payload["request_id"]))
        return True, encode_payload(
            {
                "rank": result.rank,
                "prepared_binding_ids": result.prepared_binding_ids,
            }
        )
    except Exception as exc:
        return False, str(exc)


def synthesize_sglang_target_snapshots(
    source_snapshots: FrameworkSnapshot | Iterable[FrameworkSnapshot],
    target_profile,
    topology: RemoteTopology,
) -> tuple[FrameworkSnapshot, ...]:
    """Synthesize target snapshots from explicit remote topology metadata."""

    snapshots = (
        (source_snapshots,)
        if isinstance(source_snapshots, FrameworkSnapshot)
        else tuple(source_snapshots)
    )
    synthesized = []
    for worker in topology.workers:
        worker_profile = replace(
            target_profile,
            tensor_parallel_size=worker.tensor_parallel_size,
            expert_parallel_size=worker.expert_parallel_size,
            pipeline_parallel_size=worker.pipeline_parallel_size,
        )
        target_blueprints: dict[str, ParameterRecord] = {}
        for snapshot in snapshots:
            for record in snapshot.records:
                mapped = translate_tensor(
                    record.framework_name,
                    Framework.MEGATRON,
                    Framework.SGLANG,
                    worker_profile,
                    view="local_shard",
                )
                for target in mapped.targets:
                    rule = resolve_rule(target.name, Framework.SGLANG, worker_profile)
                    component_sizes = ()
                    component_local_sizes = ()
                    if rule.packing.components and rule.packing.axis is not None:
                        logical_sizes = rule.packing.component_sizes(worker_profile)
                        component_sizes = tuple(
                            logical_sizes[component] for component in rule.packing.components
                        )
                        total_logical = sum(component_sizes)
                        local_total = target.local_shape[rule.packing.axis]
                        component_local_sizes = tuple(
                            (size * local_total) // total_logical if total_logical else size
                            for size in component_sizes
                        )
                    local_shape = target.local_shape or record.local_shape
                    target_blueprints.setdefault(
                        target.name,
                        ParameterRecord(
                            record_id=target.name,
                            framework_name=target.name,
                            tensor=_SyntheticTensor(local_shape, 1),
                            dtype=record.dtype,
                            logical_shape=target.logical_shape or record.logical_shape,
                            local_shape=local_shape,
                            actual_shape=local_shape,
                            canonical_names=target.canonical_names,
                            packing=rule.packing,
                            parallel=rule.parallel,
                            tensor_role=rule.tensor_role,
                            memory_kind=MemoryKind.CUDA,
                            component_logical_sizes=component_sizes,
                            component_local_sizes=component_local_sizes,
                            transpose=rule.transpose,
                            reshape=rule.reshape,
                            metadata={},
                        ),
                    )
        experts_per_rank = 0
        if worker_profile.num_experts is not None:
            experts_per_rank = worker_profile.num_experts // max(1, worker.expert_parallel_size)
        endpoint = replace(
            worker.endpoint(role=FrameworkRole.TARGET),
            metadata={"experts_per_rank": str(experts_per_rank)},
        )
        records = []
        for blueprint in target_blueprints.values():
            normalized = normalize_expert_canonical_names(blueprint, endpoint)
            records.extend(split_grouped_expert_record(normalized, endpoint))
        synthesized.append(
            FrameworkSnapshot(
                endpoint=endpoint,
                records=tuple(records),
                metadata={"experts_per_rank": str(experts_per_rank)},
            )
        )
    return tuple(synthesized)


def sync_megatron_to_remote_sglang(
    source_snapshots: FrameworkSnapshot | Iterable[FrameworkSnapshot],
    *,
    train_profile,
    rollout_profile,
    topology: RemoteTopology,
    remote_url: str,
):
    """Prepare remote receives, execute local sources, then commit the remote side."""

    import requests
    from uuid import uuid4

    request_id = uuid4().hex
    source_snapshots = (
        (source_snapshots,)
        if isinstance(source_snapshots, FrameworkSnapshot)
        else tuple(source_snapshots)
    )
    target_snapshots = synthesize_sglang_target_snapshots(
        source_snapshots,
        rollout_profile,
        topology,
    )
    from ..resharding import build_exchange_plan

    plan = build_exchange_plan(source_snapshots, target_snapshots)
    payload = build_sglang_receive_payload(
        request_id,
        rollout_profile,
        {rank: plan.execution_slices[rank] for rank in topology.ranks},
        world_size=max(
            max(snapshot.endpoint.rank for snapshot in source_snapshots) + 1 if source_snapshots else 0,
            topology.world_size,
        ),
    )
    prepare_response = requests.post(
        f"{remote_url.rstrip('/')}/rlite/prepare_receive",
        json={"payload_b64": encode_payload(payload)},
        timeout=30,
    )
    prepare_response.raise_for_status()
    prepare_data = prepare_response.json()
    if not prepare_data.get("success", False):
        raise RuntimeError(prepare_data.get("message", "RLite remote prepare failed."))
    prepared_payload = decode_payload(prepare_data["payload_b64"])
    remote_descriptors = {
        int(rank): descriptor for rank, descriptor in prepared_payload["rank_descriptors"].items()
    }
    try:
        for snapshot in source_snapshots:
            execution_slice = plan.execution_slices.get(snapshot.endpoint.rank)
            if execution_slice is None or not execution_slice.send_tasks:
                continue
            session = TransportSession(
                rank=snapshot.endpoint.rank,
                world_size=payload["world_size"],
                host=snapshot.endpoint.host,
                nic_name=execution_slice.selected_nic_name,
                provider_name=execution_slice.selected_provider_name,
            )
            try:
                execute_exchange_plan(
                    snapshot,
                    execution_slice,
                    frozen_coordinator_from_payload(remote_descriptors, ()),
                    session,
                )
            finally:
                session.close()
        commit_response = requests.post(
            f"{remote_url.rstrip('/')}/rlite/commit_receive",
            json={"payload_b64": encode_payload({"request_id": request_id})},
            timeout=30,
        )
        commit_response.raise_for_status()
        commit_data = commit_response.json()
        if not commit_data.get("success", False):
            raise RuntimeError(commit_data.get("message", "RLite remote commit failed."))
        return {
            "plan": plan,
            "prepare": prepared_payload,
            "commit": decode_payload(commit_data["payload_b64"]) if "payload_b64" in commit_data else commit_data,
            "train_profile": train_profile,
            "rollout_profile": rollout_profile,
        }
    except Exception:
        try:
            requests.post(
                f"{remote_url.rstrip('/')}/rlite/abort_receive",
                json={"payload_b64": encode_payload({"request_id": request_id})},
                timeout=30,
            )
        except Exception:
            pass
        raise


def dispatch_sglang_update(engine, payload, *, flush_cache: bool = True):
    """Dispatch a prepared payload into `Engine.update_weights_from_tensor`."""

    return engine.update_weights_from_tensor(
        [("__rlite_payload__", payload)],
        load_format="rlite",
        flush_cache=flush_cache,
    )
