"""SGLang-facing helpers built on top of the generic resharding core."""

from __future__ import annotations

import os
import socket
from dataclasses import replace
from typing import Iterable, Mapping, Optional

from rlite.transport import MemoryKind, RankDescriptor, TransportSession
from rlite.weight_mapping import Framework, resolve_rule, translate_tensor

from ..resharding import (
    FrameworkRole,
    FrameworkSnapshot,
    ParameterRecord,
    WorkerEndpoint,
    execute_exchange_plan,
    frozen_coordinator_from_payload,
    normalize_expert_canonical_names,
    split_grouped_expert_record,
)


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


def dispatch_sglang_update(engine, payload, *, flush_cache: bool = True):
    """Dispatch a prepared payload into `Engine.update_weights_from_tensor`."""

    return engine.update_weights_from_tensor(
        [("__rlite_payload__", payload)],
        load_format="rlite",
        flush_cache=flush_cache,
    )
