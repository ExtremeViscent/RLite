"""Megatron-facing helpers built on top of the generic resharding core."""

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
    TopologyPolicy,
    WorkerEndpoint,
    build_exchange_plan,
    execute_exchange_plan,
    normalize_expert_canonical_names,
    split_grouped_expert_record,
)


def _group_ranks(group) -> tuple[int, ...]:
    if group is None:
        return ()
    if isinstance(group, (list, tuple)):
        return tuple(int(value) for value in group)
    for attr in ("ranks", "group_ranks"):
        value = getattr(group, attr, None)
        if value:
            return tuple(int(item) for item in value)
    try:
        import torch.distributed as dist

        return tuple(int(item) for item in dist.get_process_group_ranks(group))
    except Exception:
        return ()


def _rank_in_group(rank: int, group) -> int:
    ranks = _group_ranks(group)
    if not ranks:
        return 0
    try:
        return ranks.index(int(rank))
    except ValueError:
        return 0


def _infer_distributed_rank(model, default: int) -> int:
    for attr in ("global_rank", "rank", "distributed_rank"):
        value = getattr(model, attr, None)
        if isinstance(value, int):
            return value
    return int(default)


def _first_parameter(model):
    for _, parameter in model.named_parameters(recurse=True):
        return parameter
    return None


def _make_endpoint(
    model,
    *,
    role: FrameworkRole | str,
    rank_offset: int = 0,
    host: str | None = None,
    nic_names: Iterable[str] = (),
    provider_names: Iterable[str] = (),
) -> WorkerEndpoint:
    pg_collection = getattr(model, "pg_collection", None)
    rank = _infer_distributed_rank(model, 0) + int(rank_offset)
    parameter = _first_parameter(model)
    device_id = None
    if parameter is not None:
        device = getattr(parameter, "device", None)
        device_id = getattr(device, "index", None)
    tp_group = getattr(pg_collection, "tp", None)
    ep_group = getattr(pg_collection, "ep", None)
    expt_tp_group = getattr(pg_collection, "expt_tp", None)
    pp_group = getattr(pg_collection, "pp", None)
    dp_group = getattr(pg_collection, "dp", None)
    tp_ranks = _group_ranks(tp_group)
    ep_ranks = _group_ranks(ep_group)
    expt_tp_ranks = _group_ranks(expt_tp_group)
    pp_ranks = _group_ranks(pp_group)
    dp_ranks = _group_ranks(dp_group)
    return WorkerEndpoint(
        rank=rank,
        framework=Framework.MEGATRON.value,
        role=role,
        host=host or socket.gethostname(),
        process_id=os.getpid(),
        device_id=device_id,
        nic_names=tuple(nic_names),
        provider_names=tuple(provider_names),
        tensor_parallel_rank=_rank_in_group(rank, tp_ranks),
        tensor_parallel_size=max(1, len(tp_ranks) or 1),
        expert_parallel_rank=_rank_in_group(rank, ep_ranks),
        expert_parallel_size=max(1, len(ep_ranks) or 1),
        moe_tensor_parallel_rank=_rank_in_group(rank, expt_tp_ranks or tp_ranks),
        moe_tensor_parallel_size=max(1, len(expt_tp_ranks or tp_ranks) or 1),
        pipeline_parallel_rank=_rank_in_group(rank, pp_ranks),
        pipeline_parallel_size=max(1, len(pp_ranks) or 1),
        data_parallel_rank=_rank_in_group(rank, dp_ranks),
        data_parallel_size=max(1, len(dp_ranks) or 1),
    )


def collect_megatron_snapshot(
    model,
    profile,
    role: FrameworkRole | str = FrameworkRole.SOURCE,
    *,
    rank_offset: int = 0,
    host: str | None = None,
    nic_names: Iterable[str] = (),
    provider_names: Iterable[str] = (),
) -> FrameworkSnapshot:
    """Collect a Megatron-local snapshot usable by the planner/executor."""

    endpoint = _make_endpoint(
        model,
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
    for name, parameter in model.named_parameters(recurse=True):
        mapped = translate_tensor(name, Framework.MEGATRON, Framework.MEGATRON, profile, view="local_shard")
        rule = resolve_rule(name, Framework.MEGATRON, profile)
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


def _world_size_for_execution_slice(execution_slice, local_rank: int) -> int:
    ranks = {int(local_rank), *[int(rank) for rank in execution_slice.expected_source_ranks]}
    for task in execution_slice.send_tasks:
        ranks.add(int(task.src_rank))
        ranks.add(int(task.dst_rank))
    return max(ranks) + 1 if ranks else int(local_rank) + 1


def execute_megatron_exchange(
    model,
    profile,
    execution_slice,
    coordinator,
    *,
    transport_session: Optional[TransportSession] = None,
    role: FrameworkRole | str = FrameworkRole.BIDIRECTIONAL,
    rank_offset: int = 0,
    host: str | None = None,
    nic_names: Iterable[str] = (),
    provider_names: Iterable[str] = (),
):
    """Collect a live snapshot and execute one local exchange slice."""

    snapshot = collect_megatron_snapshot(
        model,
        profile,
        role=role,
        rank_offset=rank_offset,
        host=host,
        nic_names=nic_names,
        provider_names=provider_names,
    )
    session = transport_session or TransportSession(
        rank=snapshot.endpoint.rank,
        world_size=_world_size_for_execution_slice(execution_slice, snapshot.endpoint.rank),
        host=snapshot.endpoint.host,
        nic_name=execution_slice.selected_nic_name,
        provider_name=execution_slice.selected_provider_name,
    )
    return execute_exchange_plan(snapshot, execution_slice, coordinator, session)
