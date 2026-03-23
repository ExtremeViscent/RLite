"""Centralized planning for framework-to-framework weight exchange."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from rlite.transport import MemoryKind, TransferPath, TransferTask

from .types import (
    BindingKind,
    ExchangeBundle,
    ExchangePlan,
    ExecutionSlice,
    FrameworkSnapshot,
    LinearSegment,
    LocalityTier,
    ParameterRecord,
    TensorBindingManifest,
    TopologyDecision,
    TopologyPolicy,
    WorkerEndpoint,
)


_EXPERT_RE = re.compile(r"(\.experts\.)(\d+)(\.)")


def _normalize_snapshots(
    snapshots: FrameworkSnapshot | Iterable[FrameworkSnapshot],
) -> tuple[FrameworkSnapshot, ...]:
    if isinstance(snapshots, FrameworkSnapshot):
        return (snapshots,)
    return tuple(snapshots)


def _exchange_key(canonical_names: Sequence[str]) -> str:
    seed = "\n".join(canonical_names).encode("utf-8")
    digest = hashlib.sha1(seed).hexdigest()[:12]
    first = canonical_names[0].replace(".", "_")
    return f"rlite_{first}_{digest}"


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted((int(start), int(stop)) for start, stop in intervals if int(stop) > int(start))
    if not ordered:
        return ()

    merged = [ordered[0]]
    for start, stop in ordered[1:]:
        prev_start, prev_stop = merged[-1]
        if start <= prev_stop:
            merged[-1] = (prev_start, max(prev_stop, stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def _intersect_intervals(
    left: Iterable[tuple[int, int]],
    right: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    intersections = []
    for left_start, left_stop in left:
        for right_start, right_stop in right:
            start = max(left_start, right_start)
            stop = min(left_stop, right_stop)
            if stop > start:
                intersections.append((start, stop))
    return _merge_intervals(intersections)


def _subtract_covered_intervals(
    uncovered: Iterable[tuple[int, int]],
    covered: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    remaining = list(_merge_intervals(uncovered))
    for covered_start, covered_stop in _merge_intervals(covered):
        updated: list[tuple[int, int]] = []
        for start, stop in remaining:
            if covered_stop <= start or stop <= covered_start:
                updated.append((start, stop))
                continue
            if start < covered_start:
                updated.append((start, covered_start))
            if covered_stop < stop:
                updated.append((covered_stop, stop))
        remaining = updated
    return tuple(remaining)


def _shard_slices_for_endpoint(
    endpoint: WorkerEndpoint,
    logical_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
    shard_axis: int | None,
    is_moe_sharded: bool,
) -> tuple[tuple[int, int], ...]:
    slices = []
    for index, (logical_dim, local_dim) in enumerate(zip(logical_shape, local_shape)):
        if index == shard_axis and local_dim < logical_dim:
            if is_moe_sharded:
                shard_rank = endpoint.moe_tensor_parallel_rank
            else:
                shard_rank = endpoint.tensor_parallel_rank
            start = shard_rank * local_dim
            slices.append((start, start + local_dim))
        else:
            slices.append((0, local_dim))
    return tuple(slices)


def _candidate_locality(
    source: TensorBindingManifest,
    target: TensorBindingManifest,
    source_endpoint: WorkerEndpoint,
    target_endpoint: WorkerEndpoint,
    policy: TopologyPolicy,
) -> tuple[LocalityTier, TransferPath]:
    forced = policy.forced_paths.get((source.rank, target.rank))
    if forced is not None:
        return (LocalityTier.CROSS_HOST_RDMA, forced)

    same_host = bool(source_endpoint.host) and source_endpoint.host == target_endpoint.host
    same_process = (
        same_host
        and int(source_endpoint.process_id or 0) != 0
        and int(source_endpoint.process_id) == int(target_endpoint.process_id or -1)
    )
    if source.rank == target.rank and same_process:
        return (LocalityTier.LOCAL_EXACT, TransferPath.MEMCPY)

    both_cuda = (
        source.memory_kind is MemoryKind.CUDA
        and target.memory_kind is MemoryKind.CUDA
    )
    if same_host and policy.prefer_same_host:
        return (
            LocalityTier.SAME_HOST_DIRECT,
            TransferPath.CUDA_IPC if both_cuda else TransferPath.MEMCPY,
        )

    if source_endpoint.provider_names or target_endpoint.provider_names:
        return (LocalityTier.CROSS_HOST_RDMA, TransferPath.LIBFABRIC_RMA)

    return (LocalityTier.STAGED_FALLBACK, TransferPath.STAGED_HOST)


def _locality_score(
    tier: LocalityTier,
    transferred_bytes: int,
    source_load_bytes: int,
    source_endpoint: WorkerEndpoint,
    target_endpoint: WorkerEndpoint,
    policy: TopologyPolicy,
) -> tuple[int, int, int, int]:
    if tier is LocalityTier.LOCAL_EXACT:
        locality = 0 if policy.prefer_local else policy.local_bonus
    elif tier is LocalityTier.SAME_HOST_DIRECT:
        locality = policy.same_host_bonus
    elif tier is LocalityTier.CROSS_HOST_RDMA:
        locality = policy.rdma_bonus
    else:
        locality = policy.staged_penalty
    same_host_bias = 0 if source_endpoint.host == target_endpoint.host else 1
    return (locality, source_load_bytes, same_host_bias, transferred_bytes)


def _rewrite_expert_indices(
    canonical_names: tuple[str, ...],
    expert_offset: int,
) -> tuple[str, ...]:
    if expert_offset == 0:
        return canonical_names
    rewritten = []
    for name in canonical_names:
        rewritten.append(
            _EXPERT_RE.sub(
                lambda match: f"{match.group(1)}{int(match.group(2)) + expert_offset}{match.group(3)}",
                name,
            )
        )
    return tuple(rewritten)


def _shift_axis(axis: int | None, dropped_axis: int) -> int | None:
    if axis is None:
        return None
    if axis == dropped_axis:
        return None
    if axis > dropped_axis:
        return axis - 1
    return axis


def _component_sizes_for_record(record: ParameterRecord) -> tuple[int, ...]:
    if record.component_local_sizes:
        return record.component_local_sizes
    if len(record.canonical_names) == 1:
        axis = record.packing.axis if record.packing.axis is not None else 0
        return (record.local_shape[axis],)
    raise ValueError(
        f"Record {record.record_id!r} does not expose component sizes for packed materialization."
    )


def _component_logical_sizes_for_record(record: ParameterRecord) -> tuple[int, ...]:
    if record.component_logical_sizes:
        return record.component_logical_sizes
    if len(record.canonical_names) == 1:
        return (_shape_product(record.logical_shape),)
    raise ValueError(
        f"Record {record.record_id!r} does not expose component logical sizes for packed materialization."
    )


def _canonical_flattened_intervals(record: ParameterRecord) -> tuple[tuple[tuple[int, int], ...], ...]:
    if len(record.canonical_names) == 1:
        total = _shape_product(record.logical_shape)
        return (((0, total),),)

    if record.packing.axis is None:
        raise ValueError(
            f"Record {record.record_id!r} has multiple canonicals but no packing axis."
        )

    axis = record.packing.axis
    component_sizes = _component_logical_sizes_for_record(record)
    prefix = math.prod(record.logical_shape[:axis]) if axis > 0 else 1
    suffix = math.prod(record.logical_shape[axis + 1 :]) if axis + 1 < len(record.logical_shape) else 1
    full_axis_extent = record.logical_shape[axis] * suffix

    intervals = []
    cursor = 0
    for size in component_sizes:
        component_intervals = []
        component_extent = size * suffix
        for prefix_index in range(prefix):
            base = prefix_index * full_axis_extent
            start = base + cursor * suffix
            component_intervals.append((start, start + component_extent))
        intervals.append(tuple(component_intervals))
        cursor += size
    return tuple(intervals)


def _clip_linear_segments(
    segments: Sequence[LinearSegment],
    intervals: Iterable[tuple[int, int]],
    item_size: int,
) -> tuple[LinearSegment, ...]:
    clipped = []
    for segment in segments:
        for start, stop in intervals:
            overlap_start = max(segment.logical_start, start)
            overlap_stop = min(segment.logical_stop, stop)
            if overlap_stop <= overlap_start:
                continue
            byte_offset = segment.byte_offset + (overlap_start - segment.logical_start) * item_size
            byte_length = (overlap_stop - overlap_start) * item_size
            clipped.append(
                LinearSegment(
                    logical_start=overlap_start,
                    logical_stop=overlap_stop,
                    byte_offset=byte_offset,
                    byte_length=byte_length,
                )
            )
    return tuple(clipped)


def _manifest_coverage(manifest: TensorBindingManifest) -> tuple[tuple[int, int], ...]:
    return _merge_intervals(
        (segment.logical_start, segment.logical_stop) for segment in manifest.linear_segments
    )


def _make_record(
    *,
    record_id: str,
    framework_name: str,
    tensor,
    dtype: str,
    logical_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
    actual_shape: tuple[int, ...],
    canonical_names: tuple[str, ...],
    packing,
    parallel,
    tensor_role: str,
    memory_kind,
    component_logical_sizes: tuple[int, ...],
    component_local_sizes: tuple[int, ...],
    transpose: bool,
    reshape,
    match_groups: Mapping[str, str],
    metadata: Mapping[str, str],
) -> ParameterRecord:
    return ParameterRecord(
        record_id=record_id,
        framework_name=framework_name,
        tensor=tensor,
        dtype=dtype,
        logical_shape=logical_shape,
        local_shape=local_shape,
        actual_shape=actual_shape,
        canonical_names=canonical_names,
        packing=packing,
        parallel=parallel,
        tensor_role=tensor_role,
        memory_kind=memory_kind,
        component_logical_sizes=component_logical_sizes,
        component_local_sizes=component_local_sizes,
        transpose=transpose,
        reshape=reshape,
        match_groups=match_groups,
        metadata=metadata,
    )


def split_grouped_expert_record(
    record: ParameterRecord,
    endpoint: WorkerEndpoint,
) -> tuple[ParameterRecord, ...]:
    """Split a grouped-expert tensor into one record per local expert."""

    expert_axis = record.parallel.expert_axis
    if expert_axis is None or expert_axis >= len(record.local_shape):
        return (record,)
    if record.local_shape[expert_axis] >= record.logical_shape[expert_axis]:
        return (record,)

    local_experts = record.local_shape[expert_axis]
    if local_experts <= 0:
        return (record,)

    components_per_expert = max(1, len(record.canonical_names) // record.logical_shape[expert_axis])
    expert_offset = endpoint.expert_parallel_rank * local_experts
    split_records = []
    for local_expert_index in range(local_experts):
        global_expert = expert_offset + local_expert_index
        start = global_expert * components_per_expert
        end = start + components_per_expert
        canonical_names = record.canonical_names[start:end]
        if not canonical_names:
            continue

        new_tensor = record.tensor[local_expert_index]
        new_logical_shape = tuple(
            dim for index, dim in enumerate(record.logical_shape) if index != expert_axis
        )
        new_local_shape = tuple(
            dim for index, dim in enumerate(record.local_shape) if index != expert_axis
        )
        new_actual_shape = tuple(
            dim for index, dim in enumerate(record.actual_shape) if index != expert_axis
        )
        new_packing = replace(record.packing, axis=_shift_axis(record.packing.axis, expert_axis))
        new_parallel = replace(
            record.parallel,
            shard_axis=_shift_axis(record.parallel.shard_axis, expert_axis),
            expert_axis=None,
        )
        split_records.append(
            _make_record(
                record_id=f"{record.record_id}#expert{global_expert}",
                framework_name=record.framework_name,
                tensor=new_tensor,
                dtype=record.dtype,
                logical_shape=new_logical_shape,
                local_shape=new_local_shape,
                actual_shape=new_actual_shape,
                canonical_names=canonical_names,
                packing=new_packing,
                parallel=new_parallel,
                tensor_role=record.tensor_role,
                memory_kind=record.memory_kind,
                component_logical_sizes=record.component_logical_sizes,
                component_local_sizes=record.component_local_sizes,
                transpose=record.transpose,
                reshape=record.reshape,
                match_groups=record.match_groups,
                metadata={**record.metadata, "global_expert_index": str(global_expert)},
            )
        )

    return tuple(split_records) or (record,)


def normalize_expert_canonical_names(
    record: ParameterRecord,
    endpoint: WorkerEndpoint,
) -> ParameterRecord:
    """Rewrite per-expert canonical names from local to global expert indices."""

    if endpoint.expert_parallel_size <= 1:
        return record
    if record.parallel.expert_axis is not None:
        return record
    local_experts = max(1, endpoint.expert_parallel_size)
    if ".experts." not in "\n".join(record.canonical_names):
        return record
    try:
        experts_per_rank = max(1, int(endpoint.metadata.get("experts_per_rank", "0")))
    except (TypeError, ValueError):
        experts_per_rank = 1
    try:
        expert_offset = endpoint.expert_parallel_rank * experts_per_rank
    except ValueError:
        expert_offset = 0
    if expert_offset == 0:
        return record
    return _make_record(
        record_id=record.record_id,
        framework_name=record.framework_name,
        tensor=record.tensor,
        dtype=record.dtype,
        logical_shape=record.logical_shape,
        local_shape=record.local_shape,
        actual_shape=record.actual_shape,
        canonical_names=_rewrite_expert_indices(record.canonical_names, expert_offset),
        packing=record.packing,
        parallel=record.parallel,
        tensor_role=record.tensor_role,
        memory_kind=record.memory_kind,
        component_logical_sizes=record.component_logical_sizes,
        component_local_sizes=record.component_local_sizes,
        transpose=record.transpose,
        reshape=record.reshape,
        match_groups=record.match_groups,
        metadata=record.metadata,
    )


def build_binding_manifest(
    record: ParameterRecord,
    endpoint: WorkerEndpoint,
    canonical_names: tuple[str, ...],
) -> TensorBindingManifest:
    """Create a serializable binding candidate for one record unit."""

    try:
        start = record.canonical_names.index(canonical_names[0])
    except ValueError as exc:
        raise KeyError(
            f"Record {record.record_id!r} does not contain canonical {canonical_names[0]!r}."
        ) from exc
    end = start + len(canonical_names)
    if record.canonical_names[start:end] != canonical_names:
        raise ValueError(
            f"Record {record.record_id!r} cannot materialize non-contiguous canonical bundle "
            f"{canonical_names!r}."
        )

    full_unit = len(canonical_names) == len(record.canonical_names)
    if full_unit:
        logical_shape = record.logical_shape
        local_shape = record.local_shape
    else:
        if record.packing.axis is None:
            raise ValueError(
                f"Record {record.record_id!r} has no packing axis for subset {canonical_names!r}."
            )
        axis = record.packing.axis
        logical_dim = sum(_component_logical_sizes_for_record(record)[start:end])
        logical_shape = list(record.logical_shape)
        logical_shape[axis] = logical_dim
        logical_shape = tuple(logical_shape)
        if record.linear_segments:
            component_intervals = _canonical_flattened_intervals(record)[start:end]
            selected_intervals = _merge_intervals(
                interval
                for component in component_intervals
                for interval in component
            )
            selected_segments = _clip_linear_segments(record.linear_segments, selected_intervals, record.item_size)
            if not selected_segments:
                raise ValueError(
                    f"Record {record.record_id!r} has no local storage coverage for canonical bundle "
                    f"{canonical_names!r}."
                )
            subset_numel = sum(segment.numel for segment in selected_segments)
            local_shape = (subset_numel,)
        else:
            local_dim = sum(_component_sizes_for_record(record)[start:end])
            local_shape = list(record.local_shape)
            local_shape[axis] = local_dim
            local_shape = tuple(local_shape)

    direct_subset = (
        full_unit
        or record.packing.axis in (None, 0)
    )
    binding_kind = BindingKind.DIRECT if direct_subset and not (record.transpose or record.reshape) else BindingKind.STAGED
    if record.linear_segments and binding_kind is BindingKind.DIRECT:
        if full_unit:
            linear_segments = record.linear_segments
        else:
            component_intervals = _canonical_flattened_intervals(record)[start:end]
            selected_intervals = _merge_intervals(
                interval
                for component in component_intervals
                for interval in component
            )
            linear_segments = _clip_linear_segments(record.linear_segments, selected_intervals, record.item_size)
        logical_slices = tuple((0, dim) for dim in logical_shape)
        shard_axis = None
    else:
        is_moe_sharded = record.parallel.kind.value.startswith("tp_ep")
        logical_slices = _shard_slices_for_endpoint(
            endpoint,
            logical_shape,
            local_shape,
            record.parallel.shard_axis,
            is_moe_sharded=is_moe_sharded,
        )
        linear_segments = ()
        shard_axis = record.parallel.shard_axis
    return TensorBindingManifest(
        binding_id=f"{endpoint.rank}:{record.record_id}:{start}:{end}",
        record_id=record.record_id,
        rank=endpoint.rank,
        exchange_key=_exchange_key(canonical_names),
        canonical_names=canonical_names,
        framework_name=endpoint.framework,
        framework_tensor_name=record.framework_name,
        binding_kind=binding_kind,
        memory_kind=record.memory_kind,
        dtype=record.dtype,
        logical_shape=logical_shape,
        local_shape=local_shape,
        logical_slices=logical_slices,
        linear_segments=linear_segments,
        component_start=start,
        component_end=end,
        shard_axis=shard_axis,
        metadata={
            **record.metadata,
            "tensor_role": record.tensor_role,
            "binding_kind": binding_kind.value,
        },
    )


def _collect_manifest_candidates(
    snapshots: tuple[FrameworkSnapshot, ...],
) -> tuple[
    dict[tuple[str, ...], list[TensorBindingManifest]],
    dict[str, list[TensorBindingManifest]],
    dict[str, ParameterRecord],
    dict[int, WorkerEndpoint],
]:
    full_units: dict[tuple[str, ...], list[TensorBindingManifest]] = defaultdict(list)
    singleton_units: dict[str, list[TensorBindingManifest]] = defaultdict(list)
    records_by_id: dict[str, ParameterRecord] = {}
    endpoints_by_rank: dict[int, WorkerEndpoint] = {}
    for snapshot in snapshots:
        endpoints_by_rank[snapshot.endpoint.rank] = snapshot.endpoint
        for record in snapshot.records:
            records_by_id[record.record_id] = record
            full_manifest = build_binding_manifest(record, snapshot.endpoint, record.canonical_names)
            full_units[full_manifest.canonical_names].append(full_manifest)
            if len(record.canonical_names) > 1:
                for canonical in record.canonical_names:
                    try:
                        singleton = build_binding_manifest(record, snapshot.endpoint, (canonical,))
                    except ValueError as exc:
                        if "no local storage coverage" not in str(exc):
                            raise
                    else:
                        singleton_units[canonical].append(singleton)
            else:
                singleton_units[record.canonical_names[0]].append(full_manifest)
    return full_units, singleton_units, records_by_id, endpoints_by_rank


def _choose_source_manifest(
    sources: Sequence[TensorBindingManifest],
    target: TensorBindingManifest,
    source_load_bytes: Mapping[int, int],
    source_endpoints: Mapping[int, WorkerEndpoint],
    target_endpoints: Mapping[int, WorkerEndpoint],
    policy: TopologyPolicy,
) -> tuple[TensorBindingManifest, TopologyDecision, int]:
    if not sources:
        raise ValueError(f"No source candidates available for {target.canonical_names!r}.")
    best_source = None
    best_decision = None
    best_score = None
    best_transfer_bytes = 0
    target_endpoint = target_endpoints[target.rank]
    for source in sources:
        preview_tasks = _build_transfer_tasks(
            source,
            target,
            preferred_path=TransferPath.MEMCPY,
        )
        transfer_bytes = sum(task.num_bytes for task in preview_tasks)
        if transfer_bytes <= 0:
            continue
        source_endpoint = source_endpoints[source.rank]
        locality, preferred_path = _candidate_locality(
            source,
            target,
            source_endpoint,
            target_endpoint,
            policy,
        )
        score = _locality_score(
            locality,
            transfer_bytes,
            source_load_bytes.get(source.rank, 0),
            source_endpoint,
            target_endpoint,
            policy,
        )
        decision = TopologyDecision(
            src_rank=source.rank,
            dst_rank=target.rank,
            locality_tier=locality,
            preferred_path=preferred_path,
            source_nic_name=(source_endpoint.nic_names[0] if source_endpoint.nic_names else ""),
            target_nic_name=(target_endpoint.nic_names[0] if target_endpoint.nic_names else ""),
            source_provider_name=(
                source_endpoint.provider_names[0] if source_endpoint.provider_names else ""
            ),
            target_provider_name=(
                target_endpoint.provider_names[0] if target_endpoint.provider_names else ""
            ),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_source = source
            best_decision = decision
            best_transfer_bytes = transfer_bytes
    if best_source is None or best_decision is None:
        raise ValueError(
            f"No overlapping source candidates are available for {target.canonical_names!r} on rank {target.rank}."
        )
    return best_source, best_decision, best_transfer_bytes


def _select_source_manifests(
    sources: Sequence[TensorBindingManifest],
    target: TensorBindingManifest,
    source_load_bytes: Mapping[int, int],
    source_endpoints: Mapping[int, WorkerEndpoint],
    target_endpoints: Mapping[int, WorkerEndpoint],
    policy: TopologyPolicy,
) -> tuple[tuple[tuple[TensorBindingManifest, TopologyDecision, tuple[TransferTask, ...]], ...], int]:
    if not sources:
        raise ValueError(f"No source candidates available for {target.canonical_names!r}.")

    uncovered = list(_manifest_coverage(target))
    if not uncovered:
        return ((), 0)

    selected: list[tuple[TensorBindingManifest, TopologyDecision, tuple[TransferTask, ...], tuple[tuple[int, int], ...]]] = []
    remaining = list(sources)
    total_bytes = 0

    while uncovered:
        best = None
        best_score = None
        best_bytes = 0
        for source in remaining:
            source_coverage = _manifest_coverage(source)
            overlap = _intersect_intervals(source_coverage, uncovered)
            overlap_elems = sum(stop - start for start, stop in overlap)
            if overlap_elems <= 0:
                continue

            source_endpoint = source_endpoints[source.rank]
            target_endpoint = target_endpoints[target.rank]
            locality, preferred_path = _candidate_locality(
                source,
                target,
                source_endpoint,
                target_endpoint,
                policy,
            )
            tasks = _build_transfer_tasks(source, target, preferred_path)
            transfer_bytes = sum(task.num_bytes for task in tasks)
            if transfer_bytes <= 0:
                continue
            score = (
                *_locality_score(
                    locality,
                    transfer_bytes,
                    source_load_bytes.get(source.rank, 0),
                    source_endpoint,
                    target_endpoint,
                    policy,
                ),
                -overlap_elems,
            )
            decision = TopologyDecision(
                src_rank=source.rank,
                dst_rank=target.rank,
                locality_tier=locality,
                preferred_path=preferred_path,
                source_nic_name=(source_endpoint.nic_names[0] if source_endpoint.nic_names else ""),
                target_nic_name=(target_endpoint.nic_names[0] if target_endpoint.nic_names else ""),
                source_provider_name=(
                    source_endpoint.provider_names[0] if source_endpoint.provider_names else ""
                ),
                target_provider_name=(
                    target_endpoint.provider_names[0] if target_endpoint.provider_names else ""
                ),
            )
            if best_score is None or score < best_score:
                best = (source, decision, tasks, overlap)
                best_score = score
                best_bytes = transfer_bytes

        if best is None:
            raise ValueError(
                f"Could not cover target logical intervals {tuple(uncovered)} for {target.canonical_names!r} "
                f"with source candidates on rank {target.rank}."
            )
        selected.append(best)
        remaining = [source for source in remaining if source.binding_id != best[0].binding_id]
        uncovered = list(_subtract_covered_intervals(uncovered, best[3]))
        total_bytes += best_bytes

    return (
        tuple((source, decision, tasks) for source, decision, tasks, _ in selected),
        total_bytes,
    )


def _build_transfer_tasks(
    source: TensorBindingManifest,
    target: TensorBindingManifest,
    preferred_path: TransferPath,
) -> tuple[TransferTask, ...]:
    if source.logical_shape != target.logical_shape:
        raise ValueError(
            f"Logical shape mismatch for exchange key {source.exchange_key!r}: "
            f"{source.logical_shape} vs {target.logical_shape}."
        )
    tasks = []
    for source_segment in source.linear_segments:
        for target_segment in target.linear_segments:
            overlap_start = max(source_segment.logical_start, target_segment.logical_start)
            overlap_stop = min(source_segment.logical_stop, target_segment.logical_stop)
            if overlap_stop <= overlap_start:
                continue
            byte_length = (overlap_stop - overlap_start) * source.item_size
            src_offset = source_segment.byte_offset + (
                overlap_start - source_segment.logical_start
            ) * source.item_size
            dst_offset = target_segment.byte_offset + (
                overlap_start - target_segment.logical_start
            ) * target.item_size
            tasks.append(
                TransferTask(
                    tensor_name=source.exchange_key,
                    src_rank=source.rank,
                    dst_rank=target.rank,
                    src_slice=(src_offset, byte_length),
                    dst_slice=(dst_offset, byte_length),
                    dtype=source.dtype,
                    num_bytes=byte_length,
                    src_mem_kind=source.memory_kind,
                    dst_mem_kind=target.memory_kind,
                    preferred_path=preferred_path,
                )
            )
    return tuple(tasks)


def build_exchange_plan(
    source_snapshot: FrameworkSnapshot | Iterable[FrameworkSnapshot],
    target_snapshot: FrameworkSnapshot | Iterable[FrameworkSnapshot],
    topology_policy: TopologyPolicy | None = None,
) -> ExchangePlan:
    """Build a deterministic exchange plan between source and target workers."""

    policy = topology_policy or TopologyPolicy()
    source_snapshots = _normalize_snapshots(source_snapshot)
    target_snapshots = _normalize_snapshots(target_snapshot)
    if not source_snapshots:
        raise ValueError("source_snapshot must contain at least one snapshot")
    if not target_snapshots:
        raise ValueError("target_snapshot must contain at least one snapshot")

    source_full, source_singletons, source_records, source_endpoints = _collect_manifest_candidates(
        source_snapshots
    )
    target_full, target_singletons, _target_records, target_endpoints = _collect_manifest_candidates(
        target_snapshots
    )

    exact_units = sorted(
        set(source_full).intersection(target_full),
        key=lambda value: (-len(value), value),
    )
    covered_canonicals = {canonical for unit in exact_units for canonical in unit}
    remaining_canonicals = sorted(
        set(target_singletons).difference(covered_canonicals)
    )
    missing = [canonical for canonical in remaining_canonicals if canonical not in source_singletons]
    if missing:
        raise KeyError(
            f"Target canonical tensors are missing from the source snapshot: {missing}"
        )

    bundle_units = [("exact", unit) for unit in exact_units]
    bundle_units.extend(("singleton", (canonical,)) for canonical in remaining_canonicals)

    manifests_by_rank: dict[int, list[TensorBindingManifest]] = defaultdict(list)
    send_tasks_by_rank: dict[int, list[TransferTask]] = defaultdict(list)
    target_binding_ids_by_rank: dict[int, list[str]] = defaultdict(list)
    expected_sources_by_rank: dict[int, set[int]] = defaultdict(set)
    source_load_bytes: dict[int, int] = defaultdict(int)
    topology_decisions: dict[tuple[int, int], TopologyDecision] = {}
    bundles: list[ExchangeBundle] = []

    for mode, unit in bundle_units:
        if mode == "exact":
            source_candidates = tuple(source_full[unit])
            target_candidates = tuple(target_full[unit])
            split_reason = ""
        else:
            canonical = unit[0]
            source_candidates = tuple(source_singletons[canonical])
            target_candidates = tuple(target_singletons[canonical])
            split_reason = "split_to_canonical"

        if not target_candidates:
            continue

        bundles.append(
            ExchangeBundle(
                exchange_key=_exchange_key(unit),
                canonical_names=unit,
                source_record_ids=tuple(candidate.record_id for candidate in source_candidates),
                target_record_ids=tuple(candidate.record_id for candidate in target_candidates),
                source_ranks=tuple(sorted({candidate.rank for candidate in source_candidates})),
                target_ranks=tuple(sorted({candidate.rank for candidate in target_candidates})),
                split_reason=split_reason,
            )
        )

        for target in target_candidates:
            selections, transfer_bytes = _select_source_manifests(
                source_candidates,
                target,
                source_load_bytes,
                source_endpoints,
                target_endpoints,
                policy=policy,
            )
            manifests_by_rank[target.rank].append(target)
            target_binding_ids_by_rank[target.rank].append(target.binding_id)
            for source, decision, tasks in selections:
                source_load_bytes[source.rank] += sum(task.num_bytes for task in tasks)
                topology_decisions[(source.rank, target.rank)] = decision
                manifests_by_rank[source.rank].append(source)
                if source.rank != target.rank:
                    expected_sources_by_rank[target.rank].add(source.rank)
                send_tasks_by_rank[source.rank].extend(tasks)

    execution_slices: dict[int, ExecutionSlice] = {}
    all_ranks = set(manifests_by_rank)
    plan_requires_staging = False
    plan_fallback_bytes = 0
    for rank in all_ranks:
        rank_endpoint = source_endpoints.get(rank) or target_endpoints.get(rank)
        nic_name = ""
        provider_name = ""
        if rank_endpoint is not None:
            pinned_nics = policy.pinned_nics.get(rank, ())
            pinned_providers = policy.pinned_providers.get(rank, ())
            nic_name = (
                pinned_nics[0]
                if pinned_nics
                else (rank_endpoint.nic_names[0] if rank_endpoint.nic_names else "")
            )
            provider_name = (
                pinned_providers[0]
                if pinned_providers
                else (
                    rank_endpoint.provider_names[0]
                    if rank_endpoint.provider_names
                    else ""
                )
            )
        manifests = tuple({manifest.binding_id: manifest for manifest in manifests_by_rank[rank]}.values())
        target_binding_ids = tuple(target_binding_ids_by_rank.get(rank, ()))
        staged_target_manifests = tuple(
            manifest
            for manifest in manifests
            if manifest.binding_id in target_binding_ids and manifest.binding_kind is BindingKind.STAGED
        )
        requires_staging = bool(staged_target_manifests)
        fallback_bytes = sum(manifest.num_bytes for manifest in staged_target_manifests)
        plan_requires_staging = plan_requires_staging or requires_staging
        plan_fallback_bytes += fallback_bytes
        execution_slices[rank] = ExecutionSlice(
            rank=rank,
            binding_manifests=manifests,
            send_tasks=tuple(send_tasks_by_rank.get(rank, ())),
            target_binding_ids=target_binding_ids,
            expected_source_ranks=tuple(expected_sources_by_rank.get(rank, ())),
            selected_nic_name=nic_name,
            selected_provider_name=provider_name,
            metadata={
                "requires_staging": "1" if requires_staging else "0",
                "fallback_bytes": str(fallback_bytes),
            },
        )

    return ExchangePlan(
        source_framework=source_snapshots[0].framework,
        target_framework=target_snapshots[0].framework,
        bundles=tuple(bundles),
        binding_manifests_by_rank={rank: slice_.binding_manifests for rank, slice_ in execution_slices.items()},
        execution_slices=execution_slices,
        topology_decisions=topology_decisions,
        metadata={
            "requires_staging": "1" if plan_requires_staging else "0",
            "fallback_bytes": str(plan_fallback_bytes),
        },
    )
