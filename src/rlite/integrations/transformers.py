"""Native Transformers + FSDP helpers built on top of the generic resharding core."""

from __future__ import annotations

import os
import socket
from dataclasses import replace
from typing import Iterable, Mapping, Optional, Sequence

from rlite.transport import MemoryKind, TransportSession
from rlite.weight_mapping import Framework, resolve_rule, translate_tensor

from .remote import RemoteTopology
from ..resharding import (
    FrameworkRole,
    FrameworkSnapshot,
    LinearSegment,
    ParameterRecord,
    PendingReceive,
    WorkerEndpoint,
    abort_receive,
    commit_receive,
    execute_exchange_plan,
    normalize_expert_canonical_names,
    prepare_receive,
    split_grouped_expert_record,
)

_PENDING_TRANSFORMERS_FSDP_RECEIVES: dict[str, PendingReceive] = {}


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


def _shape_product(shape: Sequence[int]) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return product


def _extract_tensor(parameter_or_buffer):
    return getattr(parameter_or_buffer, "data", parameter_or_buffer)


def _infer_dtype(parameter_or_buffer, buffer) -> str:
    return str(
        getattr(
            parameter_or_buffer,
            "dtype",
            getattr(buffer, "dtype", "uint8"),
        )
    )


def _infer_memory_kind(parameter_or_buffer, buffer, explicit=None):
    if explicit is not None:
        return explicit
    device = getattr(parameter_or_buffer, "device", getattr(buffer, "device", "cpu"))
    return MemoryKind.CUDA if str(device).startswith("cuda") else MemoryKind.CPU


def _infer_rank(model, default: int = 0) -> int:
    for attr in ("global_rank", "rank", "distributed_rank", "data_parallel_rank"):
        value = getattr(model, attr, None)
        if isinstance(value, int):
            return value
    return int(default)


def _infer_world_size(model, default: int = 1) -> int:
    for attr in ("world_size", "fsdp_world_size", "data_parallel_size"):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
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
    base_rank = _infer_rank(model, 0)
    parameter = _first_parameter(model)
    device_id = None
    if parameter is not None:
        device = getattr(parameter, "device", None)
        device_id = getattr(device, "index", None)
    dp_size = _infer_world_size(model, 1)
    dp_rank = int(getattr(model, "data_parallel_rank", base_rank))
    return WorkerEndpoint(
        rank=base_rank + int(rank_offset),
        framework=Framework.TRANSFORMERS.value,
        role=role,
        host=host or socket.gethostname(),
        process_id=os.getpid(),
        device_id=device_id,
        nic_names=tuple(nic_names),
        provider_names=tuple(provider_names),
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        expert_parallel_rank=0,
        expert_parallel_size=1,
        moe_tensor_parallel_rank=0,
        moe_tensor_parallel_size=1,
        pipeline_parallel_rank=0,
        pipeline_parallel_size=1,
        data_parallel_rank=dp_rank,
        data_parallel_size=max(1, dp_size),
    )


def _normalize_linear_segments(info, item_size: int) -> tuple[LinearSegment, ...]:
    segments = info.get("linear_segments", ())
    normalized = []
    for segment in segments:
        normalized.append(segment if isinstance(segment, LinearSegment) else LinearSegment(*segment))
    if normalized:
        return tuple(normalized)

    logical_start = info.get("logical_start")
    logical_stop = info.get("logical_stop")
    if logical_start is not None and logical_stop is not None:
        logical_start = int(logical_start)
        logical_stop = int(logical_stop)
        return (
            LinearSegment(
                logical_start=logical_start,
                logical_stop=logical_stop,
                byte_offset=int(info.get("byte_offset", 0)),
                byte_length=(logical_stop - logical_start) * item_size,
            ),
        )
    return ()


def _shard_extent(total: int, parts: int, index: int) -> tuple[int, int]:
    base = total // parts
    remainder = total % parts
    length = base + (1 if index < remainder else 0)
    start = index * base + min(index, remainder)
    return start, length


def _is_dtensor_like(value) -> bool:
    return hasattr(value, "to_local") and hasattr(value, "placements")


def _is_shard0_placement(placement) -> bool:
    dim = getattr(placement, "dim", None)
    if dim == 0:
        return True
    text = str(placement)
    return "Shard(0)" in text or "Shard(dim=0)" in text


def _infer_fsdp2_layout(model, parameter):
    buffer = _extract_tensor(parameter)
    if not _is_dtensor_like(buffer):
        return None

    placements = tuple(getattr(buffer, "placements", ()))
    if len(placements) != 1 or not _is_shard0_placement(placements[0]):
        raise ValueError("Transformers FSDP2 only supports a single default Shard(0) placement in v1.")

    device_mesh = getattr(buffer, "device_mesh", None)
    mesh_shape = tuple(int(value) for value in getattr(device_mesh, "shape", ()) or ())
    if len(mesh_shape) > 1:
        raise ValueError("Transformers FSDP2 only supports a single mesh dimension in v1.")

    local_tensor = buffer.to_local()
    logical_shape = tuple(int(value) for value in getattr(parameter, "shape", ()))
    local_shape = tuple(int(value) for value in getattr(local_tensor, "shape", ()))
    if not logical_shape:
        raise ValueError("Transformers FSDP2 does not support scalar parameters in RLite v1.")

    dp_size = mesh_shape[0] if mesh_shape else _infer_world_size(model, 1)
    dp_rank = int(getattr(model, "data_parallel_rank", _infer_rank(model, 0)))
    shard_start, shard_length = _shard_extent(logical_shape[0], max(1, dp_size), dp_rank)
    if shard_length != local_shape[0]:
        raise ValueError(
            "Transformers FSDP2 local shard shape does not match default Shard(0) chunking; "
            "custom placements are not supported in RLite v1."
        )
    suffix = _shape_product(logical_shape[1:])
    logical_start = shard_start * suffix
    logical_stop = logical_start + _shape_product(local_shape)
    byte_length = _shape_product(local_shape) * int(local_tensor.element_size())
    return {
        "fsdp_variant": "fsdp2",
        "tensor": local_tensor,
        "logical_shape": logical_shape,
        "local_shape": local_shape,
        "actual_shape": local_shape,
        "linear_segments": (
            LinearSegment(
                logical_start=logical_start,
                logical_stop=logical_stop,
                byte_offset=0,
                byte_length=byte_length,
            ),
        ),
        "mesh_ndim": len(mesh_shape) or 1,
        "placements": placements,
    }


def _iter_fsdp_parameter_infos(model):
    provider = getattr(model, "rlite_fsdp_param_infos", None)
    if callable(provider):
        provided = tuple(provider())
        if provided:
            for info in provided:
                yield dict(info)
            return

    for name, parameter in model.named_parameters(recurse=True):
        layout = getattr(parameter, "rlite_fsdp_layout", None)
        if layout is None:
            layout = _infer_fsdp2_layout(model, parameter)
        if layout is None:
            raise ValueError(
                "Unable to infer shard-native Transformers FSDP layout. Provide "
                "`model.rlite_fsdp_param_infos()` or per-parameter `rlite_fsdp_layout` metadata."
            )
        yield {"name": name, "parameter": parameter, **dict(layout)}


def _validate_fsdp_layout(info, rule, framework_name: str) -> None:
    variant = str(info.get("fsdp_variant", "")).lower()
    if variant == "fsdp1" and not bool(info.get("use_orig_params", False)):
        raise ValueError(
            "Transformers FSDP1 requires `use_orig_params=True` for shard-native RLite exchange."
        )
    if variant == "fsdp2":
        mesh_ndim = int(info.get("mesh_ndim", 1) or 1)
        if mesh_ndim > 1:
            raise ValueError("Transformers FSDP2 only supports a single mesh dimension in RLite v1.")
        placements = tuple(info.get("placements", ()))
        if placements and (len(placements) != 1 or not _is_shard0_placement(placements[0])):
            raise ValueError(
                "Transformers FSDP2 only supports a single default Shard(0) placement in RLite v1."
            )
    if rule.transpose or rule.reshape:
        raise ValueError(
            f"Transformers FSDP shard-local exchange does not support transpose/reshape rule "
            f"for {framework_name!r} without gathering full parameters."
        )


def collect_transformers_fsdp_snapshot(
    model,
    profile,
    role: FrameworkRole | str = FrameworkRole.BIDIRECTIONAL,
    *,
    rank_offset: int = 0,
    host: str | None = None,
    nic_names: Iterable[str] = (),
    provider_names: Iterable[str] = (),
) -> FrameworkSnapshot:
    """Collect a shard-native snapshot from a Transformers model under FSDP."""

    endpoint = _make_endpoint(
        model,
        role=role,
        rank_offset=rank_offset,
        host=host,
        nic_names=nic_names,
        provider_names=provider_names,
    )

    records = []
    for info in _iter_fsdp_parameter_infos(model):
        framework_name = str(info["name"])
        parameter = info.get("parameter")
        tensor = info.get("tensor") or info.get("buffer")
        if tensor is None and parameter is not None:
            tensor = _extract_tensor(parameter)
        if tensor is None:
            raise ValueError(f"Transformers FSDP parameter {framework_name!r} is missing tensor storage.")

        mapped = translate_tensor(
            framework_name,
            Framework.TRANSFORMERS,
            Framework.TRANSFORMERS,
            profile,
            view="local_shard",
        )
        rule = resolve_rule(framework_name, Framework.TRANSFORMERS, profile)
        _validate_fsdp_layout(info, rule, framework_name)

        logical_shape = tuple(
            int(value)
            for value in (
                info.get("logical_shape")
                or mapped.source_logical_shape
                or getattr(parameter, "shape", getattr(tensor, "shape", ()))
            )
        )
        actual_shape = tuple(
            int(value)
            for value in (
                info.get("actual_shape")
                or getattr(tensor, "shape", getattr(parameter, "shape", ()))
                or ()
            )
        )
        local_shape = tuple(
            int(value)
            for value in (
                info.get("local_shape")
                or actual_shape
                or mapped.source_local_shape
                or logical_shape
            )
        )
        dtype = str(info.get("dtype", _infer_dtype(parameter or tensor, tensor)))
        item_size = int(getattr(tensor, "element_size", lambda: 1)())
        linear_segments = _normalize_linear_segments(info, item_size)
        if linear_segments and all(segment.byte_length == 0 for segment in linear_segments):
            continue
        if linear_segments and mapped.parallel.expert_axis is not None:
            raise ValueError(
                f"Transformers FSDP shard-local exchange does not support grouped expert tensor "
                f"{framework_name!r} without explicit per-expert shard metadata."
            )

        component_sizes = ()
        if mapped.packing.components and mapped.packing.axis is not None:
            logical_sizes = mapped.packing.component_sizes(profile)
            component_sizes = tuple(logical_sizes[component] for component in mapped.packing.components)

        record = ParameterRecord(
            record_id=framework_name,
            framework_name=framework_name,
            tensor=tensor,
            dtype=dtype,
            logical_shape=logical_shape,
            local_shape=local_shape,
            actual_shape=actual_shape or local_shape,
            canonical_names=mapped.canonical_names,
            packing=mapped.packing,
            parallel=mapped.parallel,
            tensor_role=mapped.tensor_role,
            memory_kind=_infer_memory_kind(parameter or tensor, tensor, info.get("memory_kind")),
            component_logical_sizes=component_sizes,
            component_local_sizes=(),
            linear_segments=linear_segments,
            transpose=rule.transpose,
            reshape=rule.reshape,
            match_groups=mapped.match_groups,
            metadata={
                **dict(info.get("metadata", {})),
                "fsdp_variant": str(info.get("fsdp_variant", "")),
            },
        )
        normalized = normalize_expert_canonical_names(record, endpoint)
        records.extend(split_grouped_expert_record(normalized, endpoint))

    return FrameworkSnapshot(endpoint=endpoint, records=tuple(records))


def _world_size_for_execution_slice(execution_slice, local_rank: int) -> int:
    ranks = {int(local_rank), *[int(rank) for rank in execution_slice.expected_source_ranks]}
    for task in execution_slice.send_tasks:
        ranks.add(int(task.src_rank))
        ranks.add(int(task.dst_rank))
    return max(ranks) + 1 if ranks else int(local_rank) + 1


def execute_transformers_fsdp_exchange(
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
    """Collect a live Transformers-FSDP snapshot and execute one local exchange slice."""

    snapshot = collect_transformers_fsdp_snapshot(
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


def prepare_transformers_fsdp_receive(
    model,
    profile,
    execution_slice,
    *,
    transport_session: Optional[TransportSession] = None,
    rank_offset: int = 0,
    host: str | None = None,
    nic_names: Iterable[str] = (),
    provider_names: Iterable[str] = (),
) -> PendingReceive:
    """Prepare a Transformers FSDP model to receive remotely copied weights."""

    snapshot = collect_transformers_fsdp_snapshot(
        model,
        profile,
        role=FrameworkRole.TARGET,
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
    return prepare_receive(snapshot, execution_slice, session)


def commit_transformers_fsdp_receive(pending: PendingReceive):
    return commit_receive(pending)


def abort_transformers_fsdp_receive(pending: PendingReceive):
    return abort_receive(pending)


def store_pending_transformers_fsdp_receive(request_id: str, pending: PendingReceive) -> None:
    _PENDING_TRANSFORMERS_FSDP_RECEIVES[str(request_id)] = pending


def commit_pending_transformers_fsdp_receive(request_id: str):
    return commit_transformers_fsdp_receive(_PENDING_TRANSFORMERS_FSDP_RECEIVES.pop(str(request_id)))


def abort_pending_transformers_fsdp_receive(request_id: str):
    return abort_transformers_fsdp_receive(_PENDING_TRANSFORMERS_FSDP_RECEIVES.pop(str(request_id)))


def synthesize_transformers_fsdp_target_snapshots(
    source_snapshots: FrameworkSnapshot | Iterable[FrameworkSnapshot],
    target_profile,
    topology: RemoteTopology,
) -> tuple[FrameworkSnapshot, ...]:
    """Synthesize target snapshots for a shard-native Transformers+FSDP topology."""

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
            source_framework = Framework(snapshot.framework)
            for record in snapshot.records:
                mapped = translate_tensor(
                    record.framework_name,
                    source_framework,
                    Framework.TRANSFORMERS,
                    worker_profile,
                    view="local_shard",
                )
                for target in mapped.targets:
                    rule = resolve_rule(target.name, Framework.TRANSFORMERS, worker_profile)
                    if rule.transpose or rule.reshape:
                        raise ValueError(
                            f"Transformers FSDP shard-local exchange does not support transpose/reshape rule "
                            f"for {target.name!r} without gathering full parameters."
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
                            component_logical_sizes=record.component_logical_sizes,
                            component_local_sizes=(),
                            metadata={},
                        ),
                    )
        endpoint = worker.endpoint(role=FrameworkRole.TARGET)
        records = []
        for blueprint in target_blueprints.values():
            normalized = normalize_expert_canonical_names(blueprint, endpoint)
            records.extend(split_grouped_expert_record(normalized, endpoint))
        synthesized.append(
            FrameworkSnapshot(
                endpoint=endpoint,
                records=tuple(records),
            )
        )
    return tuple(synthesized)
