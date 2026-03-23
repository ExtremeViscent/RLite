"""Local execution helpers for exchange slices."""

from __future__ import annotations

from dataclasses import replace
from typing import AbstractSet, Mapping

from rlite.transport import RankDescriptor, TransportSession, TransferPlan

from .coordinator import ExchangeCoordinator, FrozenExchangeCoordinator
from .types import (
    BindingKind,
    ExchangeResult,
    ExecutionSlice,
    FrameworkSnapshot,
    ParameterRecord,
    TensorBinding,
    TensorBindingManifest,
)


def _is_torch_tensor(value) -> bool:
    return hasattr(value, "detach") and hasattr(value, "contiguous") and hasattr(value, "numel")


def _buffer_num_bytes(value) -> int:
    if _is_torch_tensor(value):
        return int(value.numel()) * int(value.element_size())
    return len(memoryview(value).cast("B"))


def _slice_bytes_for_axis0(
    value,
    actual_shape: tuple[int, ...],
    start: int,
    length: int,
):
    if _is_torch_tensor(value):
        return value.narrow(0, start, length)
    inner = 1
    for dim in actual_shape[1:]:
        inner *= dim
    item_bytes = _buffer_num_bytes(value) // max(1, _shape_product(actual_shape))
    byte_offset = start * inner * item_bytes
    byte_length = length * inner * item_bytes
    return memoryview(value).cast("B")[byte_offset : byte_offset + byte_length]


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return product


def _copy_into(dst, src) -> None:
    if _is_torch_tensor(dst):
        dst.copy_(src)
        return
    dst_view = memoryview(dst).cast("B")
    src_view = memoryview(src).cast("B")
    dst_view[: len(src_view)] = src_view.tobytes()


def _component_axis_offset(record: ParameterRecord, manifest: TensorBindingManifest) -> int:
    if manifest.component_start <= 0:
        return 0
    if record.component_local_sizes:
        return sum(record.component_local_sizes[: manifest.component_start])
    axis = record.packing.axis if record.packing.axis is not None else 0
    if axis != 0:
        raise NotImplementedError(
            f"Cannot infer subset offset for {record.record_id!r} on pack axis {axis}."
        )
    if len(record.canonical_names) <= 0:
        return 0
    if record.local_shape[0] % len(record.canonical_names) != 0:
        raise ValueError(
            f"Record {record.record_id!r} needs component_local_sizes for non-uniform packing."
        )
    return (record.local_shape[0] // len(record.canonical_names)) * manifest.component_start


def _component_axis_length(record: ParameterRecord, manifest: TensorBindingManifest) -> int:
    if record.component_local_sizes:
        return sum(record.component_local_sizes[manifest.component_start : manifest.component_end])
    return manifest.local_shape[0]


def _extract_source_stage(record: ParameterRecord, manifest: TensorBindingManifest):
    if manifest.component_start == 0 and manifest.component_end == len(record.canonical_names):
        if record.transpose:
            if not _is_torch_tensor(record.tensor):
                raise NotImplementedError("Transpose staging requires torch tensors.")
            return record.tensor.transpose(-2, -1).contiguous()
        return record.tensor.detach().contiguous() if _is_torch_tensor(record.tensor) else bytearray(memoryview(record.tensor).cast("B"))
    if record.packing.axis not in (None, 0):
        raise NotImplementedError(
            f"Staged subset extraction requires pack axis 0, got {record.packing.axis}."
        )
    start = _component_axis_offset(record, manifest)
    length = _component_axis_length(record, manifest)
    sliced = _slice_bytes_for_axis0(record.tensor, record.actual_shape, start, length)
    return sliced.detach().contiguous() if _is_torch_tensor(sliced) else bytearray(memoryview(sliced).cast("B"))


def _allocate_target_stage(record: ParameterRecord, manifest: TensorBindingManifest):
    if _is_torch_tensor(record.tensor):
        import torch

        return torch.empty(manifest.local_shape, dtype=record.tensor.dtype, device=record.tensor.device)
    return bytearray(manifest.num_bytes)


def _apply_target_stage(record: ParameterRecord, manifest: TensorBindingManifest, stage_buffer) -> None:
    if manifest.component_start == 0 and manifest.component_end == len(record.canonical_names):
        if record.transpose:
            if not _is_torch_tensor(record.tensor):
                raise NotImplementedError("Transpose writeback requires torch tensors.")
            record.tensor.copy_(stage_buffer.transpose(-2, -1))
            return
        _copy_into(record.tensor, stage_buffer)
        return

    if record.packing.axis not in (None, 0):
        raise NotImplementedError(
            f"Staged subset writeback requires pack axis 0, got {record.packing.axis}."
        )
    start = _component_axis_offset(record, manifest)
    length = _component_axis_length(record, manifest)
    if _is_torch_tensor(record.tensor):
        target = record.tensor.narrow(0, start, length)
        target.copy_(stage_buffer)
    else:
        dst = _slice_bytes_for_axis0(
            record.tensor,
            record.actual_shape,
            start,
            length,
        )
        _copy_into(dst, stage_buffer)


def _materialize_binding(
    record: ParameterRecord,
    manifest: TensorBindingManifest,
    *,
    as_target: bool,
) -> TensorBinding:
    if manifest.binding_kind is BindingKind.DIRECT:
        if manifest.component_start == 0 and manifest.component_end == len(record.canonical_names):
            buffer = record.tensor
        elif record.packing.axis == 0:
            start = _component_axis_offset(record, manifest)
            length = _component_axis_length(record, manifest)
            buffer = _slice_bytes_for_axis0(
                record.tensor,
                record.actual_shape,
                start,
                length,
            )
        else:
            if as_target:
                stage_buffer = _allocate_target_stage(record, manifest)
                return TensorBinding(
                    manifest=replace(manifest, binding_kind=BindingKind.STAGED),
                    buffer=stage_buffer,
                    apply_fn=lambda: _apply_target_stage(record, manifest, stage_buffer),
                )
            buffer = _extract_source_stage(record, manifest)
            return TensorBinding(
                manifest=replace(manifest, binding_kind=BindingKind.STAGED),
                buffer=buffer,
            )
        return TensorBinding(manifest=manifest, buffer=buffer)

    if as_target:
        stage_buffer = _allocate_target_stage(record, manifest)
        return TensorBinding(
            manifest=manifest,
            buffer=stage_buffer,
            apply_fn=lambda: _apply_target_stage(record, manifest, stage_buffer),
        )

    stage_buffer = _extract_source_stage(record, manifest)
    return TensorBinding(manifest=manifest, buffer=stage_buffer)


def _binding_map(
    snapshot: FrameworkSnapshot,
    manifests: tuple[TensorBindingManifest, ...],
    target_binding_ids: AbstractSet[str],
) -> dict[str, TensorBinding]:
    records = snapshot.records_by_id()
    bindings = {}
    for manifest in manifests:
        record = records[manifest.record_id]
        bindings[manifest.binding_id] = _materialize_binding(
            record,
            manifest,
            as_target=manifest.binding_id in target_binding_ids,
        )
    return bindings


def execute_exchange_plan(
    local_snapshot: FrameworkSnapshot,
    execution_slice: ExecutionSlice,
    coordinator: ExchangeCoordinator,
    transport_session: TransportSession,
) -> ExchangeResult:
    """Execute the local portion of a precomputed exchange plan."""

    local_rank = local_snapshot.endpoint.rank
    if int(execution_slice.rank) != int(local_rank):
        raise ValueError(
            f"Execution slice rank {execution_slice.rank} does not match local snapshot rank {local_rank}."
        )

    target_binding_ids = set(execution_slice.target_binding_ids)
    bindings = _binding_map(
        local_snapshot,
        execution_slice.binding_manifests,
        target_binding_ids,
    )
    prepared = []
    applied = []
    for binding in bindings.values():
        if binding.prepare_fn is not None:
            binding.prepare_fn()
            prepared.append(binding.manifest.binding_id)
        transport_session.register_tensor(
            binding.manifest.exchange_key,
            binding.buffer,
            memory_kind=binding.manifest.memory_kind,
            dtype=binding.manifest.dtype,
            shape=binding.manifest.local_shape,
        )

    coordinator.publish_binding_manifests(local_rank, execution_slice.binding_manifests)
    descriptor = transport_session.publish_descriptors()
    coordinator.publish_rank_descriptor(local_rank, descriptor)
    transport_session.install_peer_descriptors(coordinator.peer_descriptors_for(local_rank))

    transport_result = None
    if execution_slice.send_tasks:
        transport_result = transport_session.execute(TransferPlan(execution_slice.send_tasks))
    coordinator.mark_transfers_complete(local_rank)

    if execution_slice.target_binding_ids:
        coordinator.wait_for_sources(local_rank, execution_slice.expected_source_ranks)
        for binding_id in execution_slice.target_binding_ids:
            binding = bindings[binding_id]
            if binding.apply_fn is not None:
                binding.apply_fn()
                applied.append(binding.manifest.binding_id)

    warnings = ()
    if transport_result is not None:
        warnings = transport_result.warnings
    return ExchangeResult(
        rank=local_rank,
        transport_result=transport_result,
        prepared_binding_ids=tuple(prepared),
        applied_binding_ids=tuple(applied),
        warnings=warnings,
    )


def frozen_coordinator_from_payload(
    peer_descriptors: Mapping[int, RankDescriptor],
    completed_ranks: tuple[int, ...],
) -> FrozenExchangeCoordinator:
    """Build a read-only coordinator from a serialized payload."""

    return FrozenExchangeCoordinator(
        peer_descriptors=dict(peer_descriptors),
        completed_ranks=tuple(completed_ranks),
    )
