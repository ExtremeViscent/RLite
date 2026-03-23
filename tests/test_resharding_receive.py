from __future__ import annotations

from rlite.resharding import (
    ExecutionSlice,
    FrameworkRole,
    FrameworkSnapshot,
    ParameterRecord,
    WorkerEndpoint,
    build_binding_manifest,
    commit_receive,
    prepare_receive,
)
from rlite.transport import LoopbackCoordinator, LoopbackTransportBackend, MemoryKind, TransportSession
from rlite.weight_mapping.types import PackingSpec, ParallelKind, ParallelSpec, TensorPackKind


PACK_NONE = PackingSpec(TensorPackKind.NONE, (), None, ())
PACK_QKV = PackingSpec(
    TensorPackKind.FUSED_QKV,
    ("q", "k", "v"),
    0,
    ("q", "k", "v"),
)
PAR_REPLICATED = ParallelSpec(ParallelKind.REPLICATED, None, None, None, False, False)


def _endpoint(rank: int, *, framework: str, role: FrameworkRole) -> WorkerEndpoint:
    return WorkerEndpoint(
        rank=rank,
        framework=framework,
        role=role,
        host="host0",
        process_id=1000 + rank,
    )


def _record(
    record_id: str,
    *,
    tensor: bytearray,
    canonical_names: tuple[str, ...],
    logical_shape: tuple[int, ...],
    local_shape: tuple[int, ...] | None = None,
    packing: PackingSpec = PACK_NONE,
    component_logical_sizes: tuple[int, ...] = (),
    component_local_sizes: tuple[int, ...] = (),
    reshape: tuple[int, ...] | None = None,
) -> ParameterRecord:
    local_shape = local_shape or logical_shape
    return ParameterRecord(
        record_id=record_id,
        framework_name=record_id,
        tensor=tensor,
        dtype="uint8",
        logical_shape=logical_shape,
        local_shape=local_shape,
        actual_shape=local_shape,
        canonical_names=canonical_names,
        packing=packing,
        parallel=PAR_REPLICATED,
        tensor_role="weight",
        memory_kind=MemoryKind.CPU,
        component_logical_sizes=component_logical_sizes,
        component_local_sizes=component_local_sizes,
        reshape=reshape,
    )


def _snapshot(endpoint: WorkerEndpoint, *records: ParameterRecord) -> FrameworkSnapshot:
    return FrameworkSnapshot(endpoint=endpoint, records=records)


def _session(rank: int, world_size: int) -> TransportSession:
    return TransportSession(
        rank=rank,
        world_size=world_size,
        backend=LoopbackTransportBackend(coordinator=LoopbackCoordinator()),
    )


def test_prepare_receive_uses_live_direct_view_for_axis0_subset() -> None:
    target_buffer = bytearray(b"QQQQxxVV")
    target_endpoint = _endpoint(1, framework="sglang", role=FrameworkRole.TARGET)
    target_snapshot = _snapshot(
        target_endpoint,
        _record(
            "dst.qkv",
            tensor=target_buffer,
            canonical_names=("layer.q", "layer.k", "layer.v"),
            logical_shape=(8,),
            packing=PACK_QKV,
            component_logical_sizes=(4, 2, 2),
            component_local_sizes=(4, 2, 2),
        ),
    )
    manifest = build_binding_manifest(target_snapshot.records[0], target_endpoint, ("layer.k",))
    execution_slice = ExecutionSlice(
        rank=1,
        binding_manifests=(manifest,),
        send_tasks=(),
        target_binding_ids=(manifest.binding_id,),
        expected_source_ranks=(),
    )

    pending = prepare_receive(target_snapshot, execution_slice, _session(1, 2))
    binding = pending.bindings[manifest.binding_id]

    assert pending.requires_staging is False
    assert pending.fallback_bytes == 0
    assert isinstance(binding.buffer, memoryview)

    binding.buffer[:] = b"KK"
    result = commit_receive(pending)

    assert bytes(target_buffer) == b"QQQQKKVV"
    assert result.applied_binding_ids == ()


def test_prepare_receive_uses_one_owner_stage_buffer_for_reshape_fallback() -> None:
    target_buffer = bytearray(b"QQQQxxVV")
    target_endpoint = _endpoint(1, framework="sglang", role=FrameworkRole.TARGET)
    target_snapshot = _snapshot(
        target_endpoint,
        _record(
            "dst.qkv",
            tensor=target_buffer,
            canonical_names=("layer.q", "layer.k", "layer.v"),
            logical_shape=(8,),
            packing=PACK_QKV,
            component_logical_sizes=(4, 2, 2),
            component_local_sizes=(4, 2, 2),
            reshape=(8,),
        ),
    )
    manifest = build_binding_manifest(target_snapshot.records[0], target_endpoint, ("layer.k",))
    execution_slice = ExecutionSlice(
        rank=1,
        binding_manifests=(manifest,),
        send_tasks=(),
        target_binding_ids=(manifest.binding_id,),
        expected_source_ranks=(),
    )

    pending = prepare_receive(target_snapshot, execution_slice, _session(1, 2))
    binding = pending.bindings[manifest.binding_id]

    assert pending.requires_staging is True
    assert pending.fallback_bytes == 8
    assert binding.manifest.metadata["stage_mode"] == "ping_pong"

    binding.buffer[:] = b"KK"
    assert bytes(target_buffer) == b"QQQQxxVV"

    result = commit_receive(pending)

    assert bytes(target_buffer) == b"QQQQKKVV"
    assert result.applied_binding_ids == (manifest.binding_id,)
