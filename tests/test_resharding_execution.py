from __future__ import annotations

import threading
import time

from rlite.resharding import (
    FrameworkRole,
    FrameworkSnapshot,
    InMemoryExchangeCoordinator,
    ParameterRecord,
    WorkerEndpoint,
    build_exchange_plan,
    execute_exchange_plan,
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


def _session(rank: int, world_size: int, loopback: LoopbackCoordinator) -> TransportSession:
    backend = LoopbackTransportBackend(coordinator=loopback)
    return TransportSession(rank=rank, world_size=world_size, backend=backend)


def _wait_for_descriptor(coordinator: InMemoryExchangeCoordinator, local_rank: int, peer_rank: int) -> None:
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if peer_rank in coordinator.peer_descriptors_for(local_rank):
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for peer descriptor {peer_rank} on rank {local_rank}.")


def test_execute_exchange_plan_copies_only_the_requested_packed_component() -> None:
    source_snapshot = _snapshot(
        _endpoint(0, framework="megatron", role=FrameworkRole.SOURCE),
        _record(
            "src.qkv",
            tensor=bytearray(b"QQQQKKVV"),
            canonical_names=("layer.q", "layer.k", "layer.v"),
            logical_shape=(8,),
            packing=PACK_QKV,
            component_logical_sizes=(4, 2, 2),
            component_local_sizes=(4, 2, 2),
        ),
    )
    target_buffer = bytearray(b"xx")
    target_snapshot = _snapshot(
        _endpoint(1, framework="sglang", role=FrameworkRole.TARGET),
        _record(
            "dst.k",
            tensor=target_buffer,
            canonical_names=("layer.k",),
            logical_shape=(2,),
        ),
    )
    plan = build_exchange_plan(source_snapshot, target_snapshot)
    loopback = LoopbackCoordinator()
    coordinator = InMemoryExchangeCoordinator()
    results: dict[str, object] = {}

    def run_target() -> None:
        results["target"] = execute_exchange_plan(
            target_snapshot,
            plan.execution_slices[1],
            coordinator,
            _session(1, 2, loopback),
        )

    target_thread = threading.Thread(target=run_target)
    target_thread.start()
    _wait_for_descriptor(coordinator, 0, 1)
    results["source"] = execute_exchange_plan(
        source_snapshot,
        plan.execution_slices[0],
        coordinator,
        _session(0, 2, loopback),
    )
    target_thread.join(timeout=2.0)

    assert target_thread.is_alive() is False
    assert bytes(target_buffer) == b"KK"
    assert results["source"].transport_result.completed_tasks == 1


def test_execute_exchange_plan_applies_staged_target_writeback() -> None:
    source_snapshot = _snapshot(
        _endpoint(0, framework="megatron", role=FrameworkRole.SOURCE),
        _record(
            "src.weight",
            tensor=bytearray(b"abcdefgh"),
            canonical_names=("layer.weight",),
            logical_shape=(8,),
        ),
    )
    target_buffer = bytearray(b"XXXXXXXX")
    target_snapshot = _snapshot(
        _endpoint(1, framework="sglang", role=FrameworkRole.TARGET),
        _record(
            "dst.weight",
            tensor=target_buffer,
            canonical_names=("layer.weight",),
            logical_shape=(8,),
            reshape=(8,),
        ),
    )
    plan = build_exchange_plan(source_snapshot, target_snapshot)
    loopback = LoopbackCoordinator()
    coordinator = InMemoryExchangeCoordinator()
    results: dict[str, object] = {}

    def run_target() -> None:
        results["target"] = execute_exchange_plan(
            target_snapshot,
            plan.execution_slices[1],
            coordinator,
            _session(1, 2, loopback),
        )

    target_thread = threading.Thread(target=run_target)
    target_thread.start()
    _wait_for_descriptor(coordinator, 0, 1)
    results["source"] = execute_exchange_plan(
        source_snapshot,
        plan.execution_slices[0],
        coordinator,
        _session(0, 2, loopback),
    )
    target_thread.join(timeout=2.0)

    assert target_thread.is_alive() is False
    assert bytes(target_buffer) == b"abcdefgh"
    assert results["target"].applied_binding_ids
