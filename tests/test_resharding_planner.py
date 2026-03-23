from __future__ import annotations

from rlite.resharding import (
    FrameworkRole,
    FrameworkSnapshot,
    LocalityTier,
    ParameterRecord,
    TopologyPolicy,
    WorkerEndpoint,
    build_exchange_plan,
)
from rlite.transport import MemoryKind, TransferPath
from rlite.weight_mapping.types import PackingSpec, ParallelKind, ParallelSpec, TensorPackKind


PACK_NONE = PackingSpec(TensorPackKind.NONE, (), None, ())
PACK_QKV = PackingSpec(
    TensorPackKind.FUSED_QKV,
    ("q", "k", "v"),
    0,
    ("q", "k", "v"),
)
PAR_REPLICATED = ParallelSpec(ParallelKind.REPLICATED, None, None, None, False, False)
PAR_TP_COL = ParallelSpec(ParallelKind.TP_COL, 0, None, None, True, True)


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        product *= dim
    return product


def _endpoint(
    rank: int,
    *,
    framework: str,
    role: FrameworkRole,
    host: str = "host0",
    process_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    dp_rank: int = 0,
    dp_size: int = 1,
    nic_names: tuple[str, ...] = (),
    provider_names: tuple[str, ...] = (),
    metadata: dict[str, str] | None = None,
) -> WorkerEndpoint:
    return WorkerEndpoint(
        rank=rank,
        framework=framework,
        role=role,
        host=host,
        process_id=rank if process_id is None else process_id,
        tensor_parallel_rank=tp_rank,
        tensor_parallel_size=tp_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        nic_names=nic_names,
        provider_names=provider_names,
        metadata=metadata or {},
    )


def _record(
    record_id: str,
    *,
    canonical_names: tuple[str, ...],
    logical_shape: tuple[int, ...],
    local_shape: tuple[int, ...] | None = None,
    tensor: bytearray | None = None,
    packing: PackingSpec = PACK_NONE,
    parallel: ParallelSpec = PAR_REPLICATED,
    component_logical_sizes: tuple[int, ...] = (),
    component_local_sizes: tuple[int, ...] = (),
    memory_kind: MemoryKind = MemoryKind.CPU,
    dtype: str = "uint8",
) -> ParameterRecord:
    local_shape = local_shape or logical_shape
    num_bytes = _shape_product(local_shape)
    return ParameterRecord(
        record_id=record_id,
        framework_name=record_id,
        tensor=tensor if tensor is not None else bytearray(num_bytes),
        dtype=dtype,
        logical_shape=logical_shape,
        local_shape=local_shape,
        actual_shape=local_shape,
        canonical_names=canonical_names,
        packing=packing,
        parallel=parallel,
        tensor_role="weight",
        memory_kind=memory_kind,
        component_logical_sizes=component_logical_sizes,
        component_local_sizes=component_local_sizes,
    )


def _snapshot(endpoint: WorkerEndpoint, *records: ParameterRecord) -> FrameworkSnapshot:
    return FrameworkSnapshot(endpoint=endpoint, records=records)


def test_planner_splits_fused_qkv_into_canonical_components() -> None:
    source = _snapshot(
        _endpoint(0, framework="megatron", role=FrameworkRole.SOURCE),
        _record(
            "decoder.layers.0.self_attention.linear_qkv.weight",
            canonical_names=("layer.0.q", "layer.0.k", "layer.0.v"),
            logical_shape=(8,),
            local_shape=(8,),
            tensor=bytearray(b"QQQQKKVV"),
            packing=PACK_QKV,
            parallel=PAR_TP_COL,
            component_logical_sizes=(4, 2, 2),
            component_local_sizes=(4, 2, 2),
        ),
    )
    target = _snapshot(
        _endpoint(4, framework="sglang", role=FrameworkRole.TARGET),
        _record("model.layers.0.self_attn.q_proj.weight", canonical_names=("layer.0.q",), logical_shape=(4,)),
        _record("model.layers.0.self_attn.k_proj.weight", canonical_names=("layer.0.k",), logical_shape=(2,)),
        _record("model.layers.0.self_attn.v_proj.weight", canonical_names=("layer.0.v",), logical_shape=(2,)),
    )

    plan = build_exchange_plan(source, target)

    assert len(plan.bundles) == 3
    assert {bundle.split_reason for bundle in plan.bundles} == {"split_to_canonical"}
    send_tasks = list(plan.execution_slices[0].send_tasks)
    assert sorted(task.num_bytes for task in send_tasks) == [2, 2, 4]
    assert all(task.src_slice.offset == 0 for task in send_tasks)


def test_planner_tp_reshard_avoids_full_fanout() -> None:
    source_snapshots = tuple(
        _snapshot(
            _endpoint(
                rank,
                framework="megatron",
                role=FrameworkRole.SOURCE,
                tp_rank=rank,
                tp_size=2,
            ),
            _record(
                f"src.weight.rank{rank}",
                canonical_names=("layer.0.out",),
                logical_shape=(8,),
                local_shape=(4,),
                parallel=PAR_TP_COL,
            ),
        )
        for rank in range(2)
    )
    target_snapshots = tuple(
        _snapshot(
            _endpoint(
                10 + tp_rank,
                framework="sglang",
                role=FrameworkRole.TARGET,
                tp_rank=tp_rank,
                tp_size=4,
            ),
            _record(
                f"dst.weight.rank{tp_rank}",
                canonical_names=("layer.0.out",),
                logical_shape=(8,),
                local_shape=(2,),
                parallel=PAR_TP_COL,
            ),
        )
        for tp_rank in range(4)
    )

    plan = build_exchange_plan(source_snapshots, target_snapshots)

    assert sum(task.num_bytes for task in plan.execution_slices[0].send_tasks) == 4
    assert sum(task.num_bytes for task in plan.execution_slices[1].send_tasks) == 4
    assert plan.execution_slices[10].expected_source_ranks == (0,)
    assert plan.execution_slices[11].expected_source_ranks == (0,)
    assert plan.execution_slices[12].expected_source_ranks == (1,)
    assert plan.execution_slices[13].expected_source_ranks == (1,)


def test_planner_balances_replicated_sources_and_honors_pinned_nics() -> None:
    source_snapshots = (
        _snapshot(
            _endpoint(
                0,
                framework="megatron",
                role=FrameworkRole.SOURCE,
                host="node-a",
                nic_names=("mlx5_0",),
                dp_rank=0,
                dp_size=2,
            ),
            _record("src.rank0", canonical_names=("layer.0.norm",), logical_shape=(4,)),
        ),
        _snapshot(
            _endpoint(
                1,
                framework="megatron",
                role=FrameworkRole.SOURCE,
                host="node-a",
                nic_names=("mlx5_1",),
                dp_rank=1,
                dp_size=2,
            ),
            _record("src.rank1", canonical_names=("layer.0.norm",), logical_shape=(4,)),
        ),
    )
    target_snapshots = (
        _snapshot(
            _endpoint(10, framework="sglang", role=FrameworkRole.TARGET, host="node-b"),
            _record("dst.rank0", canonical_names=("layer.0.norm",), logical_shape=(4,)),
        ),
        _snapshot(
            _endpoint(11, framework="sglang", role=FrameworkRole.TARGET, host="node-c"),
            _record("dst.rank1", canonical_names=("layer.0.norm",), logical_shape=(4,)),
        ),
    )

    plan = build_exchange_plan(
        source_snapshots,
        target_snapshots,
        TopologyPolicy(
            pinned_nics={
                0: ("mlx5_0",),
                1: ("mlx5_1",),
                10: ("eth0",),
            }
        ),
    )

    chosen_sources = {decision.src_rank for decision in plan.topology_decisions.values()}
    assert chosen_sources == {0, 1}
    assert plan.execution_slices[0].selected_nic_name == "mlx5_0"
    assert plan.execution_slices[1].selected_nic_name == "mlx5_1"
    assert plan.execution_slices[10].selected_nic_name == "eth0"


def test_planner_prefers_same_host_path_over_cross_host_rdma() -> None:
    source_snapshots = (
        _snapshot(
            _endpoint(
                0,
                framework="megatron",
                role=FrameworkRole.SOURCE,
                host="node-a",
                process_id=100,
            ),
            _record("src.same_host", canonical_names=("layer.0.down",), logical_shape=(4,)),
        ),
        _snapshot(
            _endpoint(
                1,
                framework="megatron",
                role=FrameworkRole.SOURCE,
                host="node-b",
                process_id=101,
                provider_names=("verbs",),
            ),
            _record("src.rdma", canonical_names=("layer.0.down",), logical_shape=(4,)),
        ),
    )
    target = _snapshot(
        _endpoint(10, framework="sglang", role=FrameworkRole.TARGET, host="node-a", process_id=200),
        _record("dst.weight", canonical_names=("layer.0.down",), logical_shape=(4,)),
    )

    plan = build_exchange_plan(source_snapshots, target)

    decision = plan.topology_decisions[(0, 10)]
    assert decision.locality_tier is LocalityTier.SAME_HOST_DIRECT
    assert decision.preferred_path is TransferPath.MEMCPY
