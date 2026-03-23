from __future__ import annotations

import threading
import time

import pytest

from rlite.integrations import (
    RemoteTopology,
    collect_transformers_fsdp_snapshot,
    synthesize_transformers_fsdp_target_snapshots,
)
from rlite.resharding import (
    FrameworkRole,
    FrameworkSnapshot,
    InMemoryExchangeCoordinator,
    LinearSegment,
    ParameterRecord,
    WorkerEndpoint,
    build_exchange_plan,
    execute_exchange_plan,
)
from rlite.transport import LoopbackCoordinator, LoopbackTransportBackend, MemoryKind, TransportSession
from rlite.weight_mapping import make_config_like
from rlite.weight_mapping.profiles import get_profile
from rlite.weight_mapping.types import PackingSpec, ParallelKind, ParallelSpec, TensorPackKind


PACK_NONE = PackingSpec(TensorPackKind.NONE, (), None, ())
PAR_REPLICATED = ParallelSpec(ParallelKind.REPLICATED, None, None, None, False, False)
PAR_TP_ROW = ParallelSpec(ParallelKind.TP_ROW, 1, None, None, True, True)


def _profile():
    return get_profile(
        None,
        None,
        config=make_config_like(
            model_type="llama",
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=8,
            vocab_size=16,
        ),
    )


def _endpoint(
    rank: int,
    *,
    framework: str,
    role: FrameworkRole,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> WorkerEndpoint:
    return WorkerEndpoint(
        rank=rank,
        framework=framework,
        role=role,
        host="host0",
        process_id=1000 + rank,
        tensor_parallel_rank=tp_rank,
        tensor_parallel_size=tp_size,
    )


def _manual_record(
    record_id: str,
    *,
    tensor,
    canonical_names: tuple[str, ...],
    logical_shape: tuple[int, ...],
    local_shape: tuple[int, ...] | None = None,
    parallel: ParallelSpec = PAR_REPLICATED,
    linear_segments: tuple[LinearSegment, ...] = (),
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
        packing=PACK_NONE,
        parallel=parallel,
        tensor_role="weight",
        memory_kind=MemoryKind.CPU,
        linear_segments=linear_segments,
    )


def _snapshot(endpoint: WorkerEndpoint, *records: ParameterRecord) -> FrameworkSnapshot:
    return FrameworkSnapshot(endpoint=endpoint, records=records)


def _session(rank: int, world_size: int, loopback: LoopbackCoordinator) -> TransportSession:
    return TransportSession(
        rank=rank,
        world_size=world_size,
        backend=LoopbackTransportBackend(coordinator=loopback),
    )


def _wait_for_descriptor(coordinator: InMemoryExchangeCoordinator, local_rank: int, peer_rank: int) -> None:
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if peer_rank in coordinator.peer_descriptors_for(local_rank):
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for peer descriptor {peer_rank} on rank {local_rank}.")


class DummyParameter:
    def __init__(self, name: str, data, *, shape: tuple[int, ...], layout=None):
        self.name = name
        self.data = data
        self.shape = shape
        self.dtype = "uint8"
        self.device = "cpu"
        self.rlite_fsdp_layout = layout


class DummyModel:
    def __init__(self, parameters, *, infos=None):
        self._parameters = tuple(parameters)
        self._infos = tuple(infos or ())

    def named_parameters(self, recurse: bool = True):
        del recurse
        return tuple((parameter.name, parameter) for parameter in self._parameters)

    def rlite_fsdp_param_infos(self):
        return tuple(self._infos)


class FakeLocalTensor:
    def __init__(self, shape: tuple[int, ...], *, dtype: str = "uint8"):
        self.shape = shape
        self.dtype = dtype
        self.device = "cpu"

    def element_size(self) -> int:
        return 1


class FakeShard:
    def __init__(self, dim: int):
        self.dim = dim

    def __str__(self) -> str:
        return f"Shard({self.dim})"


class FakeMesh:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class FakeDTensor:
    def __init__(self, local_tensor: FakeLocalTensor, *, placements, mesh_shape: tuple[int, ...]):
        self._local_tensor = local_tensor
        self.placements = tuple(placements)
        self.device_mesh = FakeMesh(mesh_shape)

    def to_local(self):
        return self._local_tensor


def test_planner_linear_segments_handle_column_shard_overlap() -> None:
    source = (
        _snapshot(
            _endpoint(0, framework="transformers", role=FrameworkRole.SOURCE, tp_rank=0, tp_size=2),
            _manual_record(
                "src.col.0",
                tensor=bytearray(b"abcd"),
                canonical_names=("layer.weight",),
                logical_shape=(2, 4),
                local_shape=(2, 2),
                parallel=PAR_TP_ROW,
            ),
        ),
        _snapshot(
            _endpoint(2, framework="transformers", role=FrameworkRole.SOURCE, tp_rank=1, tp_size=2),
            _manual_record(
                "src.col.1",
                tensor=bytearray(b"wxyz"),
                canonical_names=("layer.weight",),
                logical_shape=(2, 4),
                local_shape=(2, 2),
                parallel=PAR_TP_ROW,
            ),
        ),
    )
    target = _snapshot(
        _endpoint(1, framework="sglang", role=FrameworkRole.TARGET),
        _manual_record(
            "dst.full",
            tensor=bytearray(b"XXXXXXXX"),
            canonical_names=("layer.weight",),
            logical_shape=(2, 4),
        ),
    )

    plan = build_exchange_plan(source, target)
    assert [(task.src_slice.offset, task.dst_slice.offset, task.num_bytes) for task in plan.execution_slices[0].send_tasks] == [
        (0, 0, 2),
        (2, 4, 2),
    ]
    assert [(task.src_slice.offset, task.dst_slice.offset, task.num_bytes) for task in plan.execution_slices[2].send_tasks] == [
        (0, 2, 2),
        (2, 6, 2),
    ]


def test_collect_transformers_fsdp_snapshot_uses_explicit_fsdp1_linear_segments() -> None:
    profile = _profile()
    flat_buffer = bytearray(b"abcdefghijklmnop")
    local_view = memoryview(flat_buffer)[4:12]
    parameter = DummyParameter(
        "model.layers.0.self_attn.q_proj.weight",
        local_view,
        shape=(4, 4),
    )
    model = DummyModel(
        [parameter],
        infos=[
            {
                "name": parameter.name,
                "parameter": parameter,
                "tensor": local_view,
                "fsdp_variant": "fsdp1",
                "use_orig_params": True,
                "logical_shape": (4, 4),
                "local_shape": (8,),
                "actual_shape": (8,),
                "linear_segments": ((4, 12, 0, 8),),
            }
        ],
    )

    snapshot = collect_transformers_fsdp_snapshot(model, profile, role=FrameworkRole.TARGET)

    assert snapshot.records[0].canonical_names == ("layers.0.attn.q",)
    assert snapshot.records[0].linear_segments == (
        LinearSegment(logical_start=4, logical_stop=12, byte_offset=0, byte_length=8),
    )


def test_collect_transformers_fsdp_snapshot_handles_full_fsdp1_shard() -> None:
    profile = _profile()
    parameter = DummyParameter("model.layers.0.self_attn.q_proj.weight", bytearray(b"abcdefghijklmnop"), shape=(4, 4))
    model = DummyModel(
        [parameter],
        infos=[
            {
                "name": parameter.name,
                "parameter": parameter,
                "tensor": bytearray(b"abcdefghijklmnop"),
                "fsdp_variant": "fsdp1",
                "use_orig_params": True,
                "logical_shape": (4, 4),
                "local_shape": (16,),
                "actual_shape": (16,),
                "linear_segments": ((0, 16, 0, 16),),
            }
        ],
    )

    snapshot = collect_transformers_fsdp_snapshot(model, profile)

    assert snapshot.records[0].linear_segments == (
        LinearSegment(logical_start=0, logical_stop=16, byte_offset=0, byte_length=16),
    )


def test_collect_transformers_fsdp_snapshot_skips_absent_fsdp1_shard() -> None:
    profile = _profile()
    parameter = DummyParameter("model.layers.0.self_attn.q_proj.weight", bytearray(), shape=(4, 4))
    model = DummyModel(
        [parameter],
        infos=[
            {
                "name": parameter.name,
                "parameter": parameter,
                "tensor": bytearray(),
                "fsdp_variant": "fsdp1",
                "use_orig_params": True,
                "logical_shape": (4, 4),
                "local_shape": (0,),
                "actual_shape": (0,),
                "linear_segments": ((0, 0, 0, 0),),
            }
        ],
    )

    snapshot = collect_transformers_fsdp_snapshot(model, profile)

    assert snapshot.records == ()


def test_collect_transformers_fsdp_snapshot_infers_fsdp2_shard0_layout() -> None:
    profile = _profile()
    local_tensor = FakeLocalTensor((2, 4))
    dtensor = FakeDTensor(local_tensor, placements=(FakeShard(0),), mesh_shape=(2,))
    parameter = DummyParameter("model.layers.0.self_attn.q_proj.weight", dtensor, shape=(4, 4))
    model = DummyModel([parameter])
    model.data_parallel_rank = 1
    model.data_parallel_size = 2

    snapshot = collect_transformers_fsdp_snapshot(model, profile, role=FrameworkRole.SOURCE)

    assert snapshot.records[0].linear_segments == (
        LinearSegment(logical_start=8, logical_stop=16, byte_offset=0, byte_length=8),
    )


def test_collect_transformers_fsdp_snapshot_rejects_fsdp1_without_use_orig_params() -> None:
    profile = _profile()
    parameter = DummyParameter("model.layers.0.self_attn.q_proj.weight", bytearray(b"abcdefgh"), shape=(4, 4))
    model = DummyModel(
        [parameter],
        infos=[
            {
                "name": parameter.name,
                "parameter": parameter,
                "tensor": bytearray(b"abcdefgh"),
                "fsdp_variant": "fsdp1",
                "use_orig_params": False,
                "logical_shape": (4, 4),
                "local_shape": (8,),
                "actual_shape": (8,),
                "linear_segments": ((0, 8, 0, 8),),
            }
        ],
    )

    with pytest.raises(ValueError, match="use_orig_params=True"):
        collect_transformers_fsdp_snapshot(model, profile)


def test_collect_transformers_fsdp_snapshot_rejects_fsdp2_custom_placement() -> None:
    profile = _profile()
    local_tensor = FakeLocalTensor((2, 4))
    dtensor = FakeDTensor(local_tensor, placements=(FakeShard(1),), mesh_shape=(2,))
    parameter = DummyParameter("model.layers.0.self_attn.q_proj.weight", dtensor, shape=(4, 4))
    model = DummyModel([parameter])

    with pytest.raises(ValueError, match="Shard\\(0\\)"):
        collect_transformers_fsdp_snapshot(model, profile)


def test_transformers_fsdp_target_direct_receive_writes_into_flat_shard_view() -> None:
    profile = _profile()
    source_snapshot = _snapshot(
        _endpoint(0, framework="megatron", role=FrameworkRole.SOURCE),
        _manual_record(
            "decoder.layers.0.self_attention.linear_q.weight",
            tensor=bytearray(b"abcdefghijklmnop"),
            canonical_names=("layers.0.attn.q",),
            logical_shape=(4, 4),
        ),
    )
    flat_buffer = bytearray(b"________________")
    local_view = memoryview(flat_buffer)[4:12]
    parameter = DummyParameter(
        "model.layers.0.self_attn.q_proj.weight",
        local_view,
        shape=(4, 4),
    )
    target_model = DummyModel(
        [parameter],
        infos=[
            {
                "name": parameter.name,
                "parameter": parameter,
                "tensor": local_view,
                "fsdp_variant": "fsdp1",
                "use_orig_params": True,
                "logical_shape": (4, 4),
                "local_shape": (8,),
                "actual_shape": (8,),
                "linear_segments": ((4, 12, 0, 8),),
            }
        ],
    )
    target_snapshot = collect_transformers_fsdp_snapshot(
        target_model,
        profile,
        role=FrameworkRole.TARGET,
        rank_offset=1,
    )

    plan = build_exchange_plan(source_snapshot, target_snapshot)
    loopback = LoopbackCoordinator()
    coordinator = InMemoryExchangeCoordinator()

    def run_target() -> None:
        execute_exchange_plan(
            target_snapshot,
            plan.execution_slices[1],
            coordinator,
            _session(1, 2, loopback),
        )

    target_thread = threading.Thread(target=run_target)
    target_thread.start()
    _wait_for_descriptor(coordinator, 0, 1)
    execute_exchange_plan(
        source_snapshot,
        plan.execution_slices[0],
        coordinator,
        _session(0, 2, loopback),
    )
    target_thread.join(timeout=2.0)

    assert target_thread.is_alive() is False
    assert bytes(flat_buffer) == b"____efghijkl____"


def test_collect_transformers_fsdp_snapshot_never_calls_summon_full_params() -> None:
    profile = _profile()
    parameter = DummyParameter("model.layers.0.self_attn.q_proj.weight", bytearray(b"abcd"), shape=(4, 4))

    class NoGatherModel(DummyModel):
        def summon_full_params(self, *args, **kwargs):
            raise AssertionError("summon_full_params should not be called")

    model = NoGatherModel(
        [parameter],
        infos=[
            {
                "name": parameter.name,
                "parameter": parameter,
                "tensor": bytearray(b"abcd"),
                "fsdp_variant": "fsdp1",
                "use_orig_params": True,
                "logical_shape": (4, 4),
                "local_shape": (4,),
                "actual_shape": (4,),
                "linear_segments": ((0, 4, 0, 4),),
            }
        ],
    )

    snapshot = collect_transformers_fsdp_snapshot(model, profile)

    assert len(snapshot.records) == 1


def test_synthesize_transformers_fsdp_target_snapshots_uses_transformers_names() -> None:
    profile = _profile()
    source_snapshot = _snapshot(
        _endpoint(0, framework="transformers", role=FrameworkRole.SOURCE),
        _manual_record(
            "model.layers.0.self_attn.q_proj.weight",
            tensor=bytearray(b"abcdefghijklmnop"),
            canonical_names=("layers.0.attn.q",),
            logical_shape=(4, 4),
        ),
    )
    topology = RemoteTopology.from_grid(
        framework="transformers",
        tp_size=1,
        dp_size=2,
        rank_offset=4,
    )

    synthesized = synthesize_transformers_fsdp_target_snapshots(source_snapshot, profile, topology)

    assert tuple(snapshot.endpoint.rank for snapshot in synthesized) == (4, 5)
    assert synthesized[0].records[0].framework_name == "model.layers.0.self_attn.q_proj.weight"
