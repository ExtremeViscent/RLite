from __future__ import annotations

import sys
from types import SimpleNamespace

from rlite.integrations import (
    RemoteTopology,
    build_sglang_update_payload,
    collect_megatron_snapshot,
    collect_sglang_snapshot,
    dispatch_sglang_update,
    encode_payload,
    synthesize_sglang_target_snapshots,
    sync_megatron_to_remote_sglang,
)
from rlite.integrations import sglang as sglang_integration
from rlite.resharding import ExchangeResult, ExecutionSlice, FrameworkRole, TensorBindingManifest
from rlite.transport import MemoryKind, RankDescriptor
from rlite.weight_mapping import Framework, get_profile, translate_tensor


class DummyParameter:
    def __init__(self, data: bytearray, *, shape: tuple[int, ...] | None = None):
        self.data = data
        self.shape = shape or (len(data),)
        self.dtype = "uint8"
        self.device = "cpu"


class DummyModel:
    def __init__(self):
        self._parameters = [
            ("model.layers.0.self_attn.qkv_proj.weight", DummyParameter(bytearray(12288))),
        ]

    def named_parameters(self, recurse: bool = True):
        del recurse
        return tuple(self._parameters)


class DummyServerArgs:
    dp_size = 1


class DummyModelRunner:
    def __init__(self):
        self.model = DummyModel()
        self.tp_size = 2
        self.pp_size = 1
        self.moe_ep_size = 1
        self.dp_rank = 0
        self.tp_rank = 1
        self.pp_rank = 0
        self.moe_ep_rank = 0
        self.gpu_id = None
        self.server_args = DummyServerArgs()


class DummyEngine:
    def __init__(self):
        self.calls = []

    def update_weights_from_tensor(self, named_tensors, **kwargs):
        self.calls.append((named_tensors, kwargs))
        return True, "Success"


class DummyMegatronGroups:
    def __init__(self, tp):
        self.tp = tuple(tp)
        self.dp = (0,)
        self.pp = (0,)
        self.ep = None
        self.expt_tp = None


class DummyMegatronModel:
    def __init__(self, rank: int, qkv_shape: tuple[int, ...]):
        self.global_rank = rank
        self.pg_collection = DummyMegatronGroups(tp=(0, 1, 2, 3))
        self._parameters = [
            (
                "decoder.layers.0.self_attention.linear_qkv.weight",
                DummyParameter(bytearray(1), shape=qkv_shape),
            ),
        ]

    def named_parameters(self, recurse: bool = True):
        del recurse
        return tuple(self._parameters)


def test_collect_sglang_snapshot_reads_parallel_ranks_and_mapping_metadata() -> None:
    profile = get_profile("qwen", "qwen2", overrides={"tensor_parallel_size": 2})
    runner = DummyModelRunner()

    snapshot = collect_sglang_snapshot(runner, profile, role=FrameworkRole.TARGET)

    assert snapshot.endpoint.tensor_parallel_rank == 1
    assert snapshot.endpoint.tensor_parallel_size == 2
    assert snapshot.endpoint.metadata["experts_per_rank"] == "0"
    assert snapshot.records[0].canonical_names == (
        "layers.0.attn.q",
        "layers.0.attn.k",
        "layers.0.attn.v",
    )
    assert snapshot.records[0].memory_kind is MemoryKind.CPU


def test_build_and_dispatch_sglang_payload_uses_rlite_contract() -> None:
    profile = get_profile("llama", "llama3")
    execution_slice = ExecutionSlice(
        rank=3,
        binding_manifests=(
            TensorBindingManifest(
                binding_id="3:weight:0:1",
                record_id="weight",
                rank=3,
                exchange_key="rlite_weight",
                canonical_names=("layer.weight",),
                framework_name="sglang",
                framework_tensor_name="weight",
                binding_kind="direct",
                memory_kind="cpu",
                dtype="uint8",
                logical_shape=(8,),
                local_shape=(8,),
                logical_slices=((0, 8),),
            ),
        ),
        send_tasks=(),
        target_binding_ids=("3:weight:0:1",),
        expected_source_ranks=(0,),
        selected_nic_name="mlx5_0",
        selected_provider_name="verbs",
    )

    payload = build_sglang_update_payload(
        profile,
        execution_slice,
        peer_descriptors={},
        completed_source_ranks=(0,),
        session_host="node-a",
    )
    engine = DummyEngine()

    result = dispatch_sglang_update(engine, payload, flush_cache=False)

    assert payload["profile"] is profile
    assert result == (True, "Success")
    assert engine.calls[0][0] == [("__rlite_payload__", payload)]
    assert engine.calls[0][1]["load_format"] == "rlite"
    assert engine.calls[0][1]["flush_cache"] is False


def test_collect_megatron_snapshot_preserves_local_parallel_rank_with_rank_offset() -> None:
    profile = get_profile("qwen", "qwen2_5", overrides={"tensor_parallel_size": 4})
    qkv_shape = translate_tensor(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        Framework.MEGATRON,
        Framework.MEGATRON,
        profile,
        view="local_shard",
    ).source_local_shape
    model = DummyMegatronModel(rank=2, qkv_shape=qkv_shape)

    snapshot = collect_megatron_snapshot(
        model,
        profile,
        role=FrameworkRole.SOURCE,
        rank_offset=4,
    )

    assert snapshot.endpoint.rank == 6
    assert snapshot.endpoint.tensor_parallel_rank == 2
    assert snapshot.endpoint.tensor_parallel_size == 4


def test_synthesize_sglang_target_snapshots_uses_rollout_tp_profile() -> None:
    train_profile = get_profile("qwen", "qwen2_5", overrides={"tensor_parallel_size": 4})
    rollout_profile = get_profile("qwen", "qwen2_5", overrides={"tensor_parallel_size": 2})
    train_qkv_shape = translate_tensor(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        Framework.MEGATRON,
        Framework.MEGATRON,
        train_profile,
        view="local_shard",
    ).source_local_shape
    expected_target_shape = translate_tensor(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        Framework.MEGATRON,
        Framework.SGLANG,
        rollout_profile,
        view="local_shard",
    ).targets[0].local_shape
    source_snapshots = tuple(
        collect_megatron_snapshot(
            DummyMegatronModel(rank=rank, qkv_shape=train_qkv_shape),
            train_profile,
            role=FrameworkRole.SOURCE,
        )
        for rank in range(4)
    )
    topology = RemoteTopology.from_grid(
        framework="sglang",
        tp_size=2,
        dp_size=2,
        rank_offset=4,
        hosts=("rollout-a", "rollout-a", "rollout-b", "rollout-b"),
    )

    synthesized = synthesize_sglang_target_snapshots(source_snapshots, rollout_profile, topology)

    assert tuple(snapshot.endpoint.rank for snapshot in synthesized) == (4, 5, 6, 7)
    assert synthesized[0].records[0].framework_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert synthesized[0].records[0].local_shape == expected_target_shape
    assert synthesized[2].records[0].local_shape == expected_target_shape


def test_sync_megatron_to_remote_sglang_runs_prepare_then_commit(monkeypatch) -> None:
    train_profile = get_profile("qwen", "qwen2_5", overrides={"tensor_parallel_size": 4})
    rollout_profile = get_profile("qwen", "qwen2_5", overrides={"tensor_parallel_size": 2})
    train_qkv_shape = translate_tensor(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        Framework.MEGATRON,
        Framework.MEGATRON,
        train_profile,
        view="local_shard",
    ).source_local_shape
    source_snapshots = tuple(
        collect_megatron_snapshot(
            DummyMegatronModel(rank=rank, qkv_shape=train_qkv_shape),
            train_profile,
            role=FrameworkRole.SOURCE,
        )
        for rank in range(4)
    )
    topology = RemoteTopology.from_grid(
        framework="sglang",
        tp_size=2,
        dp_size=2,
        rank_offset=4,
    )
    calls = []
    executed_ranks = []

    def fake_execute(snapshot, execution_slice, coordinator, session):
        del coordinator, session
        executed_ranks.append((snapshot.endpoint.rank, len(execution_slice.send_tasks)))
        return ExchangeResult(rank=snapshot.endpoint.rank, transport_result=None)

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, *, json, timeout):
        del timeout
        calls.append(url)
        if url.endswith("/rlite/prepare_receive"):
            payload = sglang_integration.decode_payload(json["payload_b64"])
            assert payload["profile"].tensor_parallel_size == 2
            assert set(payload["execution_slices_by_rank"]) == {4, 5, 6, 7}
            return FakeResponse(
                {
                    "success": True,
                    "payload_b64": encode_payload(
                        {
                            "rank_descriptors": {
                                rank: RankDescriptor(rank=rank, host="rollout")
                                for rank in topology.ranks
                            },
                            "requires_staging": False,
                            "fallback_bytes": 0,
                        }
                    ),
                }
            )
        if url.endswith("/rlite/commit_receive"):
            return FakeResponse(
                {
                    "success": True,
                    "payload_b64": encode_payload(
                        [{"rank": rank, "applied_binding_ids": ()} for rank in topology.ranks]
                    ),
                }
            )
        raise AssertionError(url)

    monkeypatch.setattr(sglang_integration, "execute_exchange_plan", fake_execute)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=fake_post))

    result = sync_megatron_to_remote_sglang(
        source_snapshots,
        train_profile=train_profile,
        rollout_profile=rollout_profile,
        topology=topology,
        remote_url="http://rollout.example",
    )

    assert calls == [
        "http://rollout.example/rlite/prepare_receive",
        "http://rollout.example/rlite/commit_receive",
    ]
    assert {rank for rank, task_count in executed_ranks if task_count > 0} == {0, 1, 2, 3}
    assert result["rollout_profile"].tensor_parallel_size == 2
