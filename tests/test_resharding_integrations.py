from __future__ import annotations

from rlite.integrations import build_sglang_update_payload, collect_sglang_snapshot, dispatch_sglang_update
from rlite.resharding import ExecutionSlice, FrameworkRole, TensorBindingManifest
from rlite.transport import MemoryKind
from rlite.weight_mapping import get_profile


class DummyParameter:
    def __init__(self, data: bytearray):
        self.data = data
        self.shape = (len(data),)
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
