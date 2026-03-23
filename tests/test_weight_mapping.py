from __future__ import annotations

import pytest

from rlite.weight_mapping import (
    Framework,
    ModelFamily,
    TensorPackKind,
    get_profile,
    make_config_like,
    resolve_rule,
    translate_key,
    translate_state_dict_keys,
    translate_tensor,
)


def test_profile_inference_from_config_like() -> None:
    config = make_config_like(
        model_type="llama",
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
    )
    profile = get_profile(None, None, config=config)

    assert profile.family == ModelFamily.LLAMA
    assert profile.variant == "llama3"
    assert profile.hidden_size == 8192
    assert profile.num_key_value_heads == 8


def test_llama_q_proj_maps_to_megatron_qkv() -> None:
    profile = get_profile("llama", "llama3")

    assert translate_key(
        "model.layers.0.self_attn.q_proj.weight",
        Framework.TRANSFORMERS,
        Framework.MEGATRON,
        profile,
    ) == ["decoder.layers.0.self_attention.linear_qkv.weight"]


def test_megatron_qkv_maps_to_transformers_split_attention() -> None:
    profile = get_profile("llama", "llama3")

    assert translate_key(
        "decoder.layers.2.self_attention.linear_qkv.weight",
        Framework.MEGATRON,
        Framework.TRANSFORMERS,
        profile,
    ) == [
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.2.self_attn.k_proj.weight",
        "model.layers.2.self_attn.v_proj.weight",
    ]


def test_sglang_qkv_maps_to_transformers_split_attention() -> None:
    profile = get_profile("qwen", "qwen2")

    assert translate_key(
        "model.layers.4.self_attn.qkv_proj.weight",
        Framework.SGLANG,
        Framework.TRANSFORMERS,
        profile,
    ) == [
        "model.layers.4.self_attn.q_proj.weight",
        "model.layers.4.self_attn.k_proj.weight",
        "model.layers.4.self_attn.v_proj.weight",
    ]


def test_glm4_fused_gate_up_maps_to_megatron_fc1() -> None:
    profile = get_profile("glm", "glm4")

    assert translate_key(
        "model.layers.1.mlp.gate_up_proj.weight",
        Framework.TRANSFORMERS,
        Framework.MEGATRON,
        profile,
    ) == ["decoder.layers.1.mlp.linear_fc1.weight"]


def test_chatglm3_legacy_query_key_value_maps_to_megatron_qkv() -> None:
    profile = get_profile("glm", "chatglm3")

    assert translate_key(
        "transformer.encoder.layers.3.self_attention.query_key_value.weight",
        Framework.TRANSFORMERS,
        Framework.MEGATRON,
        profile,
    ) == ["decoder.layers.3.self_attention.linear_qkv.weight"]


def test_gpt_dense_conv1d_rule_marks_transpose() -> None:
    profile = get_profile("gpt", "gpt_dense")
    rule = resolve_rule("transformer.h.5.attn.c_attn.weight", Framework.TRANSFORMERS, profile)

    assert rule.transpose is True
    assert rule.packing.kind == TensorPackKind.FUSED_QKV


def test_qwen2_moe_grouped_experts_expand_to_megatron_local_experts() -> None:
    profile = get_profile("qwen", "qwen2_moe")
    mapped = translate_tensor(
        "model.layers.1.mlp.experts.gate_up_proj",
        Framework.TRANSFORMERS,
        Framework.MEGATRON,
        profile,
    )

    assert mapped.target_names[0] == "decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight"
    assert mapped.target_names[-1] == "decoder.layers.1.mlp.experts.local_experts.59.linear_fc1.weight"
    assert len(mapped.target_names) == 60
    assert len(mapped.canonical_names) == 120


def test_qwen2_moe_shared_gate_maps_to_megatron_gate_weight() -> None:
    profile = get_profile("qwen", "qwen2_moe")

    assert translate_key(
        "model.layers.7.mlp.shared_expert_gate.weight",
        Framework.TRANSFORMERS,
        Framework.MEGATRON,
        profile,
    ) == ["decoder.layers.7.mlp.shared_experts.gate_weight"]


def test_deepseek_v3_fused_qkv_a_maps_to_megatron_qkv_down() -> None:
    profile = get_profile("deepseek", "deepseek_v3")

    assert translate_key(
        "model.layers.2.self_attn.fused_qkv_a_proj_with_mqa.weight",
        Framework.SGLANG,
        Framework.MEGATRON,
        profile,
    ) == ["decoder.layers.2.self_attention.linear_qkv_down_proj.weight"]


def test_deepseek_v3_q_a_maps_to_fused_sglang_projection() -> None:
    profile = get_profile("deepseek", "deepseek_v3")

    assert translate_key(
        "model.layers.2.self_attn.q_a_proj.weight",
        Framework.TRANSFORMERS,
        Framework.SGLANG,
        profile,
    ) == ["model.layers.2.self_attn.fused_qkv_a_proj_with_mqa.weight"]


def test_local_shard_metadata_for_megatron_qkv() -> None:
    profile = get_profile("llama", "llama3", overrides={"tensor_parallel_size": 4})
    mapped = translate_tensor(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        Framework.MEGATRON,
        Framework.TRANSFORMERS,
        profile,
        view="local_shard",
    )

    assert mapped.source_logical_shape == (6144, 4096)
    assert mapped.source_local_shape == (1536, 4096)
    assert mapped.parallel.kind.value == "tp_col"


def test_pipeline_hints_for_embedding_and_lm_head() -> None:
    profile = get_profile("llama", "llama2")

    embed_rule = resolve_rule("model.embed_tokens.weight", Framework.TRANSFORMERS, profile)
    lm_head_rule = resolve_rule("lm_head.weight", Framework.TRANSFORMERS, profile)

    assert embed_rule.parallel.pipeline_owner_hint == "first_stage"
    assert lm_head_rule.parallel.pipeline_owner_hint == "last_stage"


def test_round_trip_within_same_framework_keeps_key() -> None:
    profile = get_profile("glm", "glm4")

    assert translate_key(
        "model.layers.0.self_attn.q_proj.weight",
        Framework.TRANSFORMERS,
        Framework.TRANSFORMERS,
        profile,
    ) == ["model.layers.0.self_attn.q_proj.weight"]


def test_translate_state_dict_keys_returns_mapping() -> None:
    profile = get_profile("llama", "llama3")
    translated = translate_state_dict_keys(
        {
            "model.layers.0.self_attn.q_proj.weight": object(),
            "model.layers.0.mlp.gate_proj.weight": object(),
        },
        Framework.TRANSFORMERS,
        Framework.MEGATRON,
        profile,
    )

    assert translated["model.layers.0.self_attn.q_proj.weight"] == (
        "decoder.layers.0.self_attention.linear_qkv.weight",
    )
    assert translated["model.layers.0.mlp.gate_proj.weight"] == (
        "decoder.layers.0.mlp.linear_fc1.weight",
    )


def test_invalid_key_raises_key_error() -> None:
    profile = get_profile("llama", "llama2")

    with pytest.raises(KeyError):
        translate_key("model.layers.0.self_attn.not_real.weight", "transformers", "megatron", profile)


def test_invalid_view_raises_value_error() -> None:
    profile = get_profile("llama", "llama2")

    with pytest.raises(ValueError):
        translate_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            "transformers",
            "megatron",
            profile,
            view="wrong",
        )


@pytest.mark.parametrize(
    ("family", "variant", "source_framework", "target_framework", "key"),
    [
        ("qwen", "qwen2_5", Framework.MEGATRON, Framework.SGLANG, "decoder.layers.0.self_attention.linear_qkv.weight"),
        ("glm", "glm4", Framework.TRANSFORMERS, Framework.MEGATRON, "model.layers.0.self_attn.q_proj.weight"),
        ("llama", "llama3", Framework.TRANSFORMERS, Framework.MEGATRON, "model.layers.0.self_attn.q_proj.weight"),
        ("gpt", "gpt_dense", Framework.TRANSFORMERS, Framework.MEGATRON, "transformer.h.0.attn.c_attn.weight"),
        ("deepseek", "deepseek_v3", Framework.SGLANG, Framework.MEGATRON, "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight"),
    ],
)
def test_sample_keys_translate_across_supported_families(
    family: str,
    variant: str,
    source_framework: Framework,
    target_framework: Framework,
    key: str,
) -> None:
    profile = get_profile(family, variant)

    translated = translate_key(key, source_framework, target_framework, profile)

    assert translated
