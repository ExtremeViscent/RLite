"""Architecture profiles and lightweight config introspection."""

from __future__ import annotations

from dataclasses import fields
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

from .types import ArchitectureProfile, ModelFamily


def _profile(
    *,
    family: ModelFamily,
    variant: str,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    ffn_hidden_size: int,
    attention_layout: str,
    mlp_layout: str,
    qkv_order: tuple[str, ...] = ("q", "k", "v"),
    num_experts: int | None = None,
    expert_hidden_size: int | None = None,
    vocab_size: int | None = None,
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    attention_head_dim: int | None = None,
    value_head_dim: int | None = None,
    q_lora_rank: int | None = None,
    kv_lora_rank: int | None = None,
    qk_rope_head_dim: int | None = None,
    qk_nope_head_dim: int | None = None,
    supports_bias: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> ArchitectureProfile:
    return ArchitectureProfile(
        family=family,
        variant=variant,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        ffn_hidden_size=ffn_hidden_size,
        attention_layout=attention_layout,
        mlp_layout=mlp_layout,
        qkv_order=qkv_order,
        num_experts=num_experts,
        expert_hidden_size=expert_hidden_size,
        vocab_size=vocab_size,
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        attention_head_dim=attention_head_dim,
        value_head_dim=value_head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        supports_bias=supports_bias,
        metadata=MappingProxyType(dict(metadata or {})),
    )


_PROFILE_REGISTRY: dict[tuple[ModelFamily, str], ArchitectureProfile] = {
    (ModelFamily.LLAMA, "llama2"): _profile(
        family=ModelFamily.LLAMA,
        variant="llama2",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        ffn_hidden_size=11008,
        vocab_size=32000,
        attention_layout="mha",
        mlp_layout="split_gate_up",
    ),
    (ModelFamily.LLAMA, "llama3"): _profile(
        family=ModelFamily.LLAMA,
        variant="llama3",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        ffn_hidden_size=14336,
        vocab_size=128256,
        attention_layout="gqa",
        mlp_layout="split_gate_up",
    ),
    (ModelFamily.QWEN, "qwen2"): _profile(
        family=ModelFamily.QWEN,
        variant="qwen2",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        ffn_hidden_size=22016,
        vocab_size=151936,
        attention_layout="gqa",
        mlp_layout="split_gate_up",
        supports_bias=True,
    ),
    (ModelFamily.QWEN, "qwen2_5"): _profile(
        family=ModelFamily.QWEN,
        variant="qwen2_5",
        hidden_size=4096,
        num_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        ffn_hidden_size=22016,
        vocab_size=151936,
        attention_layout="gqa",
        mlp_layout="split_gate_up",
        supports_bias=True,
    ),
    (ModelFamily.QWEN, "qwen2_moe"): _profile(
        family=ModelFamily.QWEN,
        variant="qwen2_moe",
        hidden_size=2048,
        num_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        ffn_hidden_size=5632,
        expert_hidden_size=1408,
        num_experts=60,
        vocab_size=151936,
        attention_layout="mha",
        mlp_layout="split_gate_up_moe",
        supports_bias=True,
        metadata={"num_shared_experts": 4},
    ),
    (ModelFamily.GLM, "glm4"): _profile(
        family=ModelFamily.GLM,
        variant="glm4",
        hidden_size=4096,
        num_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        ffn_hidden_size=13696,
        vocab_size=151552,
        attention_layout="gqa",
        mlp_layout="fused_gate_up",
        supports_bias=True,
        metadata={"legacy_checkpoint_prefix": "model"},
    ),
    (ModelFamily.GLM, "chatglm3"): _profile(
        family=ModelFamily.GLM,
        variant="chatglm3",
        hidden_size=4096,
        num_layers=28,
        num_attention_heads=32,
        num_key_value_heads=2,
        ffn_hidden_size=13696,
        vocab_size=65024,
        attention_layout="mqa",
        mlp_layout="fused_gate_up",
        supports_bias=True,
        metadata={"legacy_checkpoint_prefix": "transformer"},
    ),
    (ModelFamily.GPT, "gpt_dense"): _profile(
        family=ModelFamily.GPT,
        variant="gpt_dense",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        ffn_hidden_size=16384,
        vocab_size=50257,
        attention_layout="mha",
        mlp_layout="classic_fc",
        supports_bias=True,
    ),
    (ModelFamily.DEEPSEEK, "deepseek_v2"): _profile(
        family=ModelFamily.DEEPSEEK,
        variant="deepseek_v2",
        hidden_size=5120,
        num_layers=60,
        num_attention_heads=128,
        num_key_value_heads=128,
        ffn_hidden_size=12288,
        expert_hidden_size=1536,
        num_experts=160,
        vocab_size=102400,
        attention_layout="mla",
        mlp_layout="deepseek_moe",
        attention_head_dim=192,
        value_head_dim=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=None,
        supports_bias=True,
        metadata={"num_shared_experts": 2},
    ),
    (ModelFamily.DEEPSEEK, "deepseek_v3"): _profile(
        family=ModelFamily.DEEPSEEK,
        variant="deepseek_v3",
        hidden_size=7168,
        num_layers=61,
        num_attention_heads=128,
        num_key_value_heads=128,
        ffn_hidden_size=18432,
        expert_hidden_size=2048,
        num_experts=256,
        vocab_size=129280,
        attention_layout="mla",
        mlp_layout="deepseek_moe",
        attention_head_dim=192,
        value_head_dim=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=1536,
        supports_bias=True,
        metadata={"num_shared_experts": 2},
    ),
}


_VARIANT_ALIASES = {
    "llama-2": "llama2",
    "llama_2": "llama2",
    "llama-3": "llama3",
    "llama_3": "llama3",
    "qwen2.5": "qwen2_5",
    "qwen2-5": "qwen2_5",
    "qwen2_5": "qwen2_5",
    "qwen2moe": "qwen2_moe",
    "glm-4": "glm4",
    "glm_4": "glm4",
    "chatglm-3": "chatglm3",
    "chatglm_3": "chatglm3",
    "gpt2": "gpt_dense",
    "deepseek-v2": "deepseek_v2",
    "deepseek-v3": "deepseek_v3",
}


def _coerce_family(family: ModelFamily | str | None) -> ModelFamily | None:
    if family is None:
        return None
    if isinstance(family, ModelFamily):
        return family
    return ModelFamily(str(family).lower())


def _normalize_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    value = str(variant).lower().replace(".", "_")
    return _VARIANT_ALIASES.get(value, value)


def _get_attr(config: object, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _infer_from_config(config: object) -> tuple[ModelFamily, str, dict[str, Any]]:
    model_type = str(_get_attr(config, "model_type", default="")).lower()
    class_name = config.__class__.__name__.lower()

    if model_type == "llama":
        variant = "llama3" if _get_attr(config, "num_key_value_heads", default=0) < _get_attr(
            config, "num_attention_heads", default=0
        ) else "llama2"
        family = ModelFamily.LLAMA
    elif model_type == "qwen2":
        family = ModelFamily.QWEN
        variant = "qwen2_5" if _get_attr(config, "max_position_embeddings", default=32768) > 32768 else "qwen2"
    elif model_type == "qwen2_moe":
        family = ModelFamily.QWEN
        variant = "qwen2_moe"
    elif model_type == "glm":
        family = ModelFamily.GLM
        variant = "glm4"
    elif "chatglm" in model_type or "chatglm" in class_name:
        family = ModelFamily.GLM
        variant = "chatglm3"
    elif model_type == "deepseek_v2":
        family = ModelFamily.DEEPSEEK
        variant = "deepseek_v3" if _get_attr(config, "q_lora_rank", default=None) else "deepseek_v2"
    elif model_type in {"gpt2", "gpt_bigcode", "gpt_oss"} or "gpt2" in class_name:
        family = ModelFamily.GPT
        variant = "gpt_dense"
    else:
        raise ValueError(f"Unable to infer model family from config type {model_type or class_name!r}.")

    overrides = {
        "hidden_size": _get_attr(config, "hidden_size"),
        "num_layers": _get_attr(config, "num_hidden_layers", "num_layers"),
        "num_attention_heads": _get_attr(config, "num_attention_heads"),
        "num_key_value_heads": _get_attr(
            config,
            "num_key_value_heads",
            default=_get_attr(
                config,
                "multi_query_group_num",
                default=_get_attr(config, "num_attention_heads"),
            ),
        ),
        "ffn_hidden_size": _get_attr(
            config,
            "intermediate_size",
            "ffn_hidden_size",
            default=_get_attr(config, "moe_intermediate_size"),
        ),
        "num_experts": _get_attr(config, "num_experts", "n_routed_experts"),
        "expert_hidden_size": _get_attr(config, "moe_intermediate_size"),
        "vocab_size": _get_attr(config, "vocab_size", "padded_vocab_size"),
        "attention_head_dim": _get_attr(config, "head_dim", "kv_channels"),
        "value_head_dim": _get_attr(config, "v_head_dim"),
        "q_lora_rank": _get_attr(config, "q_lora_rank"),
        "kv_lora_rank": _get_attr(config, "kv_lora_rank"),
        "qk_rope_head_dim": _get_attr(config, "qk_rope_head_dim"),
        "qk_nope_head_dim": _get_attr(config, "qk_nope_head_dim"),
        "supports_bias": bool(
            _get_attr(config, "attention_bias", default=False)
            or _get_attr(config, "qkv_bias", default=False)
            or _get_attr(config, "add_qkv_bias", default=False)
            or _get_attr(config, "mlp_bias", default=False)
        ),
    }
    return family, variant, {key: value for key, value in overrides.items() if value is not None}


def list_profiles() -> Mapping[tuple[ModelFamily, str], ArchitectureProfile]:
    """Return the seeded baseline profile registry."""

    return MappingProxyType(dict(_PROFILE_REGISTRY))


def get_profile(
    family: ModelFamily | str | None,
    variant: str | None,
    config: object | None = None,
    overrides: Mapping[str, object] | None = None,
) -> ArchitectureProfile:
    """Resolve a profile from the registry, optional config, and explicit overrides."""

    inferred_overrides: dict[str, Any] = {}
    resolved_family = _coerce_family(family)
    resolved_variant = _normalize_variant(variant)

    if config is not None:
        inferred_family, inferred_variant, inferred_overrides = _infer_from_config(config)
        resolved_family = resolved_family or inferred_family
        resolved_variant = resolved_variant or inferred_variant

    if resolved_family is None or resolved_variant is None:
        raise ValueError("family/variant must be provided or inferable from config.")

    try:
        base = _PROFILE_REGISTRY[(resolved_family, resolved_variant)]
    except KeyError as exc:
        raise KeyError(
            f"Unsupported profile {(resolved_family.value, resolved_variant)!r}. "
            f"Available variants: {[key for key in _PROFILE_REGISTRY if key[0] == resolved_family]}"
        ) from exc

    merged = dict(inferred_overrides)
    if overrides:
        merged.update(dict(overrides))

    if not merged:
        return base

    profile_kwargs = {field.name: getattr(base, field.name) for field in fields(base)}
    profile_kwargs.update(merged)
    metadata = dict(base.metadata)
    if "metadata" in merged and isinstance(merged["metadata"], Mapping):
        metadata.update(merged["metadata"])
    profile_kwargs["metadata"] = MappingProxyType(metadata)
    profile_kwargs["family"] = resolved_family
    profile_kwargs["variant"] = resolved_variant
    return ArchitectureProfile(**profile_kwargs)


def make_config_like(**kwargs: object) -> object:
    """Small helper for tests and downstream callers without transformers installed."""

    return SimpleNamespace(**kwargs)
