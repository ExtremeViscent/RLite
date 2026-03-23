"""Rule registry and translation APIs for shard-aware weight mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from .types import (
    ArchitectureProfile,
    Framework,
    MappedTargetSpec,
    MappedTensorSpec,
    ModelFamily,
    PackingSpec,
    ParallelKind,
    ParallelSpec,
    TensorPackKind,
    TensorRule,
)


_TOKEN_RE = re.compile(r"{([a-z_]+)}")


def _require(value: int | None, field_name: str) -> int:
    if value is None:
        raise ValueError(f"profile.{field_name} is required for this translation.")
    return value


def _div(value: int, divisor: int, label: str) -> int:
    if divisor <= 0 or value % divisor != 0:
        raise ValueError(f"Expected {label} to divide {value}, got {divisor}.")
    return value // divisor


def _shared_expert_hidden_size(profile: ArchitectureProfile) -> int:
    shared_count = int(profile.metadata.get("num_shared_experts", 1))
    if profile.expert_hidden_size is not None:
        return profile.expert_hidden_size * shared_count
    return profile.ffn_hidden_size


def _template_fields(template: str) -> tuple[str, ...]:
    return tuple(match.group(1) for match in _TOKEN_RE.finditer(template))


def _compile_pattern(template: str) -> re.Pattern[str]:
    pieces: list[str] = []
    cursor = 0
    for match in _TOKEN_RE.finditer(template):
        pieces.append(re.escape(template[cursor : match.start()]))
        field_name = match.group(1)
        if field_name in {"layer", "expert"}:
            pieces.append(rf"(?P<{field_name}>\d+)")
        else:
            pieces.append(rf"(?P<{field_name}>[^.]+)")
        cursor = match.end()
    pieces.append(re.escape(template[cursor:]))
    return re.compile("^" + "".join(pieces) + "$")


def _contexts(
    rule: TensorRule,
    groups: Mapping[str, str],
    profile: ArchitectureProfile,
) -> tuple[Mapping[str, str], ...]:
    missing = set()
    for template in rule.canonical_templates + rule.render_templates:
        missing.update(field for field in _template_fields(template) if field not in groups)

    if not missing:
        return (MappingProxyType(dict(groups)),)
    if missing == {"expert"}:
        num_experts = _require(profile.num_experts, "num_experts")
        return tuple(
            MappingProxyType({**dict(groups), "expert": str(expert_idx)})
            for expert_idx in range(num_experts)
        )
    raise ValueError(f"Cannot materialize rule {rule.name!r}; missing placeholders {sorted(missing)}.")


def _render(template: str, groups: Mapping[str, str]) -> str:
    return template.format(**groups)


def _shape_vocab(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (_require(profile.vocab_size, "vocab_size"), profile.hidden_size)


def _shape_vocab_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (
        _div(_require(profile.vocab_size, "vocab_size"), profile.tensor_parallel_size, "tensor_parallel_size"),
        profile.hidden_size,
    )


def _shape_hidden(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int]:
    return (profile.hidden_size,)


def _shape_shared_gate(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (1, profile.hidden_size)


def _shape_q(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.q_projection_size, profile.hidden_size)


def _shape_q_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.local_q_projection_size, profile.hidden_size)


def _shape_k(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.k_projection_size, profile.hidden_size)


def _shape_k_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.local_k_projection_size, profile.hidden_size)


def _shape_v(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.v_projection_size, profile.hidden_size)


def _shape_v_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.local_v_projection_size, profile.hidden_size)


def _shape_qkv(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.q_projection_size + profile.k_projection_size + profile.v_projection_size, profile.hidden_size)


def _shape_qkv_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (
        profile.local_q_projection_size + profile.local_k_projection_size + profile.local_v_projection_size,
        profile.hidden_size,
    )


def _shape_attn_out(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.hidden_size, profile.attention_output_size)


def _shape_attn_out_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.hidden_size, profile.local_attention_output_size)


def _shape_ffn_col(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.ffn_hidden_size, profile.hidden_size)


def _shape_ffn_col_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.local_ffn_hidden_size, profile.hidden_size)


def _shape_gate_up(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (2 * profile.ffn_hidden_size, profile.hidden_size)


def _shape_gate_up_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (2 * profile.local_ffn_hidden_size, profile.hidden_size)


def _shape_ffn_row(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.hidden_size, profile.ffn_hidden_size)


def _shape_ffn_row_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.hidden_size, profile.local_ffn_hidden_size)


def _shape_moe_router(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (_require(profile.num_experts, "num_experts"), profile.hidden_size)


def _shape_moe_single(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.moe_hidden_size, profile.hidden_size)


def _shape_moe_single_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.local_moe_hidden_size, profile.hidden_size)


def _shape_moe_gate_up(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (2 * profile.moe_hidden_size, profile.hidden_size)


def _shape_moe_gate_up_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (2 * profile.local_moe_hidden_size, profile.hidden_size)


def _shape_moe_down(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.hidden_size, profile.moe_hidden_size)


def _shape_moe_down_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.hidden_size, profile.local_moe_hidden_size)


def _shape_grouped_expert_gate_up(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int, int]:
    return (_require(profile.num_experts, "num_experts"), 2 * profile.moe_hidden_size, profile.hidden_size)


def _shape_grouped_expert_gate_up_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int, int]:
    return (
        _require(profile.local_num_experts, "local_num_experts"),
        2 * profile.local_moe_hidden_size,
        profile.hidden_size,
    )


def _shape_grouped_expert_down(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int, int]:
    return (_require(profile.num_experts, "num_experts"), profile.hidden_size, profile.moe_hidden_size)


def _shape_grouped_expert_down_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int, int]:
    return (
        _require(profile.local_num_experts, "local_num_experts"),
        profile.hidden_size,
        profile.local_moe_hidden_size,
    )


def _shape_shared_col(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (_shared_expert_hidden_size(profile), profile.hidden_size)


def _shape_shared_col_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (
        _div(_shared_expert_hidden_size(profile), profile.tensor_parallel_size, "tensor_parallel_size"),
        profile.hidden_size,
    )


def _shape_shared_gate_up(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (2 * _shared_expert_hidden_size(profile), profile.hidden_size)


def _shape_shared_gate_up_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (
        2 * _div(_shared_expert_hidden_size(profile), profile.tensor_parallel_size, "tensor_parallel_size"),
        profile.hidden_size,
    )


def _shape_shared_row(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.hidden_size, _shared_expert_hidden_size(profile))


def _shape_shared_row_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (
        profile.hidden_size,
        _div(_shared_expert_hidden_size(profile), profile.tensor_parallel_size, "tensor_parallel_size"),
    )


def _shape_q_a(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (_require(profile.q_lora_rank, "q_lora_rank"), profile.hidden_size)


def _shape_q_b(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.q_projection_size, _require(profile.q_lora_rank, "q_lora_rank"))


def _shape_q_b_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (profile.local_q_projection_size, _require(profile.q_lora_rank, "q_lora_rank"))


def _shape_kv_a(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (
        _require(profile.kv_lora_rank, "kv_lora_rank") + _require(profile.qk_rope_head_dim, "qk_rope_head_dim"),
        profile.hidden_size,
    )


def _shape_kv_b(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    output = profile.num_attention_heads * (
        _require(profile.qk_nope_head_dim, "qk_nope_head_dim") + profile.v_head_dim
    )
    return (output, _require(profile.kv_lora_rank, "kv_lora_rank"))


def _shape_kv_b_local(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    output = profile.local_num_attention_heads * (
        _require(profile.qk_nope_head_dim, "qk_nope_head_dim") + profile.v_head_dim
    )
    return (output, _require(profile.kv_lora_rank, "kv_lora_rank"))


def _shape_fused_q_kv_a(profile: ArchitectureProfile, _: Mapping[str, str]) -> tuple[int, int]:
    return (_require(profile.q_lora_rank, "q_lora_rank") + _shape_kv_a(profile, {})[0], profile.hidden_size)


PACK_NONE = PackingSpec(kind=TensorPackKind.NONE, components=(), axis=None, order=())
PACK_QKV = PackingSpec(
    kind=TensorPackKind.FUSED_QKV,
    components=("q", "k", "v"),
    axis=0,
    order=("q", "k", "v"),
    component_size_rules=MappingProxyType(
        {
            "q": lambda profile: profile.q_projection_size,
            "k": lambda profile: profile.k_projection_size,
            "v": lambda profile: profile.v_projection_size,
        }
    ),
)
PACK_GATE_UP = PackingSpec(
    kind=TensorPackKind.FUSED_GATE_UP,
    components=("gate", "up"),
    axis=0,
    order=("gate", "up"),
    component_size_rules=MappingProxyType(
        {"gate": lambda profile: profile.ffn_hidden_size, "up": lambda profile: profile.ffn_hidden_size}
    ),
)
PACK_MOE_GATE_UP = PackingSpec(
    kind=TensorPackKind.FUSED_GATE_UP,
    components=("gate", "up"),
    axis=0,
    order=("gate", "up"),
    component_size_rules=MappingProxyType(
        {"gate": lambda profile: profile.moe_hidden_size, "up": lambda profile: profile.moe_hidden_size}
    ),
)
PACK_Q_KV = PackingSpec(
    kind=TensorPackKind.FUSED_Q_KV,
    components=("q_a", "kv_a"),
    axis=0,
    order=("q_a", "kv_a"),
    component_size_rules=MappingProxyType(
        {
            "q_a": lambda profile: _require(profile.q_lora_rank, "q_lora_rank"),
            "kv_a": lambda profile: _shape_kv_a(profile, {})[0],
        }
    ),
)


PAR_VOCAB_FIRST = ParallelSpec(ParallelKind.VOCAB, 0, _shape_vocab, _shape_vocab_local, True, True, pipeline_owner_hint="first_stage")
PAR_VOCAB_LAST = ParallelSpec(ParallelKind.VOCAB, 0, _shape_vocab, _shape_vocab_local, True, True, pipeline_owner_hint="last_stage")
PAR_REPL_BODY = ParallelSpec(ParallelKind.REPLICATED, None, _shape_hidden, _shape_hidden, False, False, pipeline_owner_hint="body")
PAR_REPL_LAST = ParallelSpec(ParallelKind.REPLICATED, None, _shape_hidden, _shape_hidden, False, False, pipeline_owner_hint="last_stage")
PAR_REPL_SHARED_GATE = ParallelSpec(ParallelKind.REPLICATED, None, _shape_shared_gate, _shape_shared_gate, False, False, pipeline_owner_hint="body")
PAR_Q = ParallelSpec(ParallelKind.TP_COL, 0, _shape_q, _shape_q_local, True, True, pipeline_owner_hint="body")
PAR_K = ParallelSpec(ParallelKind.TP_COL, 0, _shape_k, _shape_k_local, True, True, pipeline_owner_hint="body")
PAR_V = ParallelSpec(ParallelKind.TP_COL, 0, _shape_v, _shape_v_local, True, True, pipeline_owner_hint="body")
PAR_QKV = ParallelSpec(ParallelKind.TP_COL, 0, _shape_qkv, _shape_qkv_local, True, True, pipeline_owner_hint="body")
PAR_ATTN_OUT = ParallelSpec(ParallelKind.TP_ROW, 1, _shape_attn_out, _shape_attn_out_local, True, True, pipeline_owner_hint="body")
PAR_FFN_COL = ParallelSpec(ParallelKind.TP_COL, 0, _shape_ffn_col, _shape_ffn_col_local, True, True, pipeline_owner_hint="body")
PAR_GATE_UP = ParallelSpec(ParallelKind.TP_COL, 0, _shape_gate_up, _shape_gate_up_local, True, True, pipeline_owner_hint="body")
PAR_FFN_ROW = ParallelSpec(ParallelKind.TP_ROW, 1, _shape_ffn_row, _shape_ffn_row_local, True, True, pipeline_owner_hint="body")
PAR_ROUTER = ParallelSpec(ParallelKind.REPLICATED, None, _shape_moe_router, _shape_moe_router, False, False, expert_axis=0, pipeline_owner_hint="body")
PAR_MOE_COL = ParallelSpec(ParallelKind.TP_EP_COL, 0, _shape_moe_single, _shape_moe_single_local, True, True, expert_axis=0, pipeline_owner_hint="body")
PAR_MOE_GATE_UP = ParallelSpec(ParallelKind.TP_EP_COL, 0, _shape_moe_gate_up, _shape_moe_gate_up_local, True, True, expert_axis=0, pipeline_owner_hint="body")
PAR_MOE_ROW = ParallelSpec(ParallelKind.TP_EP_ROW, 1, _shape_moe_down, _shape_moe_down_local, True, True, expert_axis=0, pipeline_owner_hint="body")
PAR_GROUPED_MOE_GATE_UP = ParallelSpec(ParallelKind.TP_EP_COL, 1, _shape_grouped_expert_gate_up, _shape_grouped_expert_gate_up_local, True, True, expert_axis=0, pipeline_owner_hint="body")
PAR_GROUPED_MOE_ROW = ParallelSpec(ParallelKind.TP_EP_ROW, 2, _shape_grouped_expert_down, _shape_grouped_expert_down_local, True, True, expert_axis=0, pipeline_owner_hint="body")
PAR_SHARED_COL = ParallelSpec(ParallelKind.TP_COL, 0, _shape_shared_col, _shape_shared_col_local, True, True, pipeline_owner_hint="body")
PAR_SHARED_GATE_UP = ParallelSpec(ParallelKind.TP_COL, 0, _shape_shared_gate_up, _shape_shared_gate_up_local, True, True, pipeline_owner_hint="body")
PAR_SHARED_ROW = ParallelSpec(ParallelKind.TP_ROW, 1, _shape_shared_row, _shape_shared_row_local, True, True, pipeline_owner_hint="body")
PAR_Q_A = ParallelSpec(ParallelKind.REPLICATED, None, _shape_q_a, _shape_q_a, False, False, pipeline_owner_hint="body")
PAR_Q_B = ParallelSpec(ParallelKind.TP_COL, 0, _shape_q_b, _shape_q_b_local, True, True, pipeline_owner_hint="body")
PAR_KV_A = ParallelSpec(ParallelKind.REPLICATED, None, _shape_kv_a, _shape_kv_a, False, False, pipeline_owner_hint="body")
PAR_KV_B = ParallelSpec(ParallelKind.TP_COL, 0, _shape_kv_b, _shape_kv_b_local, True, True, pipeline_owner_hint="body")
PAR_FUSED_Q_KV_A = ParallelSpec(ParallelKind.REPLICATED, None, _shape_fused_q_kv_a, _shape_fused_q_kv_a, False, False, pipeline_owner_hint="body")


def _rule(
    name: str,
    framework: Framework,
    family: ModelFamily,
    variants: tuple[str, ...],
    pattern: str,
    canonical_templates: tuple[str, ...],
    tensor_role: str,
    packing: PackingSpec,
    parallel: ParallelSpec,
    *,
    render_templates: tuple[str, ...] | None = None,
    transpose: bool = False,
) -> TensorRule:
    return TensorRule(
        name=name,
        framework=framework,
        family=family,
        variants=variants,
        pattern=pattern,
        canonical_templates=canonical_templates,
        render_templates=render_templates or (pattern,),
        tensor_role=tensor_role,
        packing=packing,
        parallel=parallel,
        transpose=transpose,
    )


_RULES: list[TensorRule] = []


def _add_base_rules(
    rules: list[TensorRule],
    *,
    framework: Framework,
    family: ModelFamily,
    variants: tuple[str, ...],
    embed: str,
    lm_head: str,
    final_norm: str,
    attn_norm: str,
    ffn_norm: str,
) -> None:
    rules.extend(
        [
            _rule(f"{framework.value}_{family.value}_embed", framework, family, variants, embed, ("embeddings.word",), "embedding.word", PACK_NONE, PAR_VOCAB_FIRST),
            _rule(f"{framework.value}_{family.value}_lm_head", framework, family, variants, lm_head, ("output.lm_head",), "output.lm_head", PACK_NONE, PAR_VOCAB_LAST),
            _rule(f"{framework.value}_{family.value}_final_norm", framework, family, variants, final_norm, ("decoder.final_norm",), "decoder.final_norm", PACK_NONE, PAR_REPL_LAST),
            _rule(f"{framework.value}_{family.value}_attn_norm", framework, family, variants, attn_norm, ("layers.{layer}.attn_norm",), "attn.norm", PACK_NONE, PAR_REPL_BODY),
            _rule(f"{framework.value}_{family.value}_ffn_norm", framework, family, variants, ffn_norm, ("layers.{layer}.ffn_norm",), "mlp.norm", PACK_NONE, PAR_REPL_BODY),
        ]
    )


def _add_split_attention_rules(
    rules: list[TensorRule],
    *,
    framework: Framework,
    family: ModelFamily,
    variants: tuple[str, ...],
    q: str,
    k: str,
    v: str,
    out: str,
    transpose: bool = False,
) -> None:
    rules.extend(
        [
            _rule(f"{framework.value}_{family.value}_attn_q", framework, family, variants, q, ("layers.{layer}.attn.q",), "attn.q", PACK_NONE, PAR_Q, transpose=transpose),
            _rule(f"{framework.value}_{family.value}_attn_k", framework, family, variants, k, ("layers.{layer}.attn.k",), "attn.k", PACK_NONE, PAR_K, transpose=transpose),
            _rule(f"{framework.value}_{family.value}_attn_v", framework, family, variants, v, ("layers.{layer}.attn.v",), "attn.v", PACK_NONE, PAR_V, transpose=transpose),
            _rule(f"{framework.value}_{family.value}_attn_out", framework, family, variants, out, ("layers.{layer}.attn.out",), "attn.out", PACK_NONE, PAR_ATTN_OUT, transpose=transpose),
        ]
    )


def _add_fused_qkv_rule(
    rules: list[TensorRule],
    *,
    framework: Framework,
    family: ModelFamily,
    variants: tuple[str, ...],
    qkv: str,
    out: str,
    transpose: bool = False,
) -> None:
    rules.extend(
        [
            _rule(f"{framework.value}_{family.value}_attn_qkv", framework, family, variants, qkv, ("layers.{layer}.attn.q", "layers.{layer}.attn.k", "layers.{layer}.attn.v"), "attn.qkv", PACK_QKV, PAR_QKV, transpose=transpose),
            _rule(f"{framework.value}_{family.value}_attn_out", framework, family, variants, out, ("layers.{layer}.attn.out",), "attn.out", PACK_NONE, PAR_ATTN_OUT, transpose=transpose),
        ]
    )


def _add_split_mlp_rules(
    rules: list[TensorRule],
    *,
    framework: Framework,
    family: ModelFamily,
    variants: tuple[str, ...],
    gate: str,
    up: str,
    down: str,
    transpose: bool = False,
) -> None:
    rules.extend(
        [
            _rule(f"{framework.value}_{family.value}_mlp_gate", framework, family, variants, gate, ("layers.{layer}.mlp.gate",), "mlp.gate", PACK_NONE, PAR_FFN_COL, transpose=transpose),
            _rule(f"{framework.value}_{family.value}_mlp_up", framework, family, variants, up, ("layers.{layer}.mlp.up",), "mlp.up", PACK_NONE, PAR_FFN_COL, transpose=transpose),
            _rule(f"{framework.value}_{family.value}_mlp_down", framework, family, variants, down, ("layers.{layer}.mlp.down",), "mlp.down", PACK_NONE, PAR_FFN_ROW, transpose=transpose),
        ]
    )


def _add_fused_gate_up_rules(
    rules: list[TensorRule],
    *,
    framework: Framework,
    family: ModelFamily,
    variants: tuple[str, ...],
    gate_up: str,
    down: str,
    transpose: bool = False,
) -> None:
    rules.extend(
        [
            _rule(f"{framework.value}_{family.value}_mlp_gate_up", framework, family, variants, gate_up, ("layers.{layer}.mlp.gate", "layers.{layer}.mlp.up"), "mlp.gate_up", PACK_GATE_UP, PAR_GATE_UP, transpose=transpose),
            _rule(f"{framework.value}_{family.value}_mlp_down", framework, family, variants, down, ("layers.{layer}.mlp.down",), "mlp.down", PACK_NONE, PAR_FFN_ROW, transpose=transpose),
        ]
    )


def _add_classic_mlp_rules(
    rules: list[TensorRule],
    *,
    framework: Framework,
    family: ModelFamily,
    variants: tuple[str, ...],
    fc_in: str,
    fc_out: str,
    transpose: bool = False,
) -> None:
    rules.extend(
        [
            _rule(f"{framework.value}_{family.value}_mlp_fc_in", framework, family, variants, fc_in, ("layers.{layer}.mlp.up",), "mlp.up", PACK_NONE, PAR_FFN_COL, transpose=transpose),
            _rule(f"{framework.value}_{family.value}_mlp_fc_out", framework, family, variants, fc_out, ("layers.{layer}.mlp.down",), "mlp.down", PACK_NONE, PAR_FFN_ROW, transpose=transpose),
        ]
    )


for family, variants in (
    (ModelFamily.LLAMA, ("llama2", "llama3")),
    (ModelFamily.QWEN, ("qwen2", "qwen2_5", "qwen2_moe")),
    (ModelFamily.GLM, ("glm4", "chatglm3")),
    (ModelFamily.GPT, ("gpt_dense",)),
):
    _add_base_rules(
        _RULES,
        framework=Framework.MEGATRON,
        family=family,
        variants=variants,
        embed="embedding.word_embeddings.weight",
        lm_head="output_layer.weight",
        final_norm="decoder.final_layernorm.weight",
        attn_norm="decoder.layers.{layer}.input_layernorm.weight",
        ffn_norm="decoder.layers.{layer}.pre_mlp_layernorm.weight",
    )

for family, variants in (
    (ModelFamily.LLAMA, ("llama2", "llama3")),
    (ModelFamily.QWEN, ("qwen2", "qwen2_5", "qwen2_moe")),
    (ModelFamily.GLM, ("glm4", "chatglm3")),
):
    _add_fused_qkv_rule(
        _RULES,
        framework=Framework.MEGATRON,
        family=family,
        variants=variants,
        qkv="decoder.layers.{layer}.self_attention.linear_qkv.weight",
        out="decoder.layers.{layer}.self_attention.linear_proj.weight",
    )
    _add_fused_gate_up_rules(
        _RULES,
        framework=Framework.MEGATRON,
        family=family,
        variants=variants,
        gate_up="decoder.layers.{layer}.mlp.linear_fc1.weight",
        down="decoder.layers.{layer}.mlp.linear_fc2.weight",
    )

_add_fused_qkv_rule(
    _RULES,
    framework=Framework.MEGATRON,
    family=ModelFamily.GPT,
    variants=("gpt_dense",),
    qkv="decoder.layers.{layer}.self_attention.linear_qkv.weight",
    out="decoder.layers.{layer}.self_attention.linear_proj.weight",
)
_add_classic_mlp_rules(
    _RULES,
    framework=Framework.MEGATRON,
    family=ModelFamily.GPT,
    variants=("gpt_dense",),
    fc_in="decoder.layers.{layer}.mlp.linear_fc1.weight",
    fc_out="decoder.layers.{layer}.mlp.linear_fc2.weight",
)
_RULES.extend(
    [
        _rule("megatron_qwen2_moe_router", Framework.MEGATRON, ModelFamily.QWEN, ("qwen2_moe",), "decoder.layers.{layer}.mlp.router.weight", ("layers.{layer}.moe.router",), "moe.router", PACK_NONE, PAR_ROUTER),
        _rule("megatron_qwen2_moe_expert_fc1", Framework.MEGATRON, ModelFamily.QWEN, ("qwen2_moe",), "decoder.layers.{layer}.mlp.experts.local_experts.{expert}.linear_fc1.weight", ("layers.{layer}.moe.experts.{expert}.gate", "layers.{layer}.moe.experts.{expert}.up"), "moe.experts.gate_up", PACK_MOE_GATE_UP, PAR_MOE_GATE_UP),
        _rule("megatron_qwen2_moe_expert_fc2", Framework.MEGATRON, ModelFamily.QWEN, ("qwen2_moe",), "decoder.layers.{layer}.mlp.experts.local_experts.{expert}.linear_fc2.weight", ("layers.{layer}.moe.experts.{expert}.down",), "moe.experts.down", PACK_NONE, PAR_MOE_ROW),
        _rule("megatron_qwen2_moe_shared_fc1", Framework.MEGATRON, ModelFamily.QWEN, ("qwen2_moe",), "decoder.layers.{layer}.mlp.shared_experts.linear_fc1.weight", ("layers.{layer}.moe.shared.gate", "layers.{layer}.moe.shared.up"), "moe.shared.gate_up", PACK_GATE_UP, PAR_SHARED_GATE_UP),
        _rule("megatron_qwen2_moe_shared_fc2", Framework.MEGATRON, ModelFamily.QWEN, ("qwen2_moe",), "decoder.layers.{layer}.mlp.shared_experts.linear_fc2.weight", ("layers.{layer}.moe.shared.down",), "moe.shared.down", PACK_NONE, PAR_SHARED_ROW),
        _rule("megatron_qwen2_moe_shared_gate", Framework.MEGATRON, ModelFamily.QWEN, ("qwen2_moe",), "decoder.layers.{layer}.mlp.shared_experts.gate_weight", ("layers.{layer}.moe.shared_gate",), "moe.shared_gate", PACK_NONE, PAR_REPL_SHARED_GATE),
    ]
)

_add_base_rules(
    _RULES,
    framework=Framework.MEGATRON,
    family=ModelFamily.DEEPSEEK,
    variants=("deepseek_v2", "deepseek_v3"),
    embed="embedding.word_embeddings.weight",
    lm_head="output_layer.weight",
    final_norm="decoder.final_layernorm.weight",
    attn_norm="decoder.layers.{layer}.input_layernorm.weight",
    ffn_norm="decoder.layers.{layer}.pre_mlp_layernorm.weight",
)
_RULES.extend(
    [
        _rule("megatron_deepseek_attn_q_proj", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2",), "decoder.layers.{layer}.self_attention.linear_q_proj.weight", ("layers.{layer}.attn.q",), "attn.q", PACK_NONE, PAR_Q),
        _rule("megatron_deepseek_attn_q_down", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v3",), "decoder.layers.{layer}.self_attention.linear_q_down_proj.weight", ("layers.{layer}.attn.q_a",), "attn.q_a", PACK_NONE, PAR_Q_A),
        _rule("megatron_deepseek_attn_q_up", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v3",), "decoder.layers.{layer}.self_attention.linear_q_up_proj.weight", ("layers.{layer}.attn.q_b",), "attn.q_b", PACK_NONE, PAR_Q_B),
        _rule("megatron_deepseek_attn_kv_down", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "decoder.layers.{layer}.self_attention.linear_kv_down_proj.weight", ("layers.{layer}.attn.kv_a",), "attn.kv_a", PACK_NONE, PAR_KV_A),
        _rule("megatron_deepseek_attn_qkv_down", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v3",), "decoder.layers.{layer}.self_attention.linear_qkv_down_proj.weight", ("layers.{layer}.attn.q_a", "layers.{layer}.attn.kv_a"), "attn.qkv_a", PACK_Q_KV, PAR_FUSED_Q_KV_A),
        _rule("megatron_deepseek_attn_kv_up", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "decoder.layers.{layer}.self_attention.linear_kv_up_proj.weight", ("layers.{layer}.attn.kv_b",), "attn.kv_b", PACK_NONE, PAR_KV_B),
        _rule("megatron_deepseek_attn_out", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "decoder.layers.{layer}.self_attention.linear_proj.weight", ("layers.{layer}.attn.out",), "attn.out", PACK_NONE, PAR_ATTN_OUT),
        _rule("megatron_deepseek_attn_q_norm", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v3",), "decoder.layers.{layer}.self_attention.q_layernorm.weight", ("layers.{layer}.attn.q_norm",), "attn.q_norm", PACK_NONE, PAR_REPL_BODY),
        _rule("megatron_deepseek_attn_kv_norm", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "decoder.layers.{layer}.self_attention.kv_layernorm.weight", ("layers.{layer}.attn.kv_norm",), "attn.kv_norm", PACK_NONE, PAR_REPL_BODY),
        _rule("megatron_deepseek_router", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "decoder.layers.{layer}.mlp.router.weight", ("layers.{layer}.moe.router",), "moe.router", PACK_NONE, PAR_ROUTER),
        _rule("megatron_deepseek_expert_fc1", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "decoder.layers.{layer}.mlp.experts.local_experts.{expert}.linear_fc1.weight", ("layers.{layer}.moe.experts.{expert}.gate", "layers.{layer}.moe.experts.{expert}.up"), "moe.experts.gate_up", PACK_MOE_GATE_UP, PAR_MOE_GATE_UP),
        _rule("megatron_deepseek_expert_fc2", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "decoder.layers.{layer}.mlp.experts.local_experts.{expert}.linear_fc2.weight", ("layers.{layer}.moe.experts.{expert}.down",), "moe.experts.down", PACK_NONE, PAR_MOE_ROW),
        _rule("megatron_deepseek_shared_fc1", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "decoder.layers.{layer}.mlp.shared_experts.linear_fc1.weight", ("layers.{layer}.moe.shared.gate", "layers.{layer}.moe.shared.up"), "moe.shared.gate_up", PACK_GATE_UP, PAR_SHARED_GATE_UP),
        _rule("megatron_deepseek_shared_fc2", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "decoder.layers.{layer}.mlp.shared_experts.linear_fc2.weight", ("layers.{layer}.moe.shared.down",), "moe.shared.down", PACK_NONE, PAR_SHARED_ROW),
        _rule("megatron_deepseek_shared_gate", Framework.MEGATRON, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "decoder.layers.{layer}.mlp.shared_experts.gate_weight", ("layers.{layer}.moe.shared_gate",), "moe.shared_gate", PACK_NONE, PAR_REPL_SHARED_GATE),
    ]
)


@dataclass(frozen=True)
class _CompiledRule:
    index: int
    rule: TensorRule
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class _TargetCandidate:
    index: int
    rule: TensorRule
    context: Mapping[str, str]
    canonical_names: tuple[str, ...]
    target_names: tuple[str, ...]


_COMPILED_RULES: tuple[_CompiledRule, ...] = ()


def _normalize_framework(framework: Framework | str) -> Framework:
    if isinstance(framework, Framework):
        return framework
    return Framework(str(framework).lower())


def _active_rules(framework: Framework, profile: ArchitectureProfile) -> tuple[_CompiledRule, ...]:
    return tuple(
        compiled
        for compiled in _COMPILED_RULES
        if compiled.rule.framework == framework
        and compiled.rule.family == profile.family
        and (not compiled.rule.variants or profile.variant in compiled.rule.variants)
    )


def _match_rule(
    name: str,
    framework: Framework,
    profile: ArchitectureProfile,
) -> tuple[_CompiledRule, Mapping[str, str]]:
    matches: list[tuple[_CompiledRule, Mapping[str, str]]] = []
    for compiled in _active_rules(framework, profile):
        match = compiled.pattern.match(name)
        if match:
            matches.append((compiled, MappingProxyType(match.groupdict())))
    if not matches:
        raise KeyError(
            f"No mapping rule found for {framework.value}:{name!r} with profile "
            f"{profile.family.value}/{profile.variant}."
        )
    matches.sort(key=lambda item: (len(item[0].rule.pattern), -item[0].index), reverse=True)
    return matches[0]


def _materialize_source_canonicals(
    rule: TensorRule,
    groups: Mapping[str, str],
    profile: ArchitectureProfile,
) -> tuple[str, ...]:
    values: list[str] = []
    for context in _contexts(rule, groups, profile):
        values.extend(_render(template, context) for template in rule.canonical_templates)
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _candidate_bundles(
    framework: Framework,
    groups: Mapping[str, str],
    profile: ArchitectureProfile,
) -> tuple[_TargetCandidate, ...]:
    candidates: list[_TargetCandidate] = []
    for compiled in _active_rules(framework, profile):
        for context in _contexts(compiled.rule, groups, profile):
            candidates.append(
                _TargetCandidate(
                    index=compiled.index,
                    rule=compiled.rule,
                    context=context,
                    canonical_names=tuple(
                        _render(template, context) for template in compiled.rule.canonical_templates
                    ),
                    target_names=tuple(
                        _render(template, context) for template in compiled.rule.render_templates
                    ),
                )
            )
    return tuple(candidates)


def resolve_rule(
    name: str,
    framework: Framework | str,
    profile: ArchitectureProfile,
) -> TensorRule:
    """Resolve the source rule used for a framework-specific parameter name."""

    normalized = _normalize_framework(framework)
    compiled, _ = _match_rule(name, normalized, profile)
    return compiled.rule


def translate_tensor(
    name: str,
    src: Framework | str,
    dst: Framework | str,
    profile: ArchitectureProfile,
    view: str = "logical",
) -> MappedTensorSpec:
    """Translate one source tensor name into canonical and destination materializations."""

    if view not in {"logical", "local_shard"}:
        raise ValueError(f"Unsupported view {view!r}; expected 'logical' or 'local_shard'.")

    src_framework = _normalize_framework(src)
    dst_framework = _normalize_framework(dst)
    compiled, groups = _match_rule(name, src_framework, profile)
    src_rule = compiled.rule
    canonical_names = _materialize_source_canonicals(src_rule, groups, profile)

    if src_framework == dst_framework:
        targets = tuple(
            MappedTargetSpec(
                name=_render(template, groups),
                canonical_names=canonical_names,
                logical_shape=src_rule.parallel.logical_shape(profile, groups),
                local_shape=src_rule.parallel.local_shape(profile, groups),
            )
            for template in src_rule.render_templates
        )
        return MappedTensorSpec(
            source_name=name,
            source_framework=src_framework,
            target_framework=dst_framework,
            view=view,
            rule_name=src_rule.name,
            tensor_role=src_rule.tensor_role,
            match_groups=groups,
            canonical_names=canonical_names,
            target_names=tuple(target.name for target in targets),
            packing=src_rule.packing,
            parallel=src_rule.parallel,
            source_logical_shape=src_rule.parallel.logical_shape(profile, groups),
            source_local_shape=src_rule.parallel.local_shape(profile, groups),
            targets=targets,
        )

    uncovered = set(canonical_names)
    chosen: list[_TargetCandidate] = []
    candidates = _candidate_bundles(dst_framework, groups, profile)

    while uncovered:
        best: _TargetCandidate | None = None
        best_overlap: tuple[str, ...] = ()
        best_score = (-1, -1, 0)
        for candidate in candidates:
            overlap = tuple(value for value in candidate.canonical_names if value in uncovered)
            if not overlap:
                continue
            score = (len(overlap), len(candidate.rule.canonical_templates), -candidate.index)
            if score > best_score:
                best = candidate
                best_overlap = overlap
                best_score = score
        if best is None:
            break
        chosen.append(best)
        uncovered.difference_update(best_overlap)

    if uncovered:
        raise KeyError(
            f"Unable to materialize canonical tensors {sorted(uncovered)} in destination "
            f"{dst_framework.value} for profile {profile.family.value}/{profile.variant}."
        )

    merged: dict[str, dict[str, object]] = {}
    for candidate in chosen:
        overlap = tuple(value for value in candidate.canonical_names if value in canonical_names)
        logical_shape = candidate.rule.parallel.logical_shape(profile, candidate.context)
        local_shape = candidate.rule.parallel.local_shape(profile, candidate.context)
        for target_name in candidate.target_names:
            entry = merged.setdefault(
                target_name,
                {
                    "canonical_names": [],
                    "logical_shape": logical_shape,
                    "local_shape": local_shape,
                },
            )
            entry["canonical_names"].extend(
                value for value in overlap if value not in entry["canonical_names"]
            )

    targets = tuple(
        MappedTargetSpec(
            name=target_name,
            canonical_names=tuple(entry["canonical_names"]),
            logical_shape=entry["logical_shape"],
            local_shape=entry["local_shape"],
        )
        for target_name, entry in merged.items()
    )

    return MappedTensorSpec(
        source_name=name,
        source_framework=src_framework,
        target_framework=dst_framework,
        view=view,
        rule_name=src_rule.name,
        tensor_role=src_rule.tensor_role,
        match_groups=groups,
        canonical_names=canonical_names,
        target_names=tuple(target.name for target in targets),
        packing=src_rule.packing,
        parallel=src_rule.parallel,
        source_logical_shape=src_rule.parallel.logical_shape(profile, groups),
        source_local_shape=src_rule.parallel.local_shape(profile, groups),
        targets=targets,
    )


def translate_key(
    name: str,
    src: Framework | str,
    dst: Framework | str,
    profile: ArchitectureProfile,
) -> list[str]:
    """Translate a single parameter name into one or more destination names."""

    return list(translate_tensor(name, src, dst, profile).target_names)


def translate_state_dict_keys(
    keys_or_state_dict: Mapping[str, object] | Iterable[str],
    src: Framework | str,
    dst: Framework | str,
    profile: ArchitectureProfile,
    view: str = "logical",
) -> dict[str, tuple[str, ...]]:
    """Translate a mapping or iterable of state-dict keys into destination key tuples."""

    keys = keys_or_state_dict.keys() if isinstance(keys_or_state_dict, Mapping) else keys_or_state_dict
    return {
        key: tuple(translate_tensor(key, src, dst, profile, view=view).target_names)
        for key in keys
    }

for family, variants in (
    (ModelFamily.LLAMA, ("llama2", "llama3")),
    (ModelFamily.QWEN, ("qwen2", "qwen2_5")),
):
    _add_base_rules(
        _RULES,
        framework=Framework.TRANSFORMERS,
        family=family,
        variants=variants,
        embed="model.embed_tokens.weight",
        lm_head="lm_head.weight",
        final_norm="model.norm.weight",
        attn_norm="model.layers.{layer}.input_layernorm.weight",
        ffn_norm="model.layers.{layer}.post_attention_layernorm.weight",
    )
    _add_split_attention_rules(
        _RULES,
        framework=Framework.TRANSFORMERS,
        family=family,
        variants=variants,
        q="model.layers.{layer}.self_attn.q_proj.weight",
        k="model.layers.{layer}.self_attn.k_proj.weight",
        v="model.layers.{layer}.self_attn.v_proj.weight",
        out="model.layers.{layer}.self_attn.o_proj.weight",
    )
    _add_split_mlp_rules(
        _RULES,
        framework=Framework.TRANSFORMERS,
        family=family,
        variants=variants,
        gate="model.layers.{layer}.mlp.gate_proj.weight",
        up="model.layers.{layer}.mlp.up_proj.weight",
        down="model.layers.{layer}.mlp.down_proj.weight",
    )

_add_base_rules(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.QWEN,
    variants=("qwen2_moe",),
    embed="model.embed_tokens.weight",
    lm_head="lm_head.weight",
    final_norm="model.norm.weight",
    attn_norm="model.layers.{layer}.input_layernorm.weight",
    ffn_norm="model.layers.{layer}.post_attention_layernorm.weight",
)
_add_split_attention_rules(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.QWEN,
    variants=("qwen2_moe",),
    q="model.layers.{layer}.self_attn.q_proj.weight",
    k="model.layers.{layer}.self_attn.k_proj.weight",
    v="model.layers.{layer}.self_attn.v_proj.weight",
    out="model.layers.{layer}.self_attn.o_proj.weight",
)
_RULES.extend(
    [
        _rule("transformers_qwen2_moe_router", Framework.TRANSFORMERS, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.gate.weight", ("layers.{layer}.moe.router",), "moe.router", PACK_NONE, PAR_ROUTER),
        _rule("transformers_qwen2_moe_experts_gate_up", Framework.TRANSFORMERS, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.experts.gate_up_proj", ("layers.{layer}.moe.experts.{expert}.gate", "layers.{layer}.moe.experts.{expert}.up"), "moe.experts.gate_up", PACK_MOE_GATE_UP, PAR_GROUPED_MOE_GATE_UP),
        _rule("transformers_qwen2_moe_experts_down", Framework.TRANSFORMERS, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.experts.down_proj", ("layers.{layer}.moe.experts.{expert}.down",), "moe.experts.down", PACK_NONE, PAR_GROUPED_MOE_ROW),
        _rule("transformers_qwen2_moe_shared_gate", Framework.TRANSFORMERS, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.shared_expert_gate.weight", ("layers.{layer}.moe.shared_gate",), "moe.shared_gate", PACK_NONE, PAR_REPL_SHARED_GATE),
        _rule("transformers_qwen2_moe_shared_gate_proj", Framework.TRANSFORMERS, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.shared_expert.gate_proj.weight", ("layers.{layer}.moe.shared.gate",), "moe.shared.gate", PACK_NONE, PAR_SHARED_COL),
        _rule("transformers_qwen2_moe_shared_up_proj", Framework.TRANSFORMERS, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.shared_expert.up_proj.weight", ("layers.{layer}.moe.shared.up",), "moe.shared.up", PACK_NONE, PAR_SHARED_COL),
        _rule("transformers_qwen2_moe_shared_down_proj", Framework.TRANSFORMERS, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.shared_expert.down_proj.weight", ("layers.{layer}.moe.shared.down",), "moe.shared.down", PACK_NONE, PAR_SHARED_ROW),
    ]
)

_add_base_rules(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.GLM,
    variants=("glm4",),
    embed="model.embed_tokens.weight",
    lm_head="lm_head.weight",
    final_norm="model.norm.weight",
    attn_norm="model.layers.{layer}.input_layernorm.weight",
    ffn_norm="model.layers.{layer}.post_attention_layernorm.weight",
)
_add_split_attention_rules(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.GLM,
    variants=("glm4",),
    q="model.layers.{layer}.self_attn.q_proj.weight",
    k="model.layers.{layer}.self_attn.k_proj.weight",
    v="model.layers.{layer}.self_attn.v_proj.weight",
    out="model.layers.{layer}.self_attn.o_proj.weight",
)
_add_fused_gate_up_rules(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.GLM,
    variants=("glm4",),
    gate_up="model.layers.{layer}.mlp.gate_up_proj.weight",
    down="model.layers.{layer}.mlp.down_proj.weight",
)

_add_base_rules(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.GLM,
    variants=("chatglm3",),
    embed="transformer.embedding.word_embeddings.weight",
    lm_head="transformer.output_layer.weight",
    final_norm="transformer.encoder.final_layernorm.weight",
    attn_norm="transformer.encoder.layers.{layer}.input_layernorm.weight",
    ffn_norm="transformer.encoder.layers.{layer}.post_attention_layernorm.weight",
)
_add_fused_qkv_rule(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.GLM,
    variants=("chatglm3",),
    qkv="transformer.encoder.layers.{layer}.self_attention.query_key_value.weight",
    out="transformer.encoder.layers.{layer}.self_attention.dense.weight",
)
_add_fused_gate_up_rules(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.GLM,
    variants=("chatglm3",),
    gate_up="transformer.encoder.layers.{layer}.mlp.dense_h_to_4h.weight",
    down="transformer.encoder.layers.{layer}.mlp.dense_4h_to_h.weight",
)

_add_base_rules(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.GPT,
    variants=("gpt_dense",),
    embed="transformer.wte.weight",
    lm_head="lm_head.weight",
    final_norm="transformer.ln_f.weight",
    attn_norm="transformer.h.{layer}.ln_1.weight",
    ffn_norm="transformer.h.{layer}.ln_2.weight",
)
_add_fused_qkv_rule(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.GPT,
    variants=("gpt_dense",),
    qkv="transformer.h.{layer}.attn.c_attn.weight",
    out="transformer.h.{layer}.attn.c_proj.weight",
    transpose=True,
)
_add_classic_mlp_rules(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.GPT,
    variants=("gpt_dense",),
    fc_in="transformer.h.{layer}.mlp.c_fc.weight",
    fc_out="transformer.h.{layer}.mlp.c_proj.weight",
    transpose=True,
)

_add_base_rules(
    _RULES,
    framework=Framework.TRANSFORMERS,
    family=ModelFamily.DEEPSEEK,
    variants=("deepseek_v2", "deepseek_v3"),
    embed="model.embed_tokens.weight",
    lm_head="lm_head.weight",
    final_norm="model.norm.weight",
    attn_norm="model.layers.{layer}.input_layernorm.weight",
    ffn_norm="model.layers.{layer}.post_attention_layernorm.weight",
)
_RULES.extend(
    [
        _rule("transformers_deepseek_v2_q", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2",), "model.layers.{layer}.self_attn.q_proj.weight", ("layers.{layer}.attn.q",), "attn.q", PACK_NONE, PAR_Q),
        _rule("transformers_deepseek_v3_q_a", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v3",), "model.layers.{layer}.self_attn.q_a_proj.weight", ("layers.{layer}.attn.q_a",), "attn.q_a", PACK_NONE, PAR_Q_A),
        _rule("transformers_deepseek_v3_q_norm", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v3",), "model.layers.{layer}.self_attn.q_a_layernorm.weight", ("layers.{layer}.attn.q_norm",), "attn.q_norm", PACK_NONE, PAR_REPL_BODY),
        _rule("transformers_deepseek_v3_q_b", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v3",), "model.layers.{layer}.self_attn.q_b_proj.weight", ("layers.{layer}.attn.q_b",), "attn.q_b", PACK_NONE, PAR_Q_B),
        _rule("transformers_deepseek_kv_a", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight", ("layers.{layer}.attn.kv_a",), "attn.kv_a", PACK_NONE, PAR_KV_A),
        _rule("transformers_deepseek_kv_norm", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.self_attn.kv_a_layernorm.weight", ("layers.{layer}.attn.kv_norm",), "attn.kv_norm", PACK_NONE, PAR_REPL_BODY),
        _rule("transformers_deepseek_kv_b", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.self_attn.kv_b_proj.weight", ("layers.{layer}.attn.kv_b",), "attn.kv_b", PACK_NONE, PAR_KV_B),
        _rule("transformers_deepseek_o_proj", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.self_attn.o_proj.weight", ("layers.{layer}.attn.out",), "attn.out", PACK_NONE, PAR_ATTN_OUT),
        _rule("transformers_deepseek_router", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.gate.weight", ("layers.{layer}.moe.router",), "moe.router", PACK_NONE, PAR_ROUTER),
        _rule("transformers_deepseek_experts_gate_up", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.experts.gate_up_proj", ("layers.{layer}.moe.experts.{expert}.gate", "layers.{layer}.moe.experts.{expert}.up"), "moe.experts.gate_up", PACK_MOE_GATE_UP, PAR_GROUPED_MOE_GATE_UP),
        _rule("transformers_deepseek_experts_down", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.experts.down_proj", ("layers.{layer}.moe.experts.{expert}.down",), "moe.experts.down", PACK_NONE, PAR_GROUPED_MOE_ROW),
        _rule("transformers_deepseek_shared_gate_proj", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.shared_experts.gate_proj.weight", ("layers.{layer}.moe.shared.gate",), "moe.shared.gate", PACK_NONE, PAR_SHARED_COL),
        _rule("transformers_deepseek_shared_up_proj", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.shared_experts.up_proj.weight", ("layers.{layer}.moe.shared.up",), "moe.shared.up", PACK_NONE, PAR_SHARED_COL),
        _rule("transformers_deepseek_shared_down_proj", Framework.TRANSFORMERS, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.shared_experts.down_proj.weight", ("layers.{layer}.moe.shared.down",), "moe.shared.down", PACK_NONE, PAR_SHARED_ROW),
    ]
)

for family, variants in (
    (ModelFamily.LLAMA, ("llama2", "llama3")),
    (ModelFamily.QWEN, ("qwen2", "qwen2_5", "qwen2_moe")),
    (ModelFamily.GLM, ("glm4",)),
):
    _add_base_rules(
        _RULES,
        framework=Framework.SGLANG,
        family=family,
        variants=variants,
        embed="model.embed_tokens.weight",
        lm_head="lm_head.weight",
        final_norm="model.norm.weight",
        attn_norm="model.layers.{layer}.input_layernorm.weight",
        ffn_norm="model.layers.{layer}.post_attention_layernorm.weight",
    )
    _add_fused_qkv_rule(
        _RULES,
        framework=Framework.SGLANG,
        family=family,
        variants=variants,
        qkv="model.layers.{layer}.self_attn.qkv_proj.weight",
        out="model.layers.{layer}.self_attn.o_proj.weight",
    )
    _add_fused_gate_up_rules(
        _RULES,
        framework=Framework.SGLANG,
        family=family,
        variants=variants,
        gate_up="model.layers.{layer}.mlp.gate_up_proj.weight",
        down="model.layers.{layer}.mlp.down_proj.weight",
    )

_RULES.extend(
    [
        _rule("sglang_qwen2_moe_router", Framework.SGLANG, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.gate.weight", ("layers.{layer}.moe.router",), "moe.router", PACK_NONE, PAR_ROUTER),
        _rule("sglang_qwen2_moe_shared_gate", Framework.SGLANG, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.shared_expert_gate.weight", ("layers.{layer}.moe.shared_gate",), "moe.shared_gate", PACK_NONE, PAR_REPL_SHARED_GATE),
        _rule("sglang_qwen2_moe_shared_gate_proj", Framework.SGLANG, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.shared_expert.gate_proj.weight", ("layers.{layer}.moe.shared.gate",), "moe.shared.gate", PACK_NONE, PAR_SHARED_COL),
        _rule("sglang_qwen2_moe_shared_up_proj", Framework.SGLANG, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.shared_expert.up_proj.weight", ("layers.{layer}.moe.shared.up",), "moe.shared.up", PACK_NONE, PAR_SHARED_COL),
        _rule("sglang_qwen2_moe_shared_down_proj", Framework.SGLANG, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.shared_expert.down_proj.weight", ("layers.{layer}.moe.shared.down",), "moe.shared.down", PACK_NONE, PAR_SHARED_ROW),
        _rule("sglang_qwen2_moe_experts_gate_up", Framework.SGLANG, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.experts.gate_up_proj", ("layers.{layer}.moe.experts.{expert}.gate", "layers.{layer}.moe.experts.{expert}.up"), "moe.experts.gate_up", PACK_MOE_GATE_UP, PAR_GROUPED_MOE_GATE_UP),
        _rule("sglang_qwen2_moe_experts_down", Framework.SGLANG, ModelFamily.QWEN, ("qwen2_moe",), "model.layers.{layer}.mlp.experts.down_proj", ("layers.{layer}.moe.experts.{expert}.down",), "moe.experts.down", PACK_NONE, PAR_GROUPED_MOE_ROW),
    ]
)

_add_base_rules(
    _RULES,
    framework=Framework.SGLANG,
    family=ModelFamily.DEEPSEEK,
    variants=("deepseek_v2", "deepseek_v3"),
    embed="model.embed_tokens.weight",
    lm_head="lm_head.weight",
    final_norm="model.norm.weight",
    attn_norm="model.layers.{layer}.input_layernorm.weight",
    ffn_norm="model.layers.{layer}.post_attention_layernorm.weight",
)
_RULES.extend(
    [
        _rule("sglang_deepseek_v2_q", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2",), "model.layers.{layer}.self_attn.q_proj.weight", ("layers.{layer}.attn.q",), "attn.q", PACK_NONE, PAR_Q),
        _rule("sglang_deepseek_v3_fused_qkv_a", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v3",), "model.layers.{layer}.self_attn.fused_qkv_a_proj_with_mqa.weight", ("layers.{layer}.attn.q_a", "layers.{layer}.attn.kv_a"), "attn.qkv_a", PACK_Q_KV, PAR_FUSED_Q_KV_A),
        _rule("sglang_deepseek_v3_q_b", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v3",), "model.layers.{layer}.self_attn.q_b_proj.weight", ("layers.{layer}.attn.q_b",), "attn.q_b", PACK_NONE, PAR_Q_B),
        _rule("sglang_deepseek_kv_a", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight", ("layers.{layer}.attn.kv_a",), "attn.kv_a", PACK_NONE, PAR_KV_A),
        _rule("sglang_deepseek_kv_b", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.self_attn.kv_b_proj.weight", ("layers.{layer}.attn.kv_b",), "attn.kv_b", PACK_NONE, PAR_KV_B),
        _rule("sglang_deepseek_q_norm", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v3",), "model.layers.{layer}.self_attn.q_a_layernorm.weight", ("layers.{layer}.attn.q_norm",), "attn.q_norm", PACK_NONE, PAR_REPL_BODY),
        _rule("sglang_deepseek_kv_norm", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.self_attn.kv_a_layernorm.weight", ("layers.{layer}.attn.kv_norm",), "attn.kv_norm", PACK_NONE, PAR_REPL_BODY),
        _rule("sglang_deepseek_o_proj", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.self_attn.o_proj.weight", ("layers.{layer}.attn.out",), "attn.out", PACK_NONE, PAR_ATTN_OUT),
        _rule("sglang_deepseek_router", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.gate.weight", ("layers.{layer}.moe.router",), "moe.router", PACK_NONE, PAR_ROUTER),
        _rule("sglang_deepseek_experts_gate_up", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.experts.gate_up_proj", ("layers.{layer}.moe.experts.{expert}.gate", "layers.{layer}.moe.experts.{expert}.up"), "moe.experts.gate_up", PACK_MOE_GATE_UP, PAR_GROUPED_MOE_GATE_UP),
        _rule("sglang_deepseek_experts_down", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.experts.down_proj", ("layers.{layer}.moe.experts.{expert}.down",), "moe.experts.down", PACK_NONE, PAR_GROUPED_MOE_ROW),
        _rule("sglang_deepseek_shared_gate_proj", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.shared_experts.gate_proj.weight", ("layers.{layer}.moe.shared.gate",), "moe.shared.gate", PACK_NONE, PAR_SHARED_COL),
        _rule("sglang_deepseek_shared_up_proj", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.shared_experts.up_proj.weight", ("layers.{layer}.moe.shared.up",), "moe.shared.up", PACK_NONE, PAR_SHARED_COL),
        _rule("sglang_deepseek_shared_down_proj", Framework.SGLANG, ModelFamily.DEEPSEEK, ("deepseek_v2", "deepseek_v3"), "model.layers.{layer}.mlp.shared_experts.down_proj.weight", ("layers.{layer}.moe.shared.down",), "moe.shared.down", PACK_NONE, PAR_SHARED_ROW),
    ]
)

_COMPILED_RULES = tuple(
    _CompiledRule(index=index, rule=rule, pattern=_compile_pattern(rule.pattern))
    for index, rule in enumerate(_RULES)
)
