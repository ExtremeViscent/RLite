"""Shared types for the shard-aware weight mapping utility."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Optional


class Framework(str, Enum):
    """Supported source and destination frameworks."""

    MEGATRON = "megatron"
    TRANSFORMERS = "transformers"
    SGLANG = "sglang"


class ModelFamily(str, Enum):
    """Supported model families."""

    LLAMA = "llama"
    QWEN = "qwen"
    GLM = "glm"
    GPT = "gpt"
    DEEPSEEK = "deepseek"


class TensorPackKind(str, Enum):
    """How logical tensor components are packed into a parameter."""

    NONE = "none"
    FUSED_QKV = "fused_qkv"
    FUSED_Q_KV = "fused_q_kv"
    FUSED_GATE_UP = "fused_gate_up"


class ParallelKind(str, Enum):
    """How a tensor is sharded or replicated."""

    REPLICATED = "replicated"
    VOCAB = "vocab"
    TP_COL = "tp_col"
    TP_ROW = "tp_row"
    TP_COL_REPLICATED = "tp_col_replicated"
    TP_EP_COL = "tp_ep_col"
    TP_EP_ROW = "tp_ep_row"


Shape = tuple[int, ...]
ShapeFn = Callable[["ArchitectureProfile", Mapping[str, str]], Optional[Shape]]
SizeFn = Callable[["ArchitectureProfile"], int]


@dataclass(frozen=True)
class ArchitectureProfile:
    """Canonical architecture description used by the rule engine."""

    family: ModelFamily
    variant: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    ffn_hidden_size: int
    attention_layout: str
    mlp_layout: str
    qkv_order: tuple[str, ...] = ("q", "k", "v")
    num_experts: int | None = None
    expert_hidden_size: int | None = None
    vocab_size: int | None = None
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    attention_head_dim: int | None = None
    value_head_dim: int | None = None
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    qk_nope_head_dim: int | None = None
    supports_bias: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def head_dim(self) -> int:
        return self.attention_head_dim or (self.hidden_size // self.num_attention_heads)

    @property
    def v_head_dim(self) -> int:
        return self.value_head_dim or self.head_dim

    @property
    def q_projection_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def k_projection_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def v_projection_size(self) -> int:
        return self.num_key_value_heads * self.v_head_dim

    @property
    def attention_output_size(self) -> int:
        return self.num_attention_heads * self.v_head_dim

    @property
    def local_num_attention_heads(self) -> int:
        return self.num_attention_heads // self.tensor_parallel_size

    @property
    def local_num_key_value_heads(self) -> int:
        return max(1, self.num_key_value_heads // self.tensor_parallel_size)

    @property
    def local_q_projection_size(self) -> int:
        return self.local_num_attention_heads * self.head_dim

    @property
    def local_k_projection_size(self) -> int:
        return self.local_num_key_value_heads * self.head_dim

    @property
    def local_v_projection_size(self) -> int:
        return self.local_num_key_value_heads * self.v_head_dim

    @property
    def local_attention_output_size(self) -> int:
        return self.local_num_attention_heads * self.v_head_dim

    @property
    def moe_hidden_size(self) -> int:
        return self.expert_hidden_size or self.ffn_hidden_size

    @property
    def local_ffn_hidden_size(self) -> int:
        return self.ffn_hidden_size // self.tensor_parallel_size

    @property
    def local_moe_hidden_size(self) -> int:
        return self.moe_hidden_size // self.tensor_parallel_size

    @property
    def local_num_experts(self) -> int | None:
        if self.num_experts is None:
            return None
        return max(1, self.num_experts // self.expert_parallel_size)


@dataclass(frozen=True)
class PackingSpec:
    """Describes how a tensor groups logical components together."""

    kind: TensorPackKind
    components: tuple[str, ...]
    axis: int | None
    order: tuple[str, ...]
    component_size_rules: Mapping[str, SizeFn] = field(default_factory=dict)

    def component_sizes(self, profile: ArchitectureProfile) -> Mapping[str, int]:
        return MappingProxyType(
            {name: rule(profile) for name, rule in self.component_size_rules.items()}
        )


@dataclass(frozen=True)
class ParallelSpec:
    """Describes sharding semantics for a tensor."""

    kind: ParallelKind
    shard_axis: int | None
    logical_shape_fn: ShapeFn | None
    local_shape_fn: ShapeFn | None
    requires_gather: bool
    requires_scatter: bool
    expert_axis: int | None = None
    pipeline_owner_hint: str | None = None

    def logical_shape(
        self, profile: ArchitectureProfile, groups: Mapping[str, str]
    ) -> Shape | None:
        if self.logical_shape_fn is None:
            return None
        return self.logical_shape_fn(profile, groups)

    def local_shape(
        self, profile: ArchitectureProfile, groups: Mapping[str, str]
    ) -> Shape | None:
        if self.local_shape_fn is None:
            return None
        return self.local_shape_fn(profile, groups)


@dataclass(frozen=True)
class TensorRule:
    """Declarative translation rule for one framework-specific parameter form."""

    name: str
    framework: Framework
    family: ModelFamily
    variants: tuple[str, ...]
    pattern: str
    canonical_templates: tuple[str, ...]
    render_templates: tuple[str, ...]
    tensor_role: str
    packing: PackingSpec
    parallel: ParallelSpec
    transpose: bool = False
    reshape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class MappedTargetSpec:
    """A target materialization for one or more canonical components."""

    name: str
    canonical_names: tuple[str, ...]
    logical_shape: Shape | None
    local_shape: Shape | None


@dataclass(frozen=True)
class MappedTensorSpec:
    """Structured output of translate_tensor()."""

    source_name: str
    source_framework: Framework
    target_framework: Framework
    view: str
    rule_name: str
    tensor_role: str
    match_groups: Mapping[str, str]
    canonical_names: tuple[str, ...]
    target_names: tuple[str, ...]
    packing: PackingSpec
    parallel: ParallelSpec
    source_logical_shape: Shape | None
    source_local_shape: Shape | None
    targets: tuple[MappedTargetSpec, ...]
