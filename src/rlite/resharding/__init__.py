"""Public exports for topology-aware weight exchange planning."""

from .coordinator import FrozenExchangeCoordinator, InMemoryExchangeCoordinator
from .executor import (
    abort_receive,
    commit_receive,
    execute_exchange_plan,
    frozen_coordinator_from_payload,
    prepare_receive,
)
from .planner import (
    build_binding_manifest,
    build_exchange_plan,
    normalize_expert_canonical_names,
    split_grouped_expert_record,
)
from .types import (
    BindingKind,
    ExchangeBundle,
    ExchangePlan,
    ExchangeResult,
    ExecutionSlice,
    FrameworkRole,
    FrameworkSnapshot,
    LinearSegment,
    LocalityTier,
    ParameterRecord,
    PendingReceive,
    TensorBinding,
    TensorBindingManifest,
    TopologyDecision,
    TopologyPolicy,
    WorkerEndpoint,
)

def collect_megatron_snapshot(*args, **kwargs):
    from ..integrations.megatron import collect_megatron_snapshot as _collect_megatron_snapshot

    return _collect_megatron_snapshot(*args, **kwargs)


def collect_sglang_snapshot(*args, **kwargs):
    from ..integrations.sglang import collect_sglang_snapshot as _collect_sglang_snapshot

    return _collect_sglang_snapshot(*args, **kwargs)


def collect_transformers_fsdp_snapshot(*args, **kwargs):
    from ..integrations.transformers import (
        collect_transformers_fsdp_snapshot as _collect_transformers_fsdp_snapshot,
    )

    return _collect_transformers_fsdp_snapshot(*args, **kwargs)


__all__ = [
    "BindingKind",
    "ExchangeBundle",
    "ExchangePlan",
    "ExchangeResult",
    "ExecutionSlice",
    "FrameworkRole",
    "FrameworkSnapshot",
    "FrozenExchangeCoordinator",
    "InMemoryExchangeCoordinator",
    "LinearSegment",
    "LocalityTier",
    "ParameterRecord",
    "PendingReceive",
    "TensorBinding",
    "TensorBindingManifest",
    "TopologyDecision",
    "TopologyPolicy",
    "WorkerEndpoint",
    "build_binding_manifest",
    "build_exchange_plan",
    "collect_megatron_snapshot",
    "collect_sglang_snapshot",
    "collect_transformers_fsdp_snapshot",
    "commit_receive",
    "execute_exchange_plan",
    "frozen_coordinator_from_payload",
    "normalize_expert_canonical_names",
    "prepare_receive",
    "split_grouped_expert_record",
    "abort_receive",
]
