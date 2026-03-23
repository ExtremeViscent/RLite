"""Shared types for topology-aware weight exchange planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from rlite.transport.types import (
    DTYPE_BYTE_SIZES,
    MemoryKind,
    RankDescriptor,
    TransferPath,
    TransferTask,
    TransportResult,
    _normalize_memory_kind,
    _normalize_transfer_path,
)
from rlite.weight_mapping.types import PackingSpec, ParallelSpec


class FrameworkRole(str, Enum):
    """Role a worker plays during one exchange."""

    SOURCE = "source"
    TARGET = "target"
    BIDIRECTIONAL = "bidirectional"


class BindingKind(str, Enum):
    """How a logical unit is materialized locally."""

    DIRECT = "direct"
    STAGED = "staged"


class LocalityTier(str, Enum):
    """Topology class used when picking a source or path."""

    LOCAL_EXACT = "local_exact"
    SAME_HOST_DIRECT = "same_host_direct"
    CROSS_HOST_RDMA = "cross_host_rdma"
    STAGED_FALLBACK = "staged_fallback"


SliceTuple = tuple[tuple[int, int], ...]


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return product


def _enum_value(value):
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True)
class WorkerEndpoint:
    """Runtime identity and topology metadata for one worker rank."""

    rank: int
    framework: str
    role: FrameworkRole | str
    host: str = ""
    process_id: int = 0
    device_id: Optional[int] = None
    gpu_uuid: Optional[str] = None
    nic_names: tuple[str, ...] = ()
    provider_names: tuple[str, ...] = ()
    tensor_parallel_rank: int = 0
    tensor_parallel_size: int = 1
    expert_parallel_rank: int = 0
    expert_parallel_size: int = 1
    moe_tensor_parallel_rank: int = 0
    moe_tensor_parallel_size: int = 1
    pipeline_parallel_rank: int = 0
    pipeline_parallel_size: int = 1
    data_parallel_rank: int = 0
    data_parallel_size: int = 1
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", FrameworkRole(_enum_value(self.role)))
        object.__setattr__(self, "nic_names", tuple(str(value) for value in self.nic_names))
        object.__setattr__(
            self,
            "provider_names",
            tuple(str(value) for value in self.provider_names),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        for name in (
            "tensor_parallel_size",
            "expert_parallel_size",
            "moe_tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class ParameterRecord:
    """Live local parameter metadata used to materialize transport bindings."""

    record_id: str
    framework_name: str
    tensor: Any
    dtype: str
    logical_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    actual_shape: tuple[int, ...]
    canonical_names: tuple[str, ...]
    packing: PackingSpec
    parallel: ParallelSpec
    tensor_role: str
    memory_kind: MemoryKind | str
    component_logical_sizes: tuple[int, ...] = ()
    component_local_sizes: tuple[int, ...] = ()
    transpose: bool = False
    reshape: tuple[int, ...] | None = None
    match_groups: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.memory_kind = _normalize_memory_kind(self.memory_kind)
        self.logical_shape = tuple(int(value) for value in self.logical_shape)
        self.local_shape = tuple(int(value) for value in self.local_shape)
        self.actual_shape = tuple(int(value) for value in self.actual_shape)
        self.canonical_names = tuple(str(value) for value in self.canonical_names)
        self.component_logical_sizes = tuple(int(value) for value in self.component_logical_sizes)
        self.component_local_sizes = tuple(int(value) for value in self.component_local_sizes)
        self.match_groups = dict(self.match_groups)
        self.metadata = dict(self.metadata)
        if not self.actual_shape:
            self.actual_shape = self.local_shape
        if not self.logical_shape:
            self.logical_shape = self.local_shape
        if not self.local_shape:
            self.local_shape = self.actual_shape

    @property
    def item_size(self) -> int:
        if self.dtype not in DTYPE_BYTE_SIZES:
            raise KeyError(f"Unsupported dtype {self.dtype!r} for weight exchange.")
        return DTYPE_BYTE_SIZES[self.dtype]

    @property
    def num_bytes(self) -> int:
        return _shape_product(self.local_shape) * self.item_size


@dataclass(frozen=True)
class FrameworkSnapshot:
    """All local records that belong to one framework worker."""

    endpoint: WorkerEndpoint
    records: tuple[ParameterRecord, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def framework(self) -> str:
        return self.endpoint.framework

    def records_by_id(self) -> dict[str, ParameterRecord]:
        return {record.record_id: record for record in self.records}


@dataclass(frozen=True)
class TensorBindingManifest:
    """Serializable description of one local exchange unit."""

    binding_id: str
    record_id: str
    rank: int
    exchange_key: str
    canonical_names: tuple[str, ...]
    framework_name: str
    framework_tensor_name: str
    binding_kind: BindingKind | str
    memory_kind: MemoryKind | str
    dtype: str
    logical_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    logical_slices: SliceTuple
    component_start: int = 0
    component_end: int = 0
    shard_axis: int | None = None
    preferred_path: TransferPath | str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_kind", BindingKind(_enum_value(self.binding_kind)))
        object.__setattr__(self, "memory_kind", _normalize_memory_kind(self.memory_kind))
        object.__setattr__(self, "logical_shape", tuple(int(value) for value in self.logical_shape))
        object.__setattr__(self, "local_shape", tuple(int(value) for value in self.local_shape))
        object.__setattr__(
            self,
            "logical_slices",
            tuple((int(start), int(stop)) for start, stop in self.logical_slices),
        )
        object.__setattr__(self, "canonical_names", tuple(self.canonical_names))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.component_end <= 0:
            object.__setattr__(self, "component_end", len(self.canonical_names))
        if self.preferred_path is not None:
            object.__setattr__(
                self,
                "preferred_path",
                _normalize_transfer_path(self.preferred_path),
            )

    @property
    def item_size(self) -> int:
        if self.dtype not in DTYPE_BYTE_SIZES:
            raise KeyError(f"Unsupported dtype {self.dtype!r} for weight exchange.")
        return DTYPE_BYTE_SIZES[self.dtype]

    @property
    def num_bytes(self) -> int:
        return _shape_product(self.local_shape) * self.item_size


@dataclass
class TensorBinding:
    """Live local binding that can register into the transport session."""

    manifest: TensorBindingManifest
    buffer: Any
    prepare_fn: Optional[Callable[[], None]] = None
    apply_fn: Optional[Callable[[], None]] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata)


@dataclass(frozen=True)
class ExchangeBundle:
    """One logical transfer unit chosen by the planner."""

    exchange_key: str
    canonical_names: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    target_record_ids: tuple[str, ...]
    source_ranks: tuple[int, ...]
    target_ranks: tuple[int, ...]
    split_reason: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_names", tuple(self.canonical_names))
        object.__setattr__(self, "source_record_ids", tuple(self.source_record_ids))
        object.__setattr__(self, "target_record_ids", tuple(self.target_record_ids))
        object.__setattr__(self, "source_ranks", tuple(int(value) for value in self.source_ranks))
        object.__setattr__(self, "target_ranks", tuple(int(value) for value in self.target_ranks))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TopologyDecision:
    """Chosen source/path metadata for one source-target pair."""

    src_rank: int
    dst_rank: int
    locality_tier: LocalityTier | str
    preferred_path: TransferPath | str
    source_nic_name: str = ""
    target_nic_name: str = ""
    source_provider_name: str = ""
    target_provider_name: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "locality_tier", LocalityTier(_enum_value(self.locality_tier)))
        object.__setattr__(
            self,
            "preferred_path",
            _normalize_transfer_path(self.preferred_path),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TopologyPolicy:
    """Planner knobs and optional hardware manifest overrides."""

    pinned_nics: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    pinned_providers: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    forced_paths: Mapping[tuple[int, int], TransferPath | str] = field(default_factory=dict)
    prefer_local: bool = True
    prefer_same_host: bool = True
    local_bonus: int = 0
    same_host_bonus: int = 10
    rdma_bonus: int = 20
    staged_penalty: int = 100
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pinned_nics = {int(rank): tuple(values) for rank, values in self.pinned_nics.items()}
        pinned_providers = {
            int(rank): tuple(values) for rank, values in self.pinned_providers.items()
        }
        forced_paths = {
            (int(src), int(dst)): _normalize_transfer_path(path)
            for (src, dst), path in self.forced_paths.items()
        }
        object.__setattr__(self, "pinned_nics", MappingProxyType(pinned_nics))
        object.__setattr__(self, "pinned_providers", MappingProxyType(pinned_providers))
        object.__setattr__(self, "forced_paths", MappingProxyType(forced_paths))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExecutionSlice:
    """Per-rank execution view of the global plan."""

    rank: int
    binding_manifests: tuple[TensorBindingManifest, ...]
    send_tasks: tuple[TransferTask, ...]
    target_binding_ids: tuple[str, ...]
    expected_source_ranks: tuple[int, ...]
    selected_nic_name: str = ""
    selected_provider_name: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_manifests", tuple(self.binding_manifests))
        object.__setattr__(self, "send_tasks", tuple(self.send_tasks))
        object.__setattr__(self, "target_binding_ids", tuple(self.target_binding_ids))
        object.__setattr__(
            self,
            "expected_source_ranks",
            tuple(sorted(int(rank) for rank in self.expected_source_ranks)),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExchangePlan:
    """Global result of planning a source-target exchange."""

    source_framework: str
    target_framework: str
    bundles: tuple[ExchangeBundle, ...]
    binding_manifests_by_rank: Mapping[int, tuple[TensorBindingManifest, ...]]
    execution_slices: Mapping[int, ExecutionSlice]
    topology_decisions: Mapping[tuple[int, int], TopologyDecision]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundles", tuple(self.bundles))
        object.__setattr__(
            self,
            "binding_manifests_by_rank",
            {
                int(rank): tuple(manifests)
                for rank, manifests in self.binding_manifests_by_rank.items()
            },
        )
        object.__setattr__(
            self,
            "execution_slices",
            {int(rank): value for rank, value in self.execution_slices.items()},
        )
        object.__setattr__(
            self,
            "topology_decisions",
            {tuple(map(int, key)): value for key, value in self.topology_decisions.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExchangeResult:
    """Local execution result for one worker rank."""

    rank: int
    transport_result: Optional[TransportResult]
    prepared_binding_ids: tuple[str, ...] = ()
    applied_binding_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prepared_binding_ids", tuple(self.prepared_binding_ids))
        object.__setattr__(self, "applied_binding_ids", tuple(self.applied_binding_ids))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))


@dataclass
class PendingReceive:
    """Prepared receive state that can later be committed or aborted."""

    rank: int
    transport_session: Any
    rank_descriptor: RankDescriptor
    bindings: Mapping[str, TensorBinding]
    target_binding_ids: tuple[str, ...]
    prepared_binding_ids: tuple[str, ...] = ()
    requires_staging: bool = False
    fallback_bytes: int = 0
    metadata: Mapping[str, str] = field(default_factory=dict)
    commit_actions: tuple[Callable[[], None], ...] = ()
    abort_actions: tuple[Callable[[], None], ...] = ()
    close_on_finish: bool = True
    _finished: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.target_binding_ids = tuple(self.target_binding_ids)
        self.prepared_binding_ids = tuple(self.prepared_binding_ids)
        self.metadata = dict(self.metadata)
        self.commit_actions = tuple(self.commit_actions)
        self.abort_actions = tuple(self.abort_actions)
