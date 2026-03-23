"""Typed contracts for the RLite transport layer."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


DTYPE_BYTE_SIZES = {
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
}


class MemoryKind(str, Enum):
    """Transport-visible memory location."""

    CPU = "cpu"
    CUDA = "cuda"


class TransferPath(str, Enum):
    """Concrete path used to move a tensor slice."""

    ALIAS = "alias"
    MEMCPY = "memcpy"
    CUDA_IPC = "cuda_ipc"
    GDRCOPY = "gdrcopy"
    LIBFABRIC_RMA = "libfabric_rma"
    STAGED_HOST = "staged_host"
    MOCK_LOOPBACK = "mock_loopback"
    UNAVAILABLE = "unavailable"


def _normalize_memory_kind(value: MemoryKind | str) -> MemoryKind:
    if isinstance(value, MemoryKind):
        return value
    return MemoryKind(str(value).lower())


def _normalize_transfer_path(value: TransferPath | str) -> TransferPath:
    if isinstance(value, TransferPath):
        return value
    return TransferPath(str(value).lower())


@dataclass(frozen=True)
class ByteRange:
    """A byte-addressed 1-D slice."""

    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.length < 0:
            raise ValueError("length must be non-negative")

    @property
    def end(self) -> int:
        return self.offset + self.length

    @classmethod
    def from_value(cls, value: "ByteRange | slice | Sequence[int]") -> "ByteRange":
        if isinstance(value, cls):
            return value
        if isinstance(value, slice):
            if value.step not in (None, 1):
                raise ValueError("slice step must be 1")
            if value.start is None or value.stop is None:
                raise ValueError("slice start/stop must be provided")
            return cls(offset=int(value.start), length=int(value.stop) - int(value.start))
        if len(value) != 2:
            raise ValueError("byte range tuples must contain exactly two integers")
        return cls(offset=int(value[0]), length=int(value[1]))


@dataclass(frozen=True)
class TransferTask:
    """A single planned byte-range copy between source and destination ranks."""

    tensor_name: str
    src_rank: int
    dst_rank: int
    src_slice: ByteRange | slice | Sequence[int]
    dst_slice: ByteRange | slice | Sequence[int]
    dtype: str
    num_bytes: int
    src_mem_kind: MemoryKind | str
    dst_mem_kind: MemoryKind | str
    stream_id: int = 0
    priority: int = 0
    preferred_path: Optional[TransferPath | str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "src_slice", ByteRange.from_value(self.src_slice))
        object.__setattr__(self, "dst_slice", ByteRange.from_value(self.dst_slice))
        object.__setattr__(self, "src_mem_kind", _normalize_memory_kind(self.src_mem_kind))
        object.__setattr__(self, "dst_mem_kind", _normalize_memory_kind(self.dst_mem_kind))
        if self.preferred_path is not None:
            object.__setattr__(
                self,
                "preferred_path",
                _normalize_transfer_path(self.preferred_path),
            )
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.src_rank < 0 or self.dst_rank < 0:
            raise ValueError("rank ids must be non-negative")
        if self.num_bytes <= 0:
            raise ValueError("num_bytes must be positive")
        if self.src_slice.length != self.num_bytes:
            raise ValueError("src_slice length must match num_bytes")
        if self.dst_slice.length != self.num_bytes:
            raise ValueError("dst_slice length must match num_bytes")
        if self.stream_id < 0:
            raise ValueError("stream_id must be non-negative")

    @property
    def item_size(self) -> Optional[int]:
        return DTYPE_BYTE_SIZES.get(self.dtype)


@dataclass(frozen=True)
class TransferPlan:
    """Ordered collection of transfer tasks."""

    tasks: Tuple[TransferTask, ...]
    version: str = "1"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.tasks:
            raise ValueError("transfer plan must contain at least one task")

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    @property
    def involved_ranks(self) -> Tuple[int, ...]:
        return tuple(sorted({task.src_rank for task in self.tasks} | {task.dst_rank for task in self.tasks}))

    def peers_for_rank(self, rank: int) -> Tuple[int, ...]:
        peers = set()
        for task in self.tasks:
            if task.src_rank == rank and task.dst_rank != rank:
                peers.add(task.dst_rank)
            elif task.dst_rank == rank and task.src_rank != rank:
                peers.add(task.src_rank)
        return tuple(sorted(peers))

    def tasks_for_rank(self, rank: int) -> Tuple[TransferTask, ...]:
        return tuple(task for task in self.tasks if task.src_rank == rank or task.dst_rank == rank)

    def destination_tasks_for_rank(self, rank: int) -> Tuple[TransferTask, ...]:
        return tuple(task for task in self.tasks if task.dst_rank == rank)


@dataclass
class TensorRegistration:
    """Local memory registration associated with a named tensor."""

    tensor_name: str
    buffer: Any
    num_bytes: int
    memory_kind: MemoryKind | str
    base_address: int = 0
    device_id: Optional[int] = None
    gpu_uuid: Optional[str] = None
    dtype: str = "uint8"
    shape: Tuple[int, ...] = ()
    requested_key: int = 0
    remote_key: int = 0
    provider_name: str = ""
    ipc_handle: bytes = b""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.memory_kind = _normalize_memory_kind(self.memory_kind)
        self.shape = tuple(int(value) for value in self.shape)
        self.metadata = dict(self.metadata)

    def as_memoryview(self, *, writable: bool = False) -> memoryview:
        view = memoryview(self.buffer).cast("B")
        if writable and view.readonly:
            raise TypeError(f"buffer for {self.tensor_name!r} is read-only")
        return view

    def export_descriptor(self) -> "MemoryRegionDescriptor":
        return MemoryRegionDescriptor(
            tensor_name=self.tensor_name,
            base_address=int(self.base_address),
            num_bytes=int(self.num_bytes),
            memory_kind=self.memory_kind,
            device_id=self.device_id,
            gpu_uuid=self.gpu_uuid,
            dtype=self.dtype,
            shape=self.shape,
            remote_key=int(self.remote_key),
            ipc_handle=bytes(self.ipc_handle),
            provider_name=self.provider_name,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class MemoryRegionDescriptor:
    """Exported region metadata shared with peers through the control plane."""

    tensor_name: str
    base_address: int
    num_bytes: int
    memory_kind: MemoryKind | str
    device_id: Optional[int] = None
    gpu_uuid: Optional[str] = None
    dtype: str = "uint8"
    shape: Tuple[int, ...] = ()
    remote_key: int = 0
    ipc_handle: bytes = b""
    provider_name: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_kind", _normalize_memory_kind(self.memory_kind))
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "ipc_handle", bytes(self.ipc_handle))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")


@dataclass(frozen=True)
class RankDescriptor:
    """Exported per-rank descriptor used for peer discovery."""

    rank: int
    host: str
    nic_name: str = ""
    provider_name: str = ""
    fabric_address: bytes = b""
    cuda_device_id: Optional[int] = None
    gpu_uuid: Optional[str] = None
    memory_regions: Mapping[str, MemoryRegionDescriptor] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fabric_address", bytes(self.fabric_address))
        object.__setattr__(self, "memory_regions", dict(self.memory_regions))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.rank < 0:
            raise ValueError("rank must be non-negative")


@dataclass(frozen=True)
class CapabilityReport:
    """Runtime capability probe result for a transport backend."""

    supports_fi_rma: bool
    supports_fi_hmem: bool
    supports_cuda_ipc: bool
    supports_gdrcopy: bool
    supports_peer_access: bool
    preferred_remote_path: TransferPath | str
    fallback_remote_path: TransferPath | str
    provider_name: str = ""
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preferred_remote_path",
            _normalize_transfer_path(self.preferred_remote_path),
        )
        object.__setattr__(
            self,
            "fallback_remote_path",
            _normalize_transfer_path(self.fallback_remote_path),
        )
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))


@dataclass(frozen=True)
class TransportResult:
    """Execution summary returned by a transport session."""

    completed_tasks: int
    bytes_copied: int
    path_counts: Mapping[TransferPath | str, int]
    peer_ranks: Tuple[int, ...]
    used_native: bool = False
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_paths = {
            _normalize_transfer_path(path): int(count) for path, count in self.path_counts.items()
        }
        object.__setattr__(self, "path_counts", normalized_paths)
        object.__setattr__(self, "peer_ranks", tuple(sorted(int(rank) for rank in self.peer_ranks)))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        if self.completed_tasks < 0:
            raise ValueError("completed_tasks must be non-negative")
        if self.bytes_copied < 0:
            raise ValueError("bytes_copied must be non-negative")


def buffer_address_from_view(view: memoryview) -> int:
    """Return the base address of a writable CPU buffer."""

    if view.readonly:
        raise TypeError("cannot take a stable address from a read-only buffer")
    return ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
