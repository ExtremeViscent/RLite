"""Public exports for the RLite transport layer."""

from .adapters import CopyServiceLike, MegatronCopyCommand, megatron_commands_to_plan, sglang_remote_tensor_payload
from .backends import AutoTransportBackend, LoopbackCoordinator, LoopbackTransportBackend, NativeTransportBackend
from .native import NativeTransportError, NativeTransportLibrary, NativeTransportSession, load_native_transport
from .session import TransportSession
from .types import (
    ByteRange,
    CapabilityReport,
    MemoryKind,
    MemoryRegionDescriptor,
    RankDescriptor,
    TensorRegistration,
    TransferPath,
    TransferPlan,
    TransferTask,
    TransportResult,
)

__all__ = [
    "AutoTransportBackend",
    "ByteRange",
    "CapabilityReport",
    "CopyServiceLike",
    "LoopbackCoordinator",
    "LoopbackTransportBackend",
    "MegatronCopyCommand",
    "MemoryKind",
    "MemoryRegionDescriptor",
    "NativeTransportError",
    "NativeTransportBackend",
    "NativeTransportLibrary",
    "NativeTransportSession",
    "RankDescriptor",
    "TensorRegistration",
    "TransferPath",
    "TransferPlan",
    "TransferTask",
    "TransportResult",
    "TransportSession",
    "load_native_transport",
    "megatron_commands_to_plan",
    "sglang_remote_tensor_payload",
]
