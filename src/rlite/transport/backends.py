"""Backend implementations for the transport session."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional, TYPE_CHECKING

from .native import (
    NativeTransportError,
    NativeTransportLibrary,
    NativeTransportSession,
    load_native_transport,
    unavailable_capability_report,
)
from .types import (
    CapabilityReport,
    MemoryKind,
    RankDescriptor,
    TensorRegistration,
    TransferPath,
    TransferPlan,
    TransportResult,
    buffer_address_from_view,
)

if TYPE_CHECKING:
    from .session import TransportSession


def _stable_key(rank: int, tensor_name: str) -> int:
    digest = hashlib.sha256(f"{rank}:{tensor_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _infer_memory_kind(buffer: Any, explicit: Optional[MemoryKind | str]) -> MemoryKind:
    if explicit is not None:
        return explicit if isinstance(explicit, MemoryKind) else MemoryKind(str(explicit).lower())
    device = getattr(buffer, "device", None)
    if device is not None and str(device).startswith("cuda"):
        return MemoryKind.CUDA
    return MemoryKind.CPU


def _coerce_registration(
    session: "TransportSession",
    tensor_name: str,
    buffer: Any,
    *,
    memory_kind: Optional[MemoryKind | str] = None,
    device_id: Optional[int] = None,
    gpu_uuid: Optional[str] = None,
    dtype: Optional[str] = None,
    shape: Optional[tuple[int, ...]] = None,
    metadata: Optional[Mapping[str, str]] = None,
) -> TensorRegistration:
    inferred_kind = _infer_memory_kind(buffer, memory_kind)
    if hasattr(buffer, "data_ptr") and hasattr(buffer, "numel") and hasattr(buffer, "element_size"):
        num_bytes = int(buffer.numel()) * int(buffer.element_size())
        base_address = int(buffer.data_ptr())
        inferred_shape = tuple(int(value) for value in getattr(buffer, "shape", ()))
        inferred_dtype = dtype or str(getattr(buffer, "dtype", "uint8"))
        inferred_device_id = device_id
        if inferred_kind is MemoryKind.CUDA and inferred_device_id is None:
            inferred_device_id = int(getattr(getattr(buffer, "device", None), "index", 0) or 0)
    else:
        view = memoryview(buffer).cast("B")
        num_bytes = len(view)
        base_address = buffer_address_from_view(view) if not view.readonly else 0
        inferred_shape = shape or (num_bytes,)
        inferred_dtype = dtype or "uint8"
        inferred_device_id = device_id
        if inferred_kind is MemoryKind.CUDA:
            raise TypeError("CUDA registrations must expose data_ptr/numel/element_size")

    return TensorRegistration(
        tensor_name=tensor_name,
        buffer=buffer,
        num_bytes=num_bytes,
        memory_kind=inferred_kind,
        base_address=base_address,
        device_id=inferred_device_id,
        gpu_uuid=gpu_uuid,
        dtype=inferred_dtype,
        shape=shape or inferred_shape,
        requested_key=_stable_key(session.rank, tensor_name),
        remote_key=_stable_key(session.rank, tensor_name),
        provider_name=session.provider_name,
        metadata=metadata or {},
    )


def _normalize_descriptor_items(
    descriptors: Mapping[int, RankDescriptor] | Iterable[RankDescriptor],
) -> Iterable[tuple[int, RankDescriptor]]:
    if isinstance(descriptors, Mapping):
        return descriptors.items()
    return ((descriptor.rank, descriptor) for descriptor in descriptors)


class BaseTransportBackend(ABC):
    """Backend contract used by :class:`TransportSession`."""

    @abstractmethod
    def open(self, session: "TransportSession") -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self, session: "TransportSession") -> None:
        raise NotImplementedError

    @abstractmethod
    def probe_capabilities(self, session: "TransportSession") -> CapabilityReport:
        raise NotImplementedError

    @abstractmethod
    def register_tensor(
        self,
        session: "TransportSession",
        tensor_name: str,
        buffer: Any,
        **kwargs: Any,
    ) -> TensorRegistration:
        raise NotImplementedError

    @abstractmethod
    def publish_descriptor(
        self,
        session: "TransportSession",
        descriptor: RankDescriptor,
    ) -> RankDescriptor:
        raise NotImplementedError

    @abstractmethod
    def install_peer_descriptors(
        self,
        session: "TransportSession",
        descriptors: Mapping[int, RankDescriptor] | Iterable[RankDescriptor],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute(self, session: "TransportSession", plan: TransferPlan) -> TransportResult:
        raise NotImplementedError


class LoopbackCoordinator:
    """In-process registry used by the CPU-only loopback backend."""

    def __init__(self) -> None:
        self._sessions: Dict[int, "TransportSession"] = {}

    def attach(self, session: "TransportSession") -> None:
        self._sessions[session.rank] = session

    def detach(self, rank: int) -> None:
        self._sessions.pop(rank, None)

    def get_session(self, rank: int) -> "TransportSession":
        try:
            return self._sessions[rank]
        except KeyError as exc:
            raise KeyError(f"rank {rank} is not registered in the loopback coordinator") from exc


class LoopbackTransportBackend(BaseTransportBackend):
    """CPU-only backend that performs in-process copies for tests."""

    def __init__(self, coordinator: Optional[LoopbackCoordinator] = None) -> None:
        self.coordinator = coordinator or LoopbackCoordinator()

    def open(self, session: "TransportSession") -> None:
        self.coordinator.attach(session)

    def close(self, session: "TransportSession") -> None:
        self.coordinator.detach(session.rank)

    def probe_capabilities(self, session: "TransportSession") -> CapabilityReport:
        return CapabilityReport(
            supports_fi_rma=False,
            supports_fi_hmem=False,
            supports_cuda_ipc=False,
            supports_gdrcopy=False,
            supports_peer_access=False,
            preferred_remote_path=TransferPath.MOCK_LOOPBACK,
            fallback_remote_path=TransferPath.MEMCPY,
            provider_name="mock_loopback",
            notes=(
                "Loopback backend executes CPU copies in-process and is intended for sandbox tests.",
            ),
        )

    def register_tensor(
        self,
        session: "TransportSession",
        tensor_name: str,
        buffer: Any,
        **kwargs: Any,
    ) -> TensorRegistration:
        return _coerce_registration(session, tensor_name, buffer, **kwargs)

    def publish_descriptor(
        self,
        session: "TransportSession",
        descriptor: RankDescriptor,
    ) -> RankDescriptor:
        return descriptor

    def install_peer_descriptors(
        self,
        session: "TransportSession",
        descriptors: Mapping[int, RankDescriptor] | Iterable[RankDescriptor],
    ) -> None:
        del session, descriptors

    def execute(self, session: "TransportSession", plan: TransferPlan) -> TransportResult:
        path_counts = Counter()
        bytes_copied = 0
        completed = 0
        warnings = []

        for task in plan:
            if task.src_rank != session.rank:
                continue
            if task.tensor_name not in session.registrations:
                raise KeyError(f"source tensor {task.tensor_name!r} is not registered on rank {session.rank}")
            src_reg = session.registrations[task.tensor_name]
            if src_reg.memory_kind is not MemoryKind.CPU:
                raise RuntimeError("loopback backend only supports CPU source buffers")

            if task.dst_rank == session.rank:
                dst_session = session
            else:
                dst_session = self.coordinator.get_session(task.dst_rank)
                if task.dst_rank not in session.peer_descriptors:
                    warnings.append(
                        f"peer descriptor for rank {task.dst_rank} was not installed; "
                        "loopback fallback used coordinator state"
                    )

            if task.tensor_name not in dst_session.registrations:
                raise KeyError(
                    f"destination tensor {task.tensor_name!r} is not registered on rank {task.dst_rank}"
                )
            dst_reg = dst_session.registrations[task.tensor_name]
            if dst_reg.memory_kind is not MemoryKind.CPU:
                raise RuntimeError("loopback backend only supports CPU destination buffers")

            src_view = src_reg.as_memoryview()
            dst_view = dst_reg.as_memoryview(writable=True)
            src_chunk = src_view[task.src_slice.offset : task.src_slice.end]

            if task.src_rank == task.dst_rank and task.src_slice == task.dst_slice:
                path = TransferPath.ALIAS
            elif task.src_rank == task.dst_rank:
                path = TransferPath.MEMCPY
                dst_view[task.dst_slice.offset : task.dst_slice.end] = src_chunk.tobytes()
            else:
                path = TransferPath.MOCK_LOOPBACK
                dst_view[task.dst_slice.offset : task.dst_slice.end] = src_chunk.tobytes()

            bytes_copied += task.num_bytes
            completed += 1
            path_counts[path] += 1

        return TransportResult(
            completed_tasks=completed,
            bytes_copied=bytes_copied,
            path_counts=path_counts,
            peer_ranks=plan.peers_for_rank(session.rank),
            used_native=False,
            warnings=tuple(warnings),
        )


class NativeTransportBackend(BaseTransportBackend):
    """Backend that routes execution through the native libfabric transport."""

    def __init__(self, native_library: NativeTransportLibrary) -> None:
        self.native_library = native_library
        self.native_session: Optional[NativeTransportSession] = None
        self._capability_report: Optional[CapabilityReport] = None

    def open(self, session: "TransportSession") -> None:
        if self.native_session is not None:
            return
        self.native_session = self.native_library.open_session(
            rank=session.rank,
            world_size=session.world_size,
            host=session.host,
            nic_name=session.nic_name,
            provider_name=session.provider_name,
        )
        self._capability_report = self.native_session.capability_report

    def close(self, session: "TransportSession") -> None:
        del session
        if self.native_session is not None:
            self.native_session.close()
            self.native_session = None

    def probe_capabilities(self, session: "TransportSession") -> CapabilityReport:
        del session
        if self._capability_report is not None:
            return self._capability_report
        return self.native_library.probe()

    def register_tensor(
        self,
        session: "TransportSession",
        tensor_name: str,
        buffer: Any,
        **kwargs: Any,
    ) -> TensorRegistration:
        self.open(session)
        assert self.native_session is not None
        registration = _coerce_registration(session, tensor_name, buffer, **kwargs)
        descriptor = self.native_session.register_region(registration)
        registration.base_address = descriptor.base_address or registration.base_address
        registration.memory_kind = descriptor.memory_kind
        registration.device_id = descriptor.device_id
        registration.gpu_uuid = descriptor.gpu_uuid
        registration.remote_key = descriptor.remote_key
        registration.provider_name = descriptor.provider_name or session.provider_name
        registration.ipc_handle = descriptor.ipc_handle
        return registration

    def publish_descriptor(
        self,
        session: "TransportSession",
        descriptor: RankDescriptor,
    ) -> RankDescriptor:
        self.open(session)
        assert self.native_session is not None
        return self.native_session.query_local_peer_descriptor(
            memory_regions=descriptor.memory_regions,
            metadata=descriptor.metadata,
        )

    def install_peer_descriptors(
        self,
        session: "TransportSession",
        descriptors: Mapping[int, RankDescriptor] | Iterable[RankDescriptor],
    ) -> None:
        self.open(session)
        assert self.native_session is not None
        for rank, descriptor in _normalize_descriptor_items(descriptors):
            if int(rank) == session.rank:
                continue
            self.native_session.install_peer_descriptor(descriptor)

    def execute(self, session: "TransportSession", plan: TransferPlan) -> TransportResult:
        self.open(session)
        assert self.native_session is not None
        return self.native_session.execute(plan, local_rank=session.rank)


class AutoTransportBackend(BaseTransportBackend):
    """Backend that prefers the native transport but falls back to loopback tests."""

    def __init__(
        self,
        *,
        coordinator: Optional[LoopbackCoordinator] = None,
        native_library_path: Optional[str] = None,
    ) -> None:
        self.loopback = LoopbackTransportBackend(coordinator=coordinator)
        self.native_library = load_native_transport(native_library_path, required=False)
        self.native_backend = (
            NativeTransportBackend(self.native_library) if self.native_library is not None else None
        )
        self._active_backend: Optional[BaseTransportBackend] = None
        self._native_error: Optional[Exception] = None

    def _ensure_backend(self, session: "TransportSession") -> BaseTransportBackend:
        if self._active_backend is not None:
            return self._active_backend
        if self.native_backend is not None:
            try:
                self.native_backend.open(session)
                self._active_backend = self.native_backend
                return self._active_backend
            except Exception as exc:
                self._native_error = exc
        self.loopback.open(session)
        self._active_backend = self.loopback
        return self._active_backend

    def open(self, session: "TransportSession") -> None:
        self._ensure_backend(session)

    def close(self, session: "TransportSession") -> None:
        if self._active_backend is not None:
            self._active_backend.close(session)
        self._active_backend = None

    def probe_capabilities(self, session: "TransportSession") -> CapabilityReport:
        if self._active_backend is not None:
            return self._active_backend.probe_capabilities(session)
        if self.native_backend is None:
            return unavailable_capability_report(
                "native transport library not found; using loopback execution"
            )
        try:
            return self.native_backend.probe_capabilities(session)
        except Exception as exc:
            return unavailable_capability_report(f"native transport probe failed: {exc}")

    def register_tensor(
        self,
        session: "TransportSession",
        tensor_name: str,
        buffer: Any,
        **kwargs: Any,
    ) -> TensorRegistration:
        return self._ensure_backend(session).register_tensor(session, tensor_name, buffer, **kwargs)

    def publish_descriptor(
        self,
        session: "TransportSession",
        descriptor: RankDescriptor,
    ) -> RankDescriptor:
        return self._ensure_backend(session).publish_descriptor(session, descriptor)

    def install_peer_descriptors(
        self,
        session: "TransportSession",
        descriptors: Mapping[int, RankDescriptor] | Iterable[RankDescriptor],
    ) -> None:
        self._ensure_backend(session).install_peer_descriptors(session, descriptors)

    def execute(self, session: "TransportSession", plan: TransferPlan) -> TransportResult:
        return self._ensure_backend(session).execute(session, plan)
