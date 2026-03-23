"""ctypes bindings for the optional native transport library."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
from pathlib import Path
from typing import Mapping, Optional

from .types import (
    CapabilityReport,
    MemoryKind,
    MemoryRegionDescriptor,
    RankDescriptor,
    TensorRegistration,
    TransferPath,
    TransferPlan,
    TransportResult,
    _normalize_memory_kind,
    _normalize_transfer_path,
)


RLITE_TRANSPORT_NAME_MAX = 128
RLITE_TRANSPORT_HOST_MAX = 128
RLITE_TRANSPORT_PROVIDER_MAX = 64
RLITE_TRANSPORT_UUID_MAX = 64
RLITE_TRANSPORT_ERROR_MAX = 512
RLITE_TRANSPORT_FABRIC_ADDR_MAX = 1024
RLITE_TRANSPORT_IPC_HANDLE_MAX = 128
_ERROR_BUFFER_BYTES = 2048

_MEMORY_KIND_TO_NATIVE = {
    MemoryKind.CPU: 0,
    MemoryKind.CUDA: 1,
}
_MEMORY_KIND_FROM_NATIVE = {value: key for key, value in _MEMORY_KIND_TO_NATIVE.items()}

_TRANSFER_PATH_TO_NATIVE = {
    TransferPath.ALIAS: 0,
    TransferPath.MEMCPY: 1,
    TransferPath.CUDA_IPC: 2,
    TransferPath.GDRCOPY: 3,
    TransferPath.LIBFABRIC_RMA: 4,
    TransferPath.STAGED_HOST: 5,
    TransferPath.MOCK_LOOPBACK: 6,
    TransferPath.UNAVAILABLE: 7,
}
_TRANSFER_PATH_FROM_NATIVE = {value: key for key, value in _TRANSFER_PATH_TO_NATIVE.items()}


def _normalize_string(value: Optional[str]) -> Optional[bytes]:
    if value is None or value == "":
        return None
    return value.encode("utf-8")


def _decode_bytes(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8")


def _memory_kind_to_native(value: MemoryKind | str) -> int:
    normalized = _normalize_memory_kind(value)
    return _MEMORY_KIND_TO_NATIVE[normalized]


def _memory_kind_from_native(value: int) -> MemoryKind:
    return _MEMORY_KIND_FROM_NATIVE[int(value)]


def _transfer_path_to_native(value: TransferPath | str | None) -> int:
    if value is None:
        return _TRANSFER_PATH_TO_NATIVE[TransferPath.UNAVAILABLE]
    normalized = _normalize_transfer_path(value)
    return _TRANSFER_PATH_TO_NATIVE[normalized]


def _transfer_path_from_native(value: int) -> TransferPath:
    return _TRANSFER_PATH_FROM_NATIVE[int(value)]


class _SessionHandle(ctypes.Structure):
    pass


_SessionHandlePointer = ctypes.POINTER(_SessionHandle)


class _SessionOptions(ctypes.Structure):
    _fields_ = [
        ("rank", ctypes.c_int32),
        ("world_size", ctypes.c_int32),
        ("host", ctypes.c_char_p),
        ("nic_name", ctypes.c_char_p),
        ("provider_name", ctypes.c_char_p),
    ]


class _CapabilityReport(ctypes.Structure):
    _fields_ = [
        ("supports_fi_rma", ctypes.c_uint8),
        ("supports_fi_hmem", ctypes.c_uint8),
        ("supports_cuda_ipc", ctypes.c_uint8),
        ("supports_gdrcopy", ctypes.c_uint8),
        ("supports_peer_access", ctypes.c_uint8),
        ("preferred_remote_path", ctypes.c_uint32),
        ("fallback_remote_path", ctypes.c_uint32),
        ("provider_name", ctypes.c_char * RLITE_TRANSPORT_PROVIDER_MAX),
        ("note", ctypes.c_char * 256),
    ]


class _RegionRegistration(ctypes.Structure):
    _fields_ = [
        ("tensor_name", ctypes.c_char_p),
        ("base_ptr", ctypes.c_void_p),
        ("num_bytes", ctypes.c_uint64),
        ("memory_kind", ctypes.c_uint32),
        ("device_id", ctypes.c_int32),
        ("gpu_uuid", ctypes.c_char_p),
        ("requested_key", ctypes.c_uint64),
    ]


class _RegionDescriptor(ctypes.Structure):
    _fields_ = [
        ("tensor_name", ctypes.c_char * RLITE_TRANSPORT_NAME_MAX),
        ("base_address", ctypes.c_uint64),
        ("num_bytes", ctypes.c_uint64),
        ("memory_kind", ctypes.c_uint32),
        ("device_id", ctypes.c_int32),
        ("gpu_uuid", ctypes.c_char * RLITE_TRANSPORT_UUID_MAX),
        ("remote_key", ctypes.c_uint64),
        ("ipc_handle", ctypes.c_uint8 * RLITE_TRANSPORT_IPC_HANDLE_MAX),
        ("ipc_handle_length", ctypes.c_size_t),
    ]


class _PeerDescriptor(ctypes.Structure):
    _fields_ = [
        ("rank", ctypes.c_int32),
        ("host", ctypes.c_char * RLITE_TRANSPORT_HOST_MAX),
        ("nic_name", ctypes.c_char * RLITE_TRANSPORT_PROVIDER_MAX),
        ("provider_name", ctypes.c_char * RLITE_TRANSPORT_PROVIDER_MAX),
        ("fabric_address", ctypes.c_uint8 * RLITE_TRANSPORT_FABRIC_ADDR_MAX),
        ("fabric_address_length", ctypes.c_size_t),
        ("cuda_device_id", ctypes.c_int32),
        ("gpu_uuid", ctypes.c_char * RLITE_TRANSPORT_UUID_MAX),
    ]


class _TransferTask(ctypes.Structure):
    _fields_ = [
        ("tensor_name", ctypes.c_char_p),
        ("src_rank", ctypes.c_int32),
        ("dst_rank", ctypes.c_int32),
        ("src_offset", ctypes.c_uint64),
        ("dst_offset", ctypes.c_uint64),
        ("num_bytes", ctypes.c_uint64),
        ("src_memory_kind", ctypes.c_uint32),
        ("dst_memory_kind", ctypes.c_uint32),
        ("preferred_path", ctypes.c_uint32),
        ("stream_id", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
    ]


class _ExecutionStats(ctypes.Structure):
    _fields_ = [
        ("bytes_copied", ctypes.c_uint64),
        ("completed_tasks", ctypes.c_uint32),
        ("path_counts", ctypes.c_uint32 * 8),
        ("last_error", ctypes.c_char * RLITE_TRANSPORT_ERROR_MAX),
    ]


class NativeTransportError(RuntimeError):
    """Raised when the native transport library is required but unavailable."""


def _capability_from_probe_payload(payload: Mapping[str, object]) -> CapabilityReport:
    return CapabilityReport(
        supports_fi_rma=bool(payload["supports_fi_rma"]),
        supports_fi_hmem=bool(payload["supports_fi_hmem"]),
        supports_cuda_ipc=bool(payload["supports_cuda_ipc"]),
        supports_gdrcopy=bool(payload["supports_gdrcopy"]),
        supports_peer_access=bool(payload["supports_peer_access"]),
        preferred_remote_path=payload["preferred_remote_path"],
        fallback_remote_path=payload["fallback_remote_path"],
        provider_name=str(payload.get("provider_name", "")),
        notes=tuple(str(note) for note in payload.get("notes", ())),
    )


def _capability_from_native(report: _CapabilityReport) -> CapabilityReport:
    notes = (_decode_bytes(bytes(report.note)),) if bytes(report.note).split(b"\0", 1)[0] else ()
    return CapabilityReport(
        supports_fi_rma=bool(report.supports_fi_rma),
        supports_fi_hmem=bool(report.supports_fi_hmem),
        supports_cuda_ipc=bool(report.supports_cuda_ipc),
        supports_gdrcopy=bool(report.supports_gdrcopy),
        supports_peer_access=bool(report.supports_peer_access),
        preferred_remote_path=_transfer_path_from_native(int(report.preferred_remote_path)),
        fallback_remote_path=_transfer_path_from_native(int(report.fallback_remote_path)),
        provider_name=_decode_bytes(bytes(report.provider_name)),
        notes=notes,
    )


def _region_descriptor_from_native(
    descriptor: _RegionDescriptor,
    *,
    dtype: str,
    shape: tuple[int, ...],
    provider_name: str,
    metadata: Mapping[str, str],
) -> MemoryRegionDescriptor:
    return MemoryRegionDescriptor(
        tensor_name=_decode_bytes(bytes(descriptor.tensor_name)),
        base_address=int(descriptor.base_address),
        num_bytes=int(descriptor.num_bytes),
        memory_kind=_memory_kind_from_native(int(descriptor.memory_kind)),
        device_id=None if int(descriptor.device_id) < 0 else int(descriptor.device_id),
        gpu_uuid=_decode_bytes(bytes(descriptor.gpu_uuid)) or None,
        dtype=dtype,
        shape=shape,
        remote_key=int(descriptor.remote_key),
        ipc_handle=bytes(descriptor.ipc_handle[: int(descriptor.ipc_handle_length)]),
        provider_name=provider_name,
        metadata=metadata,
    )


def _peer_descriptor_to_python(
    descriptor: _PeerDescriptor,
    *,
    memory_regions: Mapping[str, MemoryRegionDescriptor],
    metadata: Mapping[str, str],
) -> RankDescriptor:
    return RankDescriptor(
        rank=int(descriptor.rank),
        host=_decode_bytes(bytes(descriptor.host)),
        nic_name=_decode_bytes(bytes(descriptor.nic_name)),
        provider_name=_decode_bytes(bytes(descriptor.provider_name)),
        fabric_address=bytes(descriptor.fabric_address[: int(descriptor.fabric_address_length)]),
        cuda_device_id=None if int(descriptor.cuda_device_id) < 0 else int(descriptor.cuda_device_id),
        gpu_uuid=_decode_bytes(bytes(descriptor.gpu_uuid)) or None,
        memory_regions=memory_regions,
        metadata=metadata,
    )


def _set_byte_array(field: ctypes.Array[ctypes.c_uint8], value: bytes) -> None:
    field[: len(value)] = value


class NativeTransportSession:
    """Live native transport session bound to a ctypes handle."""

    def __init__(
        self,
        library: "NativeTransportLibrary",
        handle: _SessionHandlePointer,
        capability_report: CapabilityReport,
    ) -> None:
        self._library = library
        self._handle = handle
        self.capability_report = capability_report

    @property
    def closed(self) -> bool:
        return not bool(self._handle)

    def close(self) -> None:
        if self.closed:
            return
        status = self._library._handle.rlite_transport_session_close(self._handle)
        self._handle = _SessionHandlePointer()
        if status != 0:
            raise NativeTransportError(self._library.status_string(status))

    def register_region(self, registration: TensorRegistration) -> MemoryRegionDescriptor:
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        native_registration = _RegionRegistration(
            tensor_name=registration.tensor_name.encode("utf-8"),
            base_ptr=ctypes.c_void_p(int(registration.base_address)),
            num_bytes=int(registration.num_bytes),
            memory_kind=_memory_kind_to_native(registration.memory_kind),
            device_id=-1 if registration.device_id is None else int(registration.device_id),
            gpu_uuid=_normalize_string(registration.gpu_uuid),
            requested_key=int(registration.requested_key),
        )
        native_descriptor = _RegionDescriptor()
        status = self._library._handle.rlite_transport_session_register_region(
            self._handle,
            ctypes.byref(native_registration),
            ctypes.byref(native_descriptor),
            error_buffer,
            len(error_buffer),
        )
        self._library._raise_for_status(status, error_buffer)
        return _region_descriptor_from_native(
            native_descriptor,
            dtype=registration.dtype,
            shape=registration.shape,
            provider_name=self.capability_report.provider_name,
            metadata=registration.metadata,
        )

    def query_local_peer_descriptor(
        self,
        *,
        memory_regions: Mapping[str, MemoryRegionDescriptor],
        metadata: Mapping[str, str],
    ) -> RankDescriptor:
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        native_peer = _PeerDescriptor()
        status = self._library._handle.rlite_transport_session_query_local_peer(
            self._handle,
            ctypes.byref(native_peer),
            error_buffer,
            len(error_buffer),
        )
        self._library._raise_for_status(status, error_buffer)
        return _peer_descriptor_to_python(
            native_peer,
            memory_regions=memory_regions,
            metadata=metadata,
        )

    def install_peer_descriptor(self, descriptor: RankDescriptor) -> None:
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        fabric_address = bytes(descriptor.fabric_address[:RLITE_TRANSPORT_FABRIC_ADDR_MAX])
        native_peer = _PeerDescriptor(
            rank=int(descriptor.rank),
            host=descriptor.host.encode("utf-8")[: RLITE_TRANSPORT_HOST_MAX - 1],
            nic_name=descriptor.nic_name.encode("utf-8")[: RLITE_TRANSPORT_PROVIDER_MAX - 1],
            provider_name=descriptor.provider_name.encode("utf-8")[: RLITE_TRANSPORT_PROVIDER_MAX - 1],
            fabric_address_length=len(fabric_address),
            cuda_device_id=-1 if descriptor.cuda_device_id is None else int(descriptor.cuda_device_id),
            gpu_uuid=(descriptor.gpu_uuid or "").encode("utf-8")[: RLITE_TRANSPORT_UUID_MAX - 1],
        )
        _set_byte_array(native_peer.fabric_address, fabric_address)

        region_count = len(descriptor.memory_regions)
        native_regions = (_RegionDescriptor * region_count)() if region_count else None
        if native_regions is not None:
            for index, region in enumerate(descriptor.memory_regions.values()):
                native_regions[index].tensor_name = region.tensor_name.encode("utf-8")[
                    : RLITE_TRANSPORT_NAME_MAX - 1
                ]
                native_regions[index].base_address = int(region.base_address)
                native_regions[index].num_bytes = int(region.num_bytes)
                native_regions[index].memory_kind = _memory_kind_to_native(region.memory_kind)
                native_regions[index].device_id = -1 if region.device_id is None else int(region.device_id)
                native_regions[index].gpu_uuid = (region.gpu_uuid or "").encode("utf-8")[
                    : RLITE_TRANSPORT_UUID_MAX - 1
                ]
                native_regions[index].remote_key = int(region.remote_key)
                ipc_handle = bytes(region.ipc_handle[:RLITE_TRANSPORT_IPC_HANDLE_MAX])
                native_regions[index].ipc_handle_length = len(ipc_handle)
                _set_byte_array(native_regions[index].ipc_handle, ipc_handle)

        status = self._library._handle.rlite_transport_session_install_peer(
            self._handle,
            ctypes.byref(native_peer),
            native_regions,
            region_count,
            error_buffer,
            len(error_buffer),
        )
        self._library._raise_for_status(status, error_buffer)

    def execute(self, plan: TransferPlan, *, local_rank: int) -> TransportResult:
        source_tasks = tuple(task for task in plan if task.src_rank == local_rank)
        if not source_tasks:
            return TransportResult(
                completed_tasks=0,
                bytes_copied=0,
                path_counts={},
                peer_ranks=plan.peers_for_rank(local_rank),
                used_native=True,
            )

        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        native_tasks = (_TransferTask * len(source_tasks))()
        for index, task in enumerate(source_tasks):
            native_tasks[index] = _TransferTask(
                tensor_name=task.tensor_name.encode("utf-8"),
                src_rank=int(task.src_rank),
                dst_rank=int(task.dst_rank),
                src_offset=int(task.src_slice.offset),
                dst_offset=int(task.dst_slice.offset),
                num_bytes=int(task.num_bytes),
                src_memory_kind=_memory_kind_to_native(task.src_mem_kind),
                dst_memory_kind=_memory_kind_to_native(task.dst_mem_kind),
                preferred_path=_transfer_path_to_native(task.preferred_path),
                stream_id=int(task.stream_id),
                priority=int(task.priority),
            )

        native_stats = _ExecutionStats()
        status = self._library._handle.rlite_transport_session_execute(
            self._handle,
            native_tasks,
            len(source_tasks),
            ctypes.byref(native_stats),
            error_buffer,
            len(error_buffer),
        )
        self._library._raise_for_status(status, error_buffer)
        path_counts = {
            _transfer_path_from_native(index): int(count)
            for index, count in enumerate(native_stats.path_counts)
            if int(count) > 0
        }
        warnings = ()
        if bytes(native_stats.last_error).split(b"\0", 1)[0]:
            warnings = (_decode_bytes(bytes(native_stats.last_error)),)
        return TransportResult(
            completed_tasks=int(native_stats.completed_tasks),
            bytes_copied=int(native_stats.bytes_copied),
            path_counts=path_counts,
            peer_ranks=plan.peers_for_rank(local_rank),
            used_native=True,
            warnings=warnings,
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class NativeTransportLibrary:
    """Thin ctypes wrapper around the native transport shared library."""

    def __init__(self, handle: ctypes.CDLL, path: str) -> None:
        self._handle = handle
        self.path = path

        self._handle.rlite_transport_runtime_version.restype = ctypes.c_char_p
        self._handle.rlite_transport_probe_json.restype = ctypes.c_void_p
        self._handle.rlite_transport_free_string.argtypes = [ctypes.c_void_p]
        self._handle.rlite_transport_free_string.restype = None
        self._handle.rlite_transport_status_string.argtypes = [ctypes.c_int]
        self._handle.rlite_transport_status_string.restype = ctypes.c_char_p

        self._handle.rlite_transport_session_open.argtypes = [
            ctypes.POINTER(_SessionOptions),
            ctypes.POINTER(_SessionHandlePointer),
            ctypes.POINTER(_CapabilityReport),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._handle.rlite_transport_session_open.restype = ctypes.c_int

        self._handle.rlite_transport_session_close.argtypes = [_SessionHandlePointer]
        self._handle.rlite_transport_session_close.restype = ctypes.c_int

        self._handle.rlite_transport_session_register_region.argtypes = [
            _SessionHandlePointer,
            ctypes.POINTER(_RegionRegistration),
            ctypes.POINTER(_RegionDescriptor),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._handle.rlite_transport_session_register_region.restype = ctypes.c_int

        self._handle.rlite_transport_session_query_local_peer.argtypes = [
            _SessionHandlePointer,
            ctypes.POINTER(_PeerDescriptor),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._handle.rlite_transport_session_query_local_peer.restype = ctypes.c_int

        self._handle.rlite_transport_session_install_peer.argtypes = [
            _SessionHandlePointer,
            ctypes.POINTER(_PeerDescriptor),
            ctypes.POINTER(_RegionDescriptor),
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._handle.rlite_transport_session_install_peer.restype = ctypes.c_int

        self._handle.rlite_transport_session_execute.argtypes = [
            _SessionHandlePointer,
            ctypes.POINTER(_TransferTask),
            ctypes.c_size_t,
            ctypes.POINTER(_ExecutionStats),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._handle.rlite_transport_session_execute.restype = ctypes.c_int

    def _raise_for_status(self, status: int, error_buffer: ctypes.Array[ctypes.c_char]) -> None:
        if status == 0:
            return
        message = error_buffer.value.decode("utf-8") if error_buffer.value else self.status_string(status)
        raise NativeTransportError(message)

    def status_string(self, status: int) -> str:
        return self._handle.rlite_transport_status_string(int(status)).decode("utf-8")

    def runtime_version(self) -> str:
        return self._handle.rlite_transport_runtime_version().decode("utf-8")

    def probe(self) -> CapabilityReport:
        value = self._handle.rlite_transport_probe_json()
        if not value:
            raise NativeTransportError("native transport probe returned no payload")
        try:
            payload = json.loads(ctypes.cast(value, ctypes.c_char_p).value.decode("utf-8"))
        finally:
            self._handle.rlite_transport_free_string(value)
        return _capability_from_probe_payload(payload)

    def open_session(
        self,
        *,
        rank: int,
        world_size: int,
        host: str,
        nic_name: str,
        provider_name: str,
    ) -> NativeTransportSession:
        options = _SessionOptions(
            rank=int(rank),
            world_size=int(world_size),
            host=_normalize_string(host),
            nic_name=_normalize_string(nic_name),
            provider_name=_normalize_string(provider_name),
        )
        session = _SessionHandlePointer()
        report = _CapabilityReport()
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        status = self._handle.rlite_transport_session_open(
            ctypes.byref(options),
            ctypes.byref(session),
            ctypes.byref(report),
            error_buffer,
            len(error_buffer),
        )
        self._raise_for_status(status, error_buffer)
        return NativeTransportSession(self, session, _capability_from_native(report))


def default_library_search_paths() -> list[str]:
    """Return likely locations for a built native library."""

    package_dir = Path(__file__).resolve().parent
    candidates = [
        package_dir / "_native" / "librlite_transport_native.so",
        package_dir / "_native" / "rlite_transport_native.so",
        package_dir / "_native" / "rlite_transport_native.dylib",
        package_dir / "_native" / "rlite_transport_native.dll",
    ]
    found = ctypes.util.find_library("rlite_transport_native")
    if found:
        candidates.append(Path(found))
    return [str(path) for path in candidates if path.exists()]


def load_native_transport(
    library_path: Optional[str] = None,
    *,
    required: bool = False,
) -> Optional[NativeTransportLibrary]:
    """Load the optional native transport shared library."""

    search_paths = [library_path] if library_path else default_library_search_paths()
    if not search_paths and not library_path:
        env_path = os.environ.get("RLITE_TRANSPORT_NATIVE_LIB")
        if env_path:
            search_paths.append(env_path)

    last_error: Optional[Exception] = None
    for candidate in search_paths:
        try:
            return NativeTransportLibrary(ctypes.CDLL(candidate), candidate)
        except OSError as exc:
            last_error = exc

    if required:
        raise NativeTransportError(
            "failed to load the native transport library"
            + (f": {last_error}" if last_error is not None else "")
        )
    return None


def unavailable_capability_report(reason: str) -> CapabilityReport:
    """Capability report used when the native transport library is absent."""

    return CapabilityReport(
        supports_fi_rma=False,
        supports_fi_hmem=False,
        supports_cuda_ipc=False,
        supports_gdrcopy=False,
        supports_peer_access=False,
        preferred_remote_path=TransferPath.UNAVAILABLE,
        fallback_remote_path=TransferPath.MOCK_LOOPBACK,
        provider_name="mock_loopback",
        notes=(reason,),
    )
