"""High-level transport session API."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from .backends import AutoTransportBackend, BaseTransportBackend
from .types import CapabilityReport, RankDescriptor, TensorRegistration, TransferPlan, TransportResult


@dataclass
class TransportSession:
    """Owns local tensor registrations and executes transfer plans."""

    rank: int
    world_size: int
    host: str = ""
    nic_name: str = ""
    provider_name: str = "mock_loopback"
    backend: Optional[BaseTransportBackend] = None

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.host == "":
            self.host = socket.gethostname()
        if self.backend is None:
            self.backend = AutoTransportBackend()
        self._opened = False
        self.registrations: dict[str, TensorRegistration] = {}
        self.peer_descriptors: dict[int, RankDescriptor] = {}
        self._published_descriptor: Optional[RankDescriptor] = None

    def open(self, peer_descriptors: Optional[Mapping[int, RankDescriptor] | Iterable[RankDescriptor]] = None):
        if not self._opened:
            self.backend.open(self)
            self._opened = True
        if peer_descriptors is not None:
            self.install_peer_descriptors(peer_descriptors)
        return self

    def close(self) -> None:
        if self._opened:
            self.backend.close(self)
            self._opened = False

    def __enter__(self) -> "TransportSession":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def capability_report(self) -> CapabilityReport:
        return self.backend.probe_capabilities(self)

    def register_tensor(
        self,
        tensor_name: str,
        buffer: Any,
        **kwargs: Any,
    ) -> TensorRegistration:
        if not self._opened:
            self.open()
        registration = self.backend.register_tensor(self, tensor_name, buffer, **kwargs)
        self.registrations[tensor_name] = registration
        self._published_descriptor = None
        return registration

    def publish_descriptors(self) -> RankDescriptor:
        report = self.capability_report
        descriptor = RankDescriptor(
            rank=self.rank,
            host=self.host,
            nic_name=self.nic_name,
            provider_name=report.provider_name or self.provider_name,
            fabric_address=b"",
            cuda_device_id=None,
            gpu_uuid=None,
            memory_regions={
                tensor_name: registration.export_descriptor()
                for tensor_name, registration in self.registrations.items()
            },
            metadata={"world_size": str(self.world_size)},
        )
        self._published_descriptor = self.backend.publish_descriptor(self, descriptor)
        return self._published_descriptor

    def install_peer_descriptors(
        self,
        descriptors: Mapping[int, RankDescriptor] | Iterable[RankDescriptor],
    ) -> None:
        if not self._opened:
            self.open()
        if isinstance(descriptors, Mapping):
            items = descriptors.items()
        else:
            items = ((descriptor.rank, descriptor) for descriptor in descriptors)
        collected: list[RankDescriptor] = []
        for rank, descriptor in items:
            if int(rank) == self.rank:
                continue
            self.peer_descriptors[int(rank)] = descriptor
            collected.append(descriptor)
        if collected:
            self.backend.install_peer_descriptors(self, collected)

    def execute(
        self,
        plan: TransferPlan,
        rank_descriptors: Optional[Mapping[int, RankDescriptor] | Iterable[RankDescriptor]] = None,
    ) -> TransportResult:
        if not self._opened:
            self.open()
        if rank_descriptors is not None:
            self.install_peer_descriptors(rank_descriptors)
        return self.backend.execute(self, plan)
