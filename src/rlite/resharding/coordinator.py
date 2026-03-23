"""Coordinator helpers for in-process and precomputed exchanges."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Condition
from typing import Iterable, Mapping, Protocol

from rlite.transport import RankDescriptor

from .types import TensorBindingManifest


class ExchangeCoordinator(Protocol):
    """Minimal control-plane contract used by the executor."""

    def publish_binding_manifests(
        self,
        rank: int,
        manifests: Iterable[TensorBindingManifest],
    ) -> None:
        ...

    def publish_rank_descriptor(self, rank: int, descriptor: RankDescriptor) -> None:
        ...

    def peer_descriptors_for(self, rank: int) -> Mapping[int, RankDescriptor]:
        ...

    def mark_transfers_complete(self, rank: int) -> None:
        ...

    def wait_for_sources(self, rank: int, source_ranks: Iterable[int]) -> None:
        ...


@dataclass
class InMemoryExchangeCoordinator:
    """Simple in-process coordinator used by tests and local orchestration."""

    _binding_manifests: dict[int, tuple[TensorBindingManifest, ...]] = field(default_factory=dict)
    _descriptors: dict[int, RankDescriptor] = field(default_factory=dict)
    _completed_ranks: set[int] = field(default_factory=set)
    _condition: Condition = field(default_factory=Condition)

    def publish_binding_manifests(
        self,
        rank: int,
        manifests: Iterable[TensorBindingManifest],
    ) -> None:
        with self._condition:
            self._binding_manifests[int(rank)] = tuple(manifests)
            self._condition.notify_all()

    def publish_rank_descriptor(self, rank: int, descriptor: RankDescriptor) -> None:
        with self._condition:
            self._descriptors[int(rank)] = descriptor
            self._condition.notify_all()

    def peer_descriptors_for(self, rank: int) -> Mapping[int, RankDescriptor]:
        with self._condition:
            return {
                peer_rank: descriptor
                for peer_rank, descriptor in self._descriptors.items()
                if int(peer_rank) != int(rank)
            }

    def mark_transfers_complete(self, rank: int) -> None:
        with self._condition:
            self._completed_ranks.add(int(rank))
            self._condition.notify_all()

    def wait_for_sources(self, rank: int, source_ranks: Iterable[int]) -> None:
        expected = {int(value) for value in source_ranks if int(value) != int(rank)}
        with self._condition:
            while not expected.issubset(self._completed_ranks):
                self._condition.wait()


@dataclass(frozen=True)
class FrozenExchangeCoordinator:
    """Read-only coordinator for already-published execution payloads."""

    peer_descriptors: Mapping[int, RankDescriptor]
    completed_ranks: tuple[int, ...] = ()

    def publish_binding_manifests(
        self,
        rank: int,
        manifests: Iterable[TensorBindingManifest],
    ) -> None:
        del rank, manifests

    def publish_rank_descriptor(self, rank: int, descriptor: RankDescriptor) -> None:
        del rank, descriptor

    def peer_descriptors_for(self, rank: int) -> Mapping[int, RankDescriptor]:
        return {
            int(peer_rank): descriptor
            for peer_rank, descriptor in self.peer_descriptors.items()
            if int(peer_rank) != int(rank)
        }

    def mark_transfers_complete(self, rank: int) -> None:
        del rank

    def wait_for_sources(self, rank: int, source_ranks: Iterable[int]) -> None:
        expected = {int(value) for value in source_ranks if int(value) != int(rank)}
        completed = {int(value) for value in self.completed_ranks}
        missing = expected.difference(completed)
        if missing:
            raise RuntimeError(
                "Missing precompleted source ranks for frozen exchange coordinator: "
                f"{sorted(missing)}"
            )
