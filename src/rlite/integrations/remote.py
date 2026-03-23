"""Remote control-plane helpers for explicit RLite receives."""

from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from ..resharding import FrameworkRole, WorkerEndpoint

def _pickleable(value):
    if isinstance(value, MappingProxyType):
        return {key: _pickleable(item) for key, item in value.items()}
    if is_dataclass(value):
        return value.__class__(
            **{field.name: _pickleable(getattr(value, field.name)) for field in fields(value)}
        )
    if isinstance(value, Mapping):
        return {key: _pickleable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_pickleable(item) for item in value)
    if isinstance(value, list):
        return [_pickleable(item) for item in value]
    if isinstance(value, set):
        return {_pickleable(item) for item in value}
    return value


def encode_payload(payload) -> str:
    """Serialize a Python payload into a URL-safe base64 string."""

    return base64.b64encode(
        pickle.dumps(_pickleable(payload), protocol=pickle.HIGHEST_PROTOCOL)
    ).decode("ascii")


def decode_payload(payload: str):
    """Deserialize a payload previously created by :func:`encode_payload`."""

    return pickle.loads(base64.b64decode(payload.encode("ascii")))


@dataclass(frozen=True)
class RemoteWorkerSpec:
    """Explicit topology description for one remote worker rank."""

    rank: int
    framework: str
    host: str
    tensor_parallel_rank: int
    tensor_parallel_size: int
    data_parallel_rank: int = 0
    data_parallel_size: int = 1
    pipeline_parallel_rank: int = 0
    pipeline_parallel_size: int = 1
    expert_parallel_rank: int = 0
    expert_parallel_size: int = 1
    moe_tensor_parallel_rank: int = 0
    moe_tensor_parallel_size: int = 1
    process_id: int = 0
    device_id: int | None = None
    nic_names: tuple[str, ...] = ()
    provider_names: tuple[str, ...] = ()
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nic_names", tuple(self.nic_names))
        object.__setattr__(self, "provider_names", tuple(self.provider_names))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def endpoint(self, *, role: FrameworkRole | str) -> WorkerEndpoint:
        return WorkerEndpoint(
            rank=self.rank,
            framework=self.framework,
            role=role,
            host=self.host,
            process_id=self.process_id,
            device_id=self.device_id,
            nic_names=self.nic_names,
            provider_names=self.provider_names,
            tensor_parallel_rank=self.tensor_parallel_rank,
            tensor_parallel_size=self.tensor_parallel_size,
            expert_parallel_rank=self.expert_parallel_rank,
            expert_parallel_size=self.expert_parallel_size,
            moe_tensor_parallel_rank=self.moe_tensor_parallel_rank,
            moe_tensor_parallel_size=self.moe_tensor_parallel_size,
            pipeline_parallel_rank=self.pipeline_parallel_rank,
            pipeline_parallel_size=self.pipeline_parallel_size,
            data_parallel_rank=self.data_parallel_rank,
            data_parallel_size=self.data_parallel_size,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class RemoteTopology:
    """Explicit manifest for the remote workers that participate in one sync."""

    workers: tuple[RemoteWorkerSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "workers", tuple(self.workers))

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(worker.rank for worker in self.workers)

    @property
    def world_size(self) -> int:
        if not self.workers:
            return 0
        return max(self.ranks) + 1

    def by_rank(self) -> dict[int, RemoteWorkerSpec]:
        return {worker.rank: worker for worker in self.workers}

    @classmethod
    def from_workers(cls, workers: Iterable[RemoteWorkerSpec]) -> "RemoteTopology":
        return cls(tuple(workers))

    @classmethod
    def from_grid(
        cls,
        *,
        framework: str,
        tp_size: int,
        dp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        rank_offset: int = 0,
        hosts: str | Sequence[str] = "",
        nic_names: Sequence[Sequence[str] | str] | Sequence[str] | str = (),
        provider_names: Sequence[Sequence[str] | str] | Sequence[str] | str = (),
        process_ids: Sequence[int] | int = 0,
        device_ids: Sequence[int | None] | int | None = None,
        metadata: Mapping[str, str] | Sequence[Mapping[str, str] | None] | None = None,
    ) -> "RemoteTopology":
        if tp_size <= 0 or dp_size <= 0 or pp_size <= 0 or ep_size <= 0:
            raise ValueError("tp_size, dp_size, pp_size, and ep_size must be positive")

        world_size = tp_size * dp_size * pp_size

        def _per_rank_values(value, *, default):
            if isinstance(value, (str, bytes)):
                return [value for _ in range(world_size)]
            if isinstance(value, Sequence):
                if len(value) == world_size:
                    return list(value)
                if len(value) == 1:
                    return [value[0] for _ in range(world_size)]
            return [default if value is None else value for _ in range(world_size)]

        def _as_tuple(value) -> tuple[str, ...]:
            if value in (None, ""):
                return ()
            if isinstance(value, str):
                return (value,)
            return tuple(str(item) for item in value)

        hosts_by_rank = _per_rank_values(hosts, default="")
        nics_by_rank = _per_rank_values(nic_names, default=())
        providers_by_rank = _per_rank_values(provider_names, default=())
        process_ids_by_rank = _per_rank_values(process_ids, default=0)
        device_ids_by_rank = _per_rank_values(device_ids, default=None)
        metadata_by_rank = _per_rank_values(metadata, default=None)

        workers = []
        for dp_rank in range(dp_size):
            for pp_rank in range(pp_size):
                for tp_rank in range(tp_size):
                    local_rank = dp_rank * (tp_size * pp_size) + pp_rank * tp_size + tp_rank
                    workers.append(
                        RemoteWorkerSpec(
                            rank=rank_offset + local_rank,
                            framework=framework,
                            host=str(hosts_by_rank[local_rank]),
                            tensor_parallel_rank=tp_rank,
                            tensor_parallel_size=tp_size,
                            data_parallel_rank=dp_rank,
                            data_parallel_size=dp_size,
                            pipeline_parallel_rank=pp_rank,
                            pipeline_parallel_size=pp_size,
                            expert_parallel_rank=0,
                            expert_parallel_size=ep_size,
                            moe_tensor_parallel_rank=tp_rank,
                            moe_tensor_parallel_size=tp_size,
                            process_id=int(process_ids_by_rank[local_rank] or 0),
                            device_id=device_ids_by_rank[local_rank],
                            nic_names=_as_tuple(nics_by_rank[local_rank]),
                            provider_names=_as_tuple(providers_by_rank[local_rank]),
                            metadata=metadata_by_rank[local_rank] or {},
                        )
                    )
        return cls(tuple(workers))
