"""Framework-facing adapter contracts for future Megatron and SGLang wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .types import MemoryRegionDescriptor, RankDescriptor, TransferPlan, TransferTask


class CopyServiceLike(Protocol):
    """Megatron-compatible transport surface."""

    def submit_send(self, src_tensor, dest_rank: int):
        ...

    def submit_recv(self, dest_tensor, src_rank: int):
        ...

    def run(self):
        ...


@dataclass(frozen=True)
class MegatronCopyCommand:
    """Transport-neutral representation of a queued Megatron copy."""

    tensor_name: str
    src_rank: int
    dst_rank: int
    src_slice: tuple[int, int]
    dst_slice: tuple[int, int]
    dtype: str
    num_bytes: int
    src_mem_kind: str
    dst_mem_kind: str
    stream_id: int = 0
    priority: int = 0


def megatron_commands_to_plan(commands: Iterable[MegatronCopyCommand]) -> TransferPlan:
    """Convert Megatron-style queued copy commands into a transport plan."""

    return TransferPlan(
        tuple(
            TransferTask(
                tensor_name=command.tensor_name,
                src_rank=command.src_rank,
                dst_rank=command.dst_rank,
                src_slice=command.src_slice,
                dst_slice=command.dst_slice,
                dtype=command.dtype,
                num_bytes=command.num_bytes,
                src_mem_kind=command.src_mem_kind,
                dst_mem_kind=command.dst_mem_kind,
                stream_id=command.stream_id,
                priority=command.priority,
            )
            for command in commands
        )
    )


def sglang_remote_tensor_payload(rank_descriptor: RankDescriptor) -> dict[str, object]:
    """Serialize a rank descriptor into a future SGLang-friendly payload."""

    tensors = {}
    for name, region in rank_descriptor.memory_regions.items():
        assert isinstance(region, MemoryRegionDescriptor)
        tensors[name] = {
            "base_address": region.base_address,
            "num_bytes": region.num_bytes,
            "memory_kind": region.memory_kind.value,
            "device_id": region.device_id,
            "gpu_uuid": region.gpu_uuid,
            "remote_key": region.remote_key,
            "ipc_handle": region.ipc_handle.hex(),
            "provider_name": region.provider_name,
            "shape": list(region.shape),
            "dtype": region.dtype,
            "metadata": dict(region.metadata),
        }

    return {
        "rank": rank_descriptor.rank,
        "host": rank_descriptor.host,
        "nic_name": rank_descriptor.nic_name,
        "provider_name": rank_descriptor.provider_name,
        "fabric_address": rank_descriptor.fabric_address.hex(),
        "cuda_device_id": rank_descriptor.cuda_device_id,
        "gpu_uuid": rank_descriptor.gpu_uuid,
        "memory_regions": tensors,
        "metadata": dict(rank_descriptor.metadata),
    }
