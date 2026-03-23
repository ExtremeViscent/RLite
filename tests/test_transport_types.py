from __future__ import annotations

import pytest

from rlite.transport import (
    ByteRange,
    MemoryKind,
    MegatronCopyCommand,
    TransferPath,
    TransferPlan,
    TransferTask,
    megatron_commands_to_plan,
)


def test_transfer_task_normalizes_slices_and_kinds() -> None:
    task = TransferTask(
        tensor_name="decoder.layers.0.weight",
        src_rank=0,
        dst_rank=2,
        src_slice=slice(8, 16),
        dst_slice=(32, 8),
        dtype="float16",
        num_bytes=8,
        src_mem_kind="cpu",
        dst_mem_kind=MemoryKind.CPU,
        preferred_path="mock_loopback",
    )

    assert task.src_slice == ByteRange(8, 8)
    assert task.dst_slice == ByteRange(32, 8)
    assert task.src_mem_kind is MemoryKind.CPU
    assert task.preferred_path is TransferPath.MOCK_LOOPBACK


def test_transfer_task_rejects_mismatched_slice_lengths() -> None:
    with pytest.raises(ValueError):
        TransferTask(
            tensor_name="weight",
            src_rank=0,
            dst_rank=1,
            src_slice=(0, 4),
            dst_slice=(4, 8),
            dtype="uint8",
            num_bytes=4,
            src_mem_kind="cpu",
            dst_mem_kind="cpu",
        )


def test_transfer_plan_reports_peers_for_rank() -> None:
    plan = TransferPlan(
        (
            TransferTask("a", 0, 1, (0, 4), (0, 4), "uint8", 4, "cpu", "cpu"),
            TransferTask("a", 0, 2, (4, 4), (0, 4), "uint8", 4, "cpu", "cpu"),
            TransferTask("b", 2, 2, (0, 4), (8, 4), "uint8", 4, "cpu", "cpu"),
        )
    )

    assert plan.involved_ranks == (0, 1, 2)
    assert plan.peers_for_rank(0) == (1, 2)
    assert plan.peers_for_rank(2) == (0,)
    assert len(plan.destination_tasks_for_rank(2)) == 2


def test_megatron_commands_convert_to_transport_plan() -> None:
    plan = megatron_commands_to_plan(
        [
            MegatronCopyCommand(
                tensor_name="decoder.layers.1.weight",
                src_rank=3,
                dst_rank=4,
                src_slice=(64, 32),
                dst_slice=(0, 32),
                dtype="float16",
                num_bytes=32,
                src_mem_kind="cuda",
                dst_mem_kind="cuda",
            )
        ]
    )

    task = plan.tasks[0]
    assert task.tensor_name == "decoder.layers.1.weight"
    assert task.src_rank == 3
    assert task.dst_rank == 4
    assert task.num_bytes == 32
