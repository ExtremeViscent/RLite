from __future__ import annotations

import os

import pytest


def _integration_enabled() -> bool:
    return os.environ.get("RLITE_RUN_GPU_TRANSPORT_TESTS") == "1"


@pytest.mark.skipif(not _integration_enabled(), reason="requires GPU/NIC integration environment")
def test_cuda_ipc_gpu_to_gpu_path() -> None:
    pytest.importorskip("torch")
    pytest.skip("CUDA IPC integration requires Linux CUDA runtime and the built native transport library")


@pytest.mark.skipif(not _integration_enabled(), reason="requires GPU/NIC integration environment")
def test_libfabric_gpudirect_rdma_path() -> None:
    pytest.importorskip("torch")
    pytest.skip("GPUDirect RDMA integration requires Linux libfabric verbs provider and InfiniBand hardware")


@pytest.mark.skipif(not _integration_enabled(), reason="requires GPU/NIC integration environment")
def test_cpu_gpu_mixed_transfer_path() -> None:
    pytest.importorskip("torch")
    pytest.skip("Mixed CPU/GPU integration requires Linux CUDA runtime and the built native transport library")
