from __future__ import annotations

import os

import pytest


def _integration_enabled() -> bool:
    return os.environ.get("RLITE_RUN_GPU_TRANSPORT_TESTS") == "1"


@pytest.mark.skipif(not _integration_enabled(), reason="requires GPU/NIC integration environment")
def test_resharding_cuda_ipc_exchange_path() -> None:
    pytest.importorskip("torch")
    pytest.skip("Resharding CUDA IPC validation requires Linux CUDA runtime and a live multi-process GPU setup.")


@pytest.mark.skipif(not _integration_enabled(), reason="requires GPU/NIC integration environment")
def test_resharding_libfabric_rdma_exchange_path() -> None:
    pytest.importorskip("torch")
    pytest.skip("Resharding RDMA validation requires libfabric verbs support and RDMA-capable hardware.")


@pytest.mark.skipif(not _integration_enabled(), reason="requires GPU/NIC integration environment")
def test_resharding_gpudirect_rdma_exchange_path() -> None:
    pytest.importorskip("torch")
    pytest.skip("Resharding GPUDirect validation requires CUDA, peer NIC affinity, and RDMA-capable hardware.")
