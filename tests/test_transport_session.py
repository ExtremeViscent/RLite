from __future__ import annotations

from rlite.transport import LoopbackCoordinator, LoopbackTransportBackend, TransferPlan, TransferTask, TransportSession


def _make_session(rank: int, world_size: int, coordinator: LoopbackCoordinator) -> TransportSession:
    backend = LoopbackTransportBackend(coordinator=coordinator)
    session = TransportSession(rank=rank, world_size=world_size, backend=backend)
    session.open()
    return session


def test_session_executes_remote_cpu_copy() -> None:
    coordinator = LoopbackCoordinator()
    src = _make_session(0, 2, coordinator)
    dst = _make_session(1, 2, coordinator)

    src.register_tensor("weight", bytearray(b"abcdefgh"))
    dst.register_tensor("weight", bytearray(b"XXXXXXXX"))

    dst_descriptor = dst.publish_descriptors()
    src_descriptor = src.publish_descriptors()

    src.install_peer_descriptors([dst_descriptor])
    dst.install_peer_descriptors([src_descriptor])

    plan = TransferPlan(
        (
            TransferTask("weight", 0, 1, (2, 4), (1, 4), "uint8", 4, "cpu", "cpu"),
        )
    )

    src_result = src.execute(plan)
    dst_result = dst.execute(plan)

    assert src_result.completed_tasks == 1
    assert dst_result.completed_tasks == 0
    assert bytes(dst.registrations["weight"].as_memoryview()) == b"XcdefXXX"


def test_session_executes_many_to_many_plan_without_extra_peers() -> None:
    coordinator = LoopbackCoordinator()
    sessions = [_make_session(rank, 4, coordinator) for rank in range(4)]

    sessions[0].register_tensor("packed", bytearray(b"AAAABBBBCCCCDDDD"))
    sessions[1].register_tensor("packed", bytearray(b"1111222233334444"))
    sessions[2].register_tensor("packed", bytearray(b"................"))
    sessions[3].register_tensor("packed", bytearray(b"................"))

    descriptors = [session.publish_descriptors() for session in sessions]
    for session in sessions:
        session.install_peer_descriptors(descriptors)

    plan = TransferPlan(
        (
            TransferTask("packed", 0, 2, (0, 4), (0, 4), "uint8", 4, "cpu", "cpu"),
            TransferTask("packed", 1, 2, (8, 4), (4, 4), "uint8", 4, "cpu", "cpu"),
            TransferTask("packed", 0, 3, (12, 4), (0, 4), "uint8", 4, "cpu", "cpu"),
            TransferTask("packed", 1, 3, (0, 4), (4, 4), "uint8", 4, "cpu", "cpu"),
        )
    )

    results = [session.execute(plan) for session in sessions]

    assert results[0].completed_tasks == 2
    assert results[1].completed_tasks == 2
    assert results[2].peer_ranks == (0, 1)
    assert results[3].peer_ranks == (0, 1)
    assert bytes(sessions[2].registrations["packed"].as_memoryview())[:8] == b"AAAA3333"
    assert bytes(sessions[3].registrations["packed"].as_memoryview())[:8] == b"DDDD1111"


def test_native_probe_falls_back_cleanly_when_library_is_missing() -> None:
    session = TransportSession(rank=0, world_size=1)
    report = session.capability_report

    assert report.provider_name == "mock_loopback"
    assert report.supports_fi_rma is False
    assert report.fallback_remote_path.value == "mock_loopback"
