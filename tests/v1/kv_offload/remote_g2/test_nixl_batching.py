# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Remote-G2 multi-block NIXL transactions."""

from __future__ import annotations

import ctypes
from collections import deque
from typing import Any

import pytest

import vllm.v1.kv_offload.remote_g2.nixl_adapter as nixl_adapter
from vllm.v1.kv_offload.remote_g2.nixl_adapter import RawNixlRemoteG2Adapter


class _FakeNixlAgent:
    def __init__(self) -> None:
        self.desc_calls: list[tuple[list[tuple[int, int, int]], str]] = []
        self.xfers: list[tuple[str, Any, Any, str, str]] = []
        self.released: list[str] = []
        self.metadata_checks = 0
        self.poll_checks = 0
        self.initial_state = "DONE"
        self.transfer_states: deque[object] = deque()
        self.poll_states: deque[object] = deque()
        self.transfer_error: BaseException | None = None
        self.release_error: BaseException | None = None

    def register_memory(self, regions, *, mem_type: str) -> None:
        return

    def get_agent_metadata(self) -> bytes:
        return b"fake-target-metadata"

    def add_remote_agent(self, metadata: bytes) -> str:
        return "remote-handle"

    def get_xfer_descs(self, regions, *, mem_type: str):
        snapshot = [(int(ptr), int(size), int(device)) for ptr, size, device in regions]
        self.desc_calls.append((snapshot, mem_type))
        return f"descs-{len(self.desc_calls)}"

    def check_remote_metadata(self, handle: str) -> bool:
        self.metadata_checks += 1
        return True

    def initialize_xfer(
        self, op: str, local_descs: Any, remote_descs: Any, remote_handle: str
    ) -> str:
        handle = f"xfer-{len(self.xfers)}"
        self.xfers.append((op, local_descs, remote_descs, remote_handle, handle))
        return handle

    def transfer(self, handle: str) -> str:
        if self.transfer_error is not None:
            raise self.transfer_error
        if self.transfer_states:
            state = self.transfer_states.popleft()
            if isinstance(state, BaseException):
                raise state
            return str(state)
        return self.initial_state

    def check_xfer_state(self, handle: str) -> str:
        self.poll_checks += 1
        if not self.poll_states:
            return "DONE"
        state = self.poll_states.popleft()
        if isinstance(state, BaseException):
            raise state
        return str(state)

    def release_xfer_handle(self, handle: str) -> None:
        self.released.append(handle)
        if self.release_error is not None:
            raise self.release_error


def _real_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_layers: int = 2,
    max_blocks_per_xfer: int = 64,
) -> tuple[RawNixlRemoteG2Adapter, _FakeNixlAgent, list[int], list[int]]:
    fake = _FakeNixlAgent()
    monkeypatch.setattr(nixl_adapter, "NIXL_AVAILABLE", True)
    monkeypatch.setattr(nixl_adapter, "nixl_agent_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(nixl_adapter, "nixl_agent", lambda *args, **kwargs: fake)

    local_bases = [1_000 * (layer + 1) for layer in range(num_layers)]
    peer_bases = [10_000 * (layer + 1) for layer in range(num_layers)]
    adapter = RawNixlRemoteG2Adapter(
        "target",
        local_bases,
        [8_192] * num_layers,
        max_blocks_per_xfer=max_blocks_per_xfer,
    )
    adapter.add_peer(
        "source",
        peer_agent_metadata=b"fake-source-metadata",
        peer_layer_pool_base_ptrs=peer_bases,
        peer_layer_pool_size_bytes=[8_192] * num_layers,
    )
    return adapter, fake, local_bases, peer_bases


def test_read_blocks_preserves_pairing_and_chunks_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, fake, local_bases, peer_bases = _real_adapter(
        monkeypatch, num_layers=2, max_blocks_per_xfer=2
    )

    adapter.read_blocks(
        "source",
        peer_byte_offsets=[300, 900, 100],
        local_byte_offsets=[400, 0, 800],
        byte_lengths=[10, 20, 30],
    )

    assert fake.metadata_checks == 1
    assert len(fake.xfers) == 2
    assert fake.released == ["xfer-0", "xfer-1"]
    assert fake.poll_checks == 0  # transfer() returned DONE immediately.

    assert fake.desc_calls == [
        (
            [
                (local_bases[0] + 400, 10, 0),
                (local_bases[1] + 400, 10, 0),
                (local_bases[0], 20, 0),
                (local_bases[1], 20, 0),
            ],
            "VRAM",
        ),
        (
            [
                (peer_bases[0] + 300, 10, 0),
                (peer_bases[1] + 300, 10, 0),
                (peer_bases[0] + 900, 20, 0),
                (peer_bases[1] + 900, 20, 0),
            ],
            "DRAM",
        ),
        (
            [
                (local_bases[0] + 800, 30, 0),
                (local_bases[1] + 800, 30, 0),
            ],
            "VRAM",
        ),
        (
            [
                (peer_bases[0] + 100, 30, 0),
                (peer_bases[1] + 100, 30, 0),
            ],
            "DRAM",
        ),
    ]


@pytest.mark.parametrize(
    ("num_blocks", "expected_xfers"),
    [(0, 0), (1, 1), (64, 1), (65, 2)],
)
def test_batch_boundaries(
    monkeypatch: pytest.MonkeyPatch, num_blocks: int, expected_xfers: int
) -> None:
    adapter, fake, _local_bases, _peer_bases = _real_adapter(
        monkeypatch, max_blocks_per_xfer=64
    )
    adapter.read_blocks(
        "source",
        peer_byte_offsets=list(range(num_blocks)),
        local_byte_offsets=list(range(num_blocks)),
        byte_lengths=[1] * num_blocks,
    )
    assert len(fake.xfers) == expected_xfers
    assert len(fake.released) == expected_xfers
    assert fake.metadata_checks == (1 if num_blocks else 0)


@pytest.mark.parametrize(
    ("initial_state", "poll_states", "transfer_error", "error_match"),
    [
        ("DONE", [], None, None),
        ("PROC", ["DONE"], None, None),
        ("ERR", [], None, "state 'ERR'"),
        ("UNKNOWN", [], None, "state 'UNKNOWN'"),
        ("PROC", ["ERR"], None, "state 'ERR'"),
        ("PROC", [RuntimeError("poll failed")], None, "poll failed"),
        ("DONE", [], RuntimeError("post failed"), "post failed"),
    ],
)
def test_xfer_handle_released_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    initial_state: str,
    poll_states: list[object],
    transfer_error: BaseException | None,
    error_match: str | None,
) -> None:
    adapter, fake, _local_bases, _peer_bases = _real_adapter(monkeypatch)
    fake.initial_state = initial_state
    fake.poll_states.extend(poll_states)
    fake.transfer_error = transfer_error

    if error_match is None:
        adapter.read_block("source", 0, 0, 16)
    else:
        with pytest.raises(RuntimeError, match=error_match):
            adapter.read_block("source", 0, 0, 16)
    assert fake.released == ["xfer-0"]


def test_release_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, fake, _local_bases, _peer_bases = _real_adapter(monkeypatch)
    fake.release_error = RuntimeError("release failed")
    with pytest.raises(RuntimeError, match="release failed"):
        adapter.read_block("source", 0, 0, 16)
    assert fake.released == ["xfer-0"]


def test_second_chunk_failure_stops_later_chunks_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, fake, _local_bases, _peer_bases = _real_adapter(
        monkeypatch, max_blocks_per_xfer=2
    )
    fake.transfer_states.extend(["DONE", "ERR", "DONE"])

    with pytest.raises(RuntimeError, match="state 'ERR'"):
        adapter.read_blocks(
            "source",
            peer_byte_offsets=[0, 10, 20, 30, 40],
            local_byte_offsets=[0, 10, 20, 30, 40],
            byte_lengths=[10] * 5,
        )

    assert len(fake.xfers) == 2
    assert fake.released == ["xfer-0", "xfer-1"]
    assert fake.metadata_checks == 1


def test_mock_batch_copies_noncontiguous_blocks_across_layers() -> None:
    page_size = 32
    num_blocks = 5
    num_layers = 3
    source = [bytearray(num_blocks * page_size) for _ in range(num_layers)]
    target = [bytearray(num_blocks * page_size) for _ in range(num_layers)]
    for layer, pool in enumerate(source):
        for block in range(num_blocks):
            value = (layer + 1) * 16 + block
            pool[block * page_size : (block + 1) * page_size] = bytes(
                [value] * page_size
            )

    def ptr(pool: bytearray) -> int:
        return ctypes.addressof((ctypes.c_char * len(pool)).from_buffer(pool))

    adapter = RawNixlRemoteG2Adapter(
        "target",
        [ptr(pool) for pool in target],
        [len(pool) for pool in target],
        use_mock=True,
        max_blocks_per_xfer=2,
    )
    adapter.add_peer(
        "source",
        b"mock:source",
        [ptr(pool) for pool in source],
        [len(pool) for pool in source],
    )
    adapter.read_blocks(
        "source",
        peer_byte_offsets=[4 * page_size, 0, 2 * page_size],
        local_byte_offsets=[0, 3 * page_size, page_size],
        byte_lengths=[page_size] * 3,
    )

    for layer in range(num_layers):
        assert target[layer][0:page_size] == source[layer][4 * page_size :]
        assert (
            target[layer][page_size : 2 * page_size]
            == source[layer][2 * page_size : 3 * page_size]
        )
        assert (
            target[layer][3 * page_size : 4 * page_size] == source[layer][0:page_size]
        )


def test_fault_injection_counts_logical_blocks_and_stops_at_fault() -> None:
    page_size = 16
    source = bytearray(bytes(range(4 * page_size)))
    target = bytearray(4 * page_size)

    def ptr(pool: bytearray) -> int:
        return ctypes.addressof((ctypes.c_char * len(pool)).from_buffer(pool))

    adapter = RawNixlRemoteG2Adapter(
        "target",
        [ptr(target)],
        [len(target)],
        use_mock=True,
        fault_inject_every=3,
    )
    adapter.add_peer("source", b"mock:source", [ptr(source)], [len(source)])

    with pytest.raises(RuntimeError, match=r"read_block #3"):
        adapter.read_blocks(
            "source",
            peer_byte_offsets=[0, page_size, 2 * page_size, 3 * page_size],
            local_byte_offsets=[0, page_size, 2 * page_size, 3 * page_size],
            byte_lengths=[page_size] * 4,
        )

    # Logical reads #1 and #2 complete, #3 is injected, and #4 is not
    # attempted. A later one-block wrapper call therefore becomes #4.
    assert target[: 2 * page_size] == source[: 2 * page_size]
    assert target[2 * page_size :] == bytes(2 * page_size)
    adapter.read_block("source", 3 * page_size, 3 * page_size, page_size)
    assert target[3 * page_size :] == source[3 * page_size :]


@pytest.mark.parametrize(
    ("peer_offsets", "local_offsets", "lengths", "error_match"),
    [
        ([0], [], [1], "equal length"),
        ([-1], [0], [1], "negative byte offset"),
        ([0], [0], [0], "must be positive"),
        ([8_190], [0], [4], "exceeds peer"),
        ([0], [8_190], [4], "exceeds local"),
    ],
)
def test_batch_validation_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    peer_offsets: list[int],
    local_offsets: list[int],
    lengths: list[int],
    error_match: str,
) -> None:
    adapter, fake, _local_bases, _peer_bases = _real_adapter(monkeypatch)
    with pytest.raises(ValueError, match=error_match):
        adapter.read_blocks("source", peer_offsets, local_offsets, lengths)
    assert fake.xfers == []
    assert fake.released == []


def test_invalid_batch_size_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nixl_adapter, "NIXL_AVAILABLE", True)
    monkeypatch.setattr(nixl_adapter, "nixl_agent_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(nixl_adapter, "nixl_agent", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="max_blocks_per_xfer must be positive"):
        RawNixlRemoteG2Adapter("target", [1_000], [1_000], max_blocks_per_xfer=0)
