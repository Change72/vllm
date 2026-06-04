# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test for the Remote G2 (KV-P2P) data plane.

Exercises every layer **without** vLLM engine, GPU, or a real NIXL
install:

1. Source side:
   - Allocates a "host pool" as a single ``bytearray`` and fills the
     first N block slots with deterministic per-block payloads.
   - Builds a ``SourceG2DescriptorRegistry`` and upserts a
     ``SourceG2DescriptorRecord`` per filled block (byte offset =
     block_id * page_size).
   - Starts a ``SourceG2RpcServer`` on a temp Unix socket.
   - Registers a mock NIXL source bundle pointing at the pool.

2. Target side:
   - Allocates an empty "GPU pool" ``bytearray``.
   - Builds a ``RawNixlRemoteG2Adapter`` in mock mode pointed at the
     local pool.
   - Builds a plan referring to a subset of the source's block hashes,
     then uses ``TargetG2RpcClient`` to ``resolve_and_lease`` and
     ``get_metadata`` from the source.
   - Calls ``adapter.add_peer`` and ``read_block`` for each resolved
     descriptor.
   - Verifies the target pool now byte-matches the source pool for the
     requested blocks (i.e. the P2P pull worked).
   - Releases the lease and confirms the source dropped its lease entry.

The transport is the mock memcpy path inside ``RawNixlRemoteG2Adapter``
(both pools live in this process). Replace ``use_mock=True`` with a
real ``nixl`` install and the same flow drives a real NIXL READ.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
import time
from collections.abc import Iterator

import pytest

from vllm.v1.kv_offload.remote_g2.data_model import (
    REMOTE_KV_REUSE_PLAN_VERSION,
    RemoteKvReusePlan,
    SourceG2DescriptorRecord,
    SourceG2DescriptorRegistry,
)
from vllm.v1.kv_offload.remote_g2.nixl_adapter import (
    NixlSourceBundle,
    RawNixlRemoteG2Adapter,
    _build_mock_source_bundle,
)
from vllm.v1.kv_offload.remote_g2.source_rpc import SourceG2RpcServer
from vllm.v1.kv_offload.remote_g2.target_client import TargetG2RpcClient

NUM_BLOCKS = 8
PAGE_SIZE = 4096
PLAN_PREFIX_BLOCKS = 4


def _block_payload(block_idx: int, page_size: int) -> bytes:
    """Deterministic per-block fill so we can verify byte-for-byte."""
    base = bytes([(block_idx * 17 + i) & 0xFF for i in range(min(64, page_size))])
    return (base * ((page_size // len(base)) + 1))[:page_size]


def _ptr_of(buf: bytearray) -> int:
    return ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))


@pytest.fixture
def tmp_socket() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "src.sock")


def _source_setup(
    socket_path: str,
    *,
    populated_blocks: int,
) -> tuple[bytearray, SourceG2DescriptorRegistry, SourceG2RpcServer, NixlSourceBundle]:
    pool = bytearray(NUM_BLOCKS * PAGE_SIZE)
    for b in range(populated_blocks):
        pool[b * PAGE_SIZE : (b + 1) * PAGE_SIZE] = _block_payload(b, PAGE_SIZE)
    pool_base = _ptr_of(pool)

    registry = SourceG2DescriptorRegistry(
        source_worker_id=1, source_dp_rank=0, lease_ttl_ms=10_000
    )
    for b in range(populated_blocks):
        block_hash = 1_000 + b  # router-style int hash
        registry.upsert_descriptor(
            SourceG2DescriptorRecord(
                block_hash=block_hash,
                source_worker_id=1,
                source_dp_rank=0,
                tier="host_pinned",
                descriptor_generation=1,
                pool_id="src-pool",
                byte_offset=b * PAGE_SIZE,
                byte_length=PAGE_SIZE,
                block_id=b,
                metadata={
                    "nixl_memory_desc": {
                        "ptr": pool_base + b * PAGE_SIZE,
                        "size": PAGE_SIZE,
                        "device_id": 0,
                        "memory_type": "DRAM",
                        "name": "src-pool",
                    }
                },
            )
        )

    bundle = _build_mock_source_bundle(
        "src-agent", pool_base, NUM_BLOCKS * PAGE_SIZE, source_generation=1
    )

    server = SourceG2RpcServer(registry, socket_path=socket_path, recv_timeout_ms=100)
    server.set_nixl_bundle_provider(lambda b=bundle: b)
    server.start()
    return pool, registry, server, bundle


def _make_plan(block_hashes: list[int]) -> RemoteKvReusePlan:
    return RemoteKvReusePlan(
        plan_id="plan-1",
        request_id="req-1",
        target_worker_id=2,
        target_dp_rank=0,
        source_worker_id=1,
        source_dp_rank=0,
        source_tier="host_pinned",
        block_hashes=tuple(block_hashes),
        start_block_index=0,
        planned_prefix_blocks=len(block_hashes),
        block_size_tokens=16,
        created_at_ms=0,
        expires_at_ms=10**15,
        plan_version=REMOTE_KV_REUSE_PLAN_VERSION,
    )


def test_remote_g2_end_to_end_with_mock_nixl(tmp_socket: str) -> None:
    """Source registers blocks → target resolves + reads → bytes match."""
    pool, registry, server, bundle = _source_setup(
        tmp_socket, populated_blocks=NUM_BLOCKS
    )
    try:
        # --- target side ---
        target_pool = bytearray(NUM_BLOCKS * PAGE_SIZE)
        target_base = _ptr_of(target_pool)
        adapter = RawNixlRemoteG2Adapter(
            "tgt-agent",
            target_base,
            len(target_pool),
            use_mock=True,
        )
        client = TargetG2RpcClient(tmp_socket, timeout_ms=2_000)

        plan = _make_plan([1_000 + b for b in range(PLAN_PREFIX_BLOCKS)])

        # Step 1: pull the source's NIXL bundle and hand it our agent
        # metadata in the same RPC (matches the TRT-LLM POC handshake).
        meta = client.get_metadata(peer_agent_metadata=adapter.agent_metadata)
        assert meta is not None, "source should publish a bundle"
        assert meta["source_worker_id"] == 1
        assert meta["pool_size_bytes"] == NUM_BLOCKS * PAGE_SIZE

        adapter.add_peer(
            peer_name="src-agent",
            peer_agent_metadata=meta["agent_metadata"],
            peer_pool_base_ptr=meta["pool_base_ptr"],
            peer_pool_size_bytes=meta["pool_size_bytes"],
        )

        # Step 2: resolve the plan; expect a full prefix hit.
        result = client.resolve_and_lease(plan)
        assert result.reason == "ok"
        assert result.lease_id is not None
        assert len(result.descriptors) == PLAN_PREFIX_BLOCKS

        # Step 3: pull each block; place at target block index = source idx
        # for simplicity. Verify bytes match.
        for desc, target_block_idx in zip(
            result.descriptors, range(PLAN_PREFIX_BLOCKS)
        ):
            adapter.read_block(
                peer_name="src-agent",
                peer_byte_offset=desc.byte_offset,
                local_byte_offset=target_block_idx * PAGE_SIZE,
                byte_length=desc.byte_length,
            )

        for b in range(PLAN_PREFIX_BLOCKS):
            expected = _block_payload(b, PAGE_SIZE)
            actual = bytes(target_pool[b * PAGE_SIZE : (b + 1) * PAGE_SIZE])
            assert actual == expected, f"block {b} content mismatch"

        # Step 4: release the lease and confirm the source dropped it.
        assert client.release_lease(result.lease_id) is True
        # Best-effort: re-releasing returns False.
        assert client.release_lease(result.lease_id) is False

        client.close()
    finally:
        server.stop()
        # Touch ``pool`` and ``registry`` and ``bundle`` so they stay alive
        # until the server has stopped (avoids ctypes pointer-into-freed-mem).
        del registry, bundle, pool


def test_remote_g2_partial_prefix_when_missing_block(tmp_socket: str) -> None:
    """If a plan block isn't in the registry, prefix stops at first miss."""
    pool, registry, server, bundle = _source_setup(
        tmp_socket, populated_blocks=2  # only blocks 1000, 1001 are live
    )
    try:
        plan = _make_plan([1_000, 1_001, 1_002, 1_003])

        client = TargetG2RpcClient(tmp_socket, timeout_ms=2_000)
        result = client.resolve_and_lease(plan)
        # Reason for the FIRST missed block; descriptors cover the
        # contiguous prefix.
        assert result.lease_id is not None
        assert len(result.descriptors) == 2
        # The per_block_status array length isn't included in the wire
        # roundtrip (target_client drops it), so we just check the
        # descriptor prefix is correct.
        assert tuple(d.block_hash for d in result.descriptors) == (1_000, 1_001)

        client.release_lease(result.lease_id)
        client.close()
    finally:
        server.stop()
        del registry, bundle, pool


def test_remote_g2_metadata_not_ready_until_bundle_set(tmp_socket: str) -> None:
    """get_metadata returns the canonical 'not_ready' error when the
    NIXL bundle provider hasn't been installed yet (M2 state)."""
    registry = SourceG2DescriptorRegistry(source_worker_id=1, source_dp_rank=0)
    server = SourceG2RpcServer(registry, socket_path=tmp_socket, recv_timeout_ms=100)
    server.start()
    try:
        client = TargetG2RpcClient(tmp_socket, timeout_ms=2_000)
        bundle = client.get_metadata()
        assert bundle is None  # 'nixl_source_bundle_not_ready'
        client.close()
    finally:
        server.stop()


def test_remote_g2_lease_ttl_expiry() -> None:
    """Direct registry-level test: leases past TTL are reaped."""
    clock = [0]

    def fake_clock() -> int:
        return clock[0]

    registry = SourceG2DescriptorRegistry(
        source_worker_id=1,
        source_dp_rank=0,
        lease_ttl_ms=100,
        clock_ms=fake_clock,
    )
    registry.upsert_descriptor(
        SourceG2DescriptorRecord(
            block_hash=42,
            source_worker_id=1,
            source_dp_rank=0,
            tier="host_pinned",
            descriptor_generation=1,
            pool_id="pool",
            byte_offset=0,
            byte_length=PAGE_SIZE,
            block_id=0,
        )
    )
    plan = _make_plan([42])
    res = registry.resolve_and_lease(plan)
    assert res.lease_id is not None

    # Advance clock past TTL.
    clock[0] = 1_000
    expired = registry.expire_stale_leases()
    assert expired == 1
    # Re-releasing should now be a no-op (already released by expiry).
    assert registry.release_lease(res.lease_id) is False


if __name__ == "__main__":  # pragma: no cover
    # Allow `python tests/v1/kv_offload/remote_g2/test_e2e_smoke.py` as a
    # quick check. Prints PASS/FAIL for each test; useful before vLLM is
    # installed in CI.
    import traceback

    cases: list[tuple[str, callable]] = []
    cases.append(
        ("ttl_expiry", test_remote_g2_lease_ttl_expiry),
    )

    def _wrap(fn):
        def runner():
            with tempfile.TemporaryDirectory() as tmp:
                fn(os.path.join(tmp, "src.sock"))

        return runner

    cases.append(
        ("end_to_end_mock_nixl", _wrap(test_remote_g2_end_to_end_with_mock_nixl))
    )
    cases.append(
        (
            "partial_prefix_missing_block",
            _wrap(test_remote_g2_partial_prefix_when_missing_block),
        )
    )
    cases.append(
        (
            "metadata_not_ready",
            _wrap(test_remote_g2_metadata_not_ready_until_bundle_set),
        )
    )

    failures = 0
    for name, fn in cases:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({(time.perf_counter() - t0) * 1000:.1f} ms)")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    raise SystemExit(0 if failures == 0 else 1)
