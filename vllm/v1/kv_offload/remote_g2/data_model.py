# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data model for Remote G2 KV-P2P.

Mirrors the TRT-LLM types in ``tensorrt_llm/_torch/pyexecutor/connectors/
remote_g2.py`` (RemoteKvReusePlan, SourceG2DescriptorRecord,
SourceG2DescriptorRegistry, RemoteG2Descriptor, RemoteG2ResolveResult)
with vLLM-specific simplifications:

* Single ``threading.RLock`` index instead of TRT-LLM's two-index
  (engine C++ lookup tree + Python registry) design. CPUOffloadingManager
  is pure-Python with O(1) lookup, so a single RLock suffices.
* Plan hashes are kept as Router-native ``int`` (XXH3-64) on the wire and
  converted at the boundary where the vLLM ``OffloadKey`` (bytes) index
  is queried.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

REMOTE_KV_REUSE_PLAN_VERSION = 1
REMOTE_G2_TIERS = frozenset({"host_pinned", "g2"})


@dataclass
class RemoteG2HandshakePayload:
    """Per-rank NIXL agent + pool layout pushed up from worker to scheduler.

    Each TP rank's worker process builds its own NIXL source agent over
    its own CPU pinned pool. The scheduler-side source RPC server caches
    one of these per ``tp_rank`` so it can answer ``get_metadata``
    requests on behalf of the right rank.

    Built in worker-side ``RemoteG2OffloadingSpec.get_handshake_metadata``
    after KV cache registration. Wrapped in the connector's
    ``KVConnectorHandshakeMetadata`` envelope and flown up via vLLM's
    ``collective_rpc("get_kv_connector_handshake_metadata")`` ->
    ``set_xfer_handshake_metadata({tp_rank: payload})`` mechanism (see
    ``vllm/v1/engine/core.py:170``).
    """

    tp_rank: int
    agent_name: str
    agent_metadata: bytes
    layer_pool_base_ptrs: list[int]
    layer_pool_size_bytes: list[int]
    page_size_bytes: int
    source_generation: int = 1


def _now_ms() -> int:
    # Wall-clock millis (not monotonic) so plan.expires_at_ms (from the
    # Router) and our lease TTLs share the same time base across processes.
    return int(time.time() * 1000)


def _is_remote_g2_tier(tier: str) -> bool:
    return tier in REMOTE_G2_TIERS


@dataclass(frozen=True)
class RemoteKvReusePlan:
    """Plan produced by Dynamo Router and delivered to the target worker
    via ``kv_transfer_params["remote_g2_plan"]``.

    Fields mirror the TRT-LLM ``RemoteKvReusePlan`` exactly so a single
    Router code path serves both engines. ``kv_block_hashes`` is empty
    when the producer hasn't been updated to populate the engine-side
    hash; in that case ``block_hashes`` is used for the source lookup.
    """

    plan_id: str
    request_id: str
    target_worker_id: int
    target_dp_rank: int
    source_worker_id: int
    source_dp_rank: int
    source_tier: str
    block_hashes: tuple[int, ...]
    start_block_index: int
    planned_prefix_blocks: int
    block_size_tokens: int
    created_at_ms: int
    expires_at_ms: int
    plan_version: int = REMOTE_KV_REUSE_PLAN_VERSION
    kv_block_hashes: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RemoteKvReusePlan:
        required = (
            "plan_id",
            "request_id",
            "target_worker_id",
            "target_dp_rank",
            "source_worker_id",
            "source_dp_rank",
            "source_tier",
            "start_block_index",
            "block_hashes",
            "planned_prefix_blocks",
            "block_size_tokens",
            "created_at_ms",
            "expires_at_ms",
        )
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"remote G2 plan missing fields: {missing}")

        block_hashes = tuple(int(h) for h in data["block_hashes"])
        kv_block_hashes = tuple(int(h) for h in data.get("kv_block_hashes", ()))
        if kv_block_hashes and len(kv_block_hashes) != len(block_hashes):
            raise ValueError(
                "kv_block_hashes length must match block_hashes when provided"
            )
        planned_prefix_blocks = int(data["planned_prefix_blocks"])
        if planned_prefix_blocks < 0:
            raise ValueError("planned_prefix_blocks must be non-negative")
        start_block_index = int(data["start_block_index"])
        if start_block_index < 0:
            raise ValueError("start_block_index must be non-negative")

        return cls(
            plan_id=str(data["plan_id"]),
            request_id=str(data["request_id"]),
            target_worker_id=int(data["target_worker_id"]),
            target_dp_rank=int(data["target_dp_rank"]),
            source_worker_id=int(data["source_worker_id"]),
            source_dp_rank=int(data["source_dp_rank"]),
            source_tier=str(data["source_tier"]),
            block_hashes=block_hashes,
            kv_block_hashes=kv_block_hashes,
            start_block_index=start_block_index,
            planned_prefix_blocks=min(planned_prefix_blocks, len(block_hashes)),
            block_size_tokens=int(data["block_size_tokens"]),
            created_at_ms=int(data["created_at_ms"]),
            expires_at_ms=int(data["expires_at_ms"]),
            plan_version=int(data.get("plan_version", REMOTE_KV_REUSE_PLAN_VERSION)),
        )

    def is_remote_g2(self) -> bool:
        return _is_remote_g2_tier(self.source_tier)

    def is_expired(self, now_ms: int | None = None) -> bool:
        return self.expires_at_ms <= (now_ms if now_ms is not None else _now_ms())

    @property
    def planned_hashes(self) -> tuple[int, ...]:
        return self.block_hashes[: self.planned_prefix_blocks]

    @property
    def planned_kv_block_hashes(self) -> tuple[int, ...]:
        return self.kv_block_hashes[: self.planned_prefix_blocks]


@dataclass(frozen=True)
class RemoteG2Descriptor:
    """Source-resolved descriptor returned to the target worker."""

    block_hash: int
    descriptor_generation: int
    pool_id: str
    byte_offset: int
    byte_length: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteG2BlockStatus:
    block_hash: int
    status: str
    descriptor_generation: int | None = None


@dataclass(frozen=True)
class RemoteG2ResolveResult:
    lease_id: str | None
    descriptors: tuple[RemoteG2Descriptor, ...]
    num_tokens: int
    reason: str = "ok"
    source_generation: int = 0
    per_block_status: tuple[RemoteG2BlockStatus, ...] = ()


@dataclass
class RemoteG2Lease:
    lease_id: str
    plan_id: str
    request_id: str
    target_worker_id: int
    target_dp_rank: int
    block_hashes: tuple[int, ...]
    descriptor_generations: tuple[int, ...]
    expires_at_ms: int
    pin_refs: tuple[Any, ...] = ()
    released: bool = False
    release_reason: str | None = None


@dataclass
class SourceG2DescriptorRecord:
    """Live host-pool block descriptor on the source side.

    ``byte_offset`` is the offset into the contiguous mmap host pool
    (see ``SharedOffloadRegion``). ``metadata['nixl_memory_desc']``
    carries the NIXL descriptor (ptr/size/device_id/memory_type).
    """

    block_hash: int
    source_worker_id: int
    source_dp_rank: int
    tier: str
    descriptor_generation: int
    pool_id: str
    byte_offset: int
    byte_length: int
    block_id: int = -1
    live: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    lease_count: int = 0

    def is_resolvable_for(self, source_worker_id: int, source_dp_rank: int) -> bool:
        return (
            self.live
            and self.source_worker_id == source_worker_id
            and self.source_dp_rank == source_dp_rank
            and _is_remote_g2_tier(self.tier)
        )


class SourceG2DescriptorRegistry:
    """Source-worker owned live host-pool descriptor + lease registry.

    Single-index design (a ``dict[block_hash_int, record]``) guarded by a
    ``threading.RLock``. Hit path: ``resolve_and_lease`` is O(N_plan) with
    only hashmap lookups under the lock.

    The registry also stashes the host-pool layout (base ptr, page size,
    rank, row stride, tier, pool_id, device_id) so the manager (which
    runs scheduler-side) and the RPC server (worker-side) — which in
    vLLM v1 sit on *different* Spec instances within the same EngineCore
    process — share the same view. Use ``get_or_create()`` to obtain the
    process-wide registry for a given ``(source_worker_id,
    source_dp_rank)``.
    """

    # Process-wide cache so the scheduler-side and worker-side Specs
    # share state. Keyed by (source_worker_id, source_dp_rank).
    _instances: dict[tuple[int, int], SourceG2DescriptorRegistry] = {}
    _instances_lock = threading.Lock()

    def __init__(
        self,
        *,
        source_worker_id: int,
        source_dp_rank: int,
        source_generation: int = 1,
        lease_ttl_ms: int = 30_000,
        clock_ms: Callable[[], int] = _now_ms,
        acquire_pin: Callable[[SourceG2DescriptorRecord, str], Any] | None = None,
        release_pin: Callable[[Any], None] | None = None,
    ) -> None:
        self.source_worker_id = int(source_worker_id)
        self.source_dp_rank = int(source_dp_rank)
        self.source_generation = int(source_generation)
        self.lease_ttl_ms = int(lease_ttl_ms)
        self._clock_ms = clock_ms
        self._acquire_pin = acquire_pin
        self._release_pin = release_pin
        self._records: dict[int, SourceG2DescriptorRecord] = {}
        self._leases: dict[str, RemoteG2Lease] = {}
        self._lock = threading.RLock()
        # Monotonic counter that guarantees lease_id uniqueness even
        # when two threads land in resolve_and_lease at the same
        # clock millisecond with the same (plan_id, request_id).
        self._lease_id_counter: int = 0

        # Pool layout — set by the worker-side spec after KV caches are
        # registered. Until then, descriptor publish path is gated.
        # vLLM v1 allocates ONE CPU tensor per transformer layer (e.g. 36
        # for Qwen3-8B). All layers share the same block-id space (a
        # given block_id maps to the same row in every per-layer
        # tensor), but each layer has its own contiguous pool with its
        # own base pointer. ``_layer_pool_base_ptrs[i]`` is the start of
        # layer i's pool. Page size and row stride are uniform across
        # layers in homogeneous models (asserted in set_pool_layout).
        self._layer_pool_base_ptrs: list[int] = []
        self._layer_pool_size_bytes: list[int] = []
        self._page_size_bytes: int = 0
        self._row_stride_bytes: int = 0
        self._rank: int = 0
        self._tier: str = "host_pinned"
        self._pool_id: str = "g2-host-pinned"
        self._device_id: int = 0
        self._descriptor_generation_counter: dict[int, int] = {}

        # Target-side counters (incremented by manager when this engine
        # acts as a *target* and the plan-driven path fires). Exposed via
        # SourceG2RpcServer.stats so tests can verify cross-engine flow.
        self.plan_seen_count: int = 0
        self.plan_resolved_count: int = 0
        self.plan_load_specs_emitted: int = 0
        self.plan_blocks_loaded: int = 0
        # Post-completion counters (incremented in ``complete_load`` ONLY
        # for requests that actually emitted a RemoteG2LoadSpec, i.e. a
        # real NIXL READ). Because a failed READ trips the
        # ``assert transfer_result.success`` in the offloading worker and
        # crashes the engine step *before* ``complete_load`` runs, reaching
        # these increments proves the emitted transfer completed — unlike
        # the plan-time ``plan_blocks_loaded`` above, which only proves a
        # spec was prepared. ``plan_bytes_completed`` sums the descriptors'
        # own ``byte_length`` (the exact amount the transfer handler moves),
        # so it is model-agnostic and not a fixed block-size estimate.
        self.plan_loads_completed: int = 0
        self.plan_bytes_completed: int = 0

        # Transport provenance for the target-side READ path, published by
        # the worker-side spec once the NIXL adapter is built. Defaults are
        # fail-closed (mock=True, backend "unset") so a perf gate that reads
        # these over the stats RPC rejects a run until a real NIXL/UCX
        # adapter has reported in. (Layer count is exposed separately via
        # the existing ``num_layers`` property, from the pool layout, so the
        # gate can convert single-layer logical bytes into approx wire bytes.)
        self.transport_backend: str = "unset"
        self.transport_mock: bool = True

        # Source-side eviction protection. The CPUOffloadingManager's
        # policy enforces eviction by ref_cnt and a per-call ``protected``
        # set; a lease that has resolved a descriptor for hash X must
        # bump policy.get(key_for_X).ref_cnt so a parallel store on the
        # scheduler thread doesn't evict X out from under the in-flight
        # NIXL READ. ``set_policy`` is first-wins: the SCHEDULER-side
        # manager (where stores actually run) registers first; the
        # worker-side manager's call is a no-op.
        self._policy: Any = None
        self._hash_to_key: dict[int, Any] = {}
        # Lease pin counters and stats — independent from
        # SourceG2DescriptorRecord.lease_count (which is a per-record
        # display counter); these drive the actual policy.ref_cnt
        # bumps and let tests assert pin/unpin balance.
        self.pin_count_total: int = 0
        self.unpin_count_total: int = 0
        self.pin_failures: int = 0

    @classmethod
    def get_or_create(
        cls,
        source_worker_id: int,
        source_dp_rank: int,
        **kwargs: Any,
    ) -> SourceG2DescriptorRegistry:
        """Return the process-wide registry for ``(source_worker_id,
        source_dp_rank)``, creating it if necessary.

        The first call constructs the registry with the passed kwargs;
        subsequent calls return the same instance regardless of kwargs
        (so the second-Spec-instance code path doesn't reset state).
        """
        key = (int(source_worker_id), int(source_dp_rank))
        with cls._instances_lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls(
                    source_worker_id=source_worker_id,
                    source_dp_rank=source_dp_rank,
                    **kwargs,
                )
                cls._instances[key] = inst
        return inst

    @classmethod
    def _clear_singletons_for_tests(cls) -> None:
        """Drop all cached registries. Used by unit tests; not safe to
        call while production specs are alive."""
        with cls._instances_lock:
            cls._instances.clear()

    # --- pool layout (set by worker-side spec) ---

    def set_pool_layout(
        self,
        *,
        layer_pool_base_ptrs: list[int],
        layer_pool_size_bytes: list[int],
        page_size_bytes: int,
        rank: int = 0,
        num_workers: int = 1,
        row_stride_bytes: int | None = None,
        tier: str = "host_pinned",
        pool_id: str = "g2-host-pinned",
        device_id: int = 0,
    ) -> None:
        if page_size_bytes <= 0:
            raise ValueError("page_size_bytes must be positive")
        if not layer_pool_base_ptrs:
            raise ValueError("layer_pool_base_ptrs must be non-empty")
        if len(layer_pool_base_ptrs) != len(layer_pool_size_bytes):
            raise ValueError(
                "layer_pool_base_ptrs and layer_pool_size_bytes "
                "must have the same length"
            )
        for i, ptr in enumerate(layer_pool_base_ptrs):
            if ptr <= 0:
                raise ValueError(f"layer {i} pool_base_ptr is non-positive")
        with self._lock:
            self._layer_pool_base_ptrs = [int(p) for p in layer_pool_base_ptrs]
            self._layer_pool_size_bytes = [int(s) for s in layer_pool_size_bytes]
            self._page_size_bytes = int(page_size_bytes)
            self._rank = int(rank)
            self._row_stride_bytes = int(
                row_stride_bytes
                if row_stride_bytes is not None
                else page_size_bytes * max(num_workers, 1)
            )
            self._tier = str(tier)
            self._pool_id = str(pool_id)
            self._device_id = int(device_id)

    def pool_layout_ready(self) -> bool:
        return bool(self._layer_pool_base_ptrs) and self._page_size_bytes > 0

    @property
    def num_layers(self) -> int:
        return len(self._layer_pool_base_ptrs)

    def set_transport_info(self, *, backend: str, mock: bool) -> None:
        """Record the target-side READ transport (called by the spec once
        the NIXL adapter is built). Read back over the stats RPC by the
        fail-closed perf gate to reject a mock-memcpy fallback."""
        self.transport_backend = str(backend)
        self.transport_mock = bool(mock)

    def record_completed_load(self, bytes_emitted: int) -> bool:
        """Count one completed plan-driven load. Returns whether it counted.

        Called from ``RemoteG2OffloadingManager.complete_load`` with the
        request's ``bytes_emitted`` (the descriptor byte_length sum for the
        RemoteG2LoadSpec it actually emitted). A non-positive value means no
        spec was emitted for this request (it fell back to the local path),
        so nothing is counted -- this is what keeps a leased-but-fully-local
        request from being scored as a transfer. The caller zeroes its
        per-request tally after a True return so a re-fired ``complete_load``
        passes 0 here and cannot double-count.
        """
        if bytes_emitted <= 0:
            return False
        self.plan_loads_completed += 1
        self.plan_bytes_completed += int(bytes_emitted)
        return True

    @property
    def layer_pool_base_ptrs(self) -> list[int]:
        return list(self._layer_pool_base_ptrs)

    @property
    def layer_pool_size_bytes(self) -> list[int]:
        return list(self._layer_pool_size_bytes)

    @property
    def page_size_bytes(self) -> int:
        return self._page_size_bytes

    def upsert_for_block(
        self, block_hash_int: int, block_id: int
    ) -> SourceG2DescriptorRecord | None:
        """Build and store a descriptor for ``(block_hash, block_id)``.

        Returns the record (so callers can log) or ``None`` if the pool
        layout has not been set yet (descriptor publish is gated).

        Multi-layer pools share the same ``block_id`` namespace and the
        same byte offset within each per-layer pool (row stride is
        uniform). The descriptor stores ``block_id`` and the per-layer
        ``byte_offset``; the transfer handler combines those with the
        per-layer base pointers in the bundle metadata to issue one
        NIXL READ per layer.
        """
        with self._lock:
            if not self.pool_layout_ready():
                return None
            byte_offset = (
                block_id * self._row_stride_bytes + self._rank * self._page_size_bytes
            )
            gen = self._descriptor_generation_counter.get(block_hash_int, 0) + 1
            self._descriptor_generation_counter[block_hash_int] = gen
            record = SourceG2DescriptorRecord(
                block_hash=block_hash_int,
                source_worker_id=self.source_worker_id,
                source_dp_rank=self.source_dp_rank,
                tier=self._tier,
                descriptor_generation=gen,
                pool_id=self._pool_id,
                byte_offset=byte_offset,
                byte_length=self._page_size_bytes,
                block_id=block_id,
                metadata={
                    "nixl_memory_desc": {
                        # Layer 0 ptr kept for inspection / debugging;
                        # the transfer handler uses the bundle's full
                        # layer_pool_base_ptrs list.
                        "ptr": self._layer_pool_base_ptrs[0] + byte_offset,
                        "size": self._page_size_bytes,
                        "device_id": self._device_id,
                        "memory_type": "DRAM",
                        "name": self._pool_id,
                        "num_layers": len(self._layer_pool_base_ptrs),
                    }
                },
            )
            self._records[block_hash_int] = record
        return record

    # --- policy plumbing (scheduler-manager side) ---

    def set_policy(self, policy: Any) -> bool:
        """Register the CPUOffloadingManager's policy with the registry.

        Last-wins. In vLLM v1's EngineCore startup, the WORKER-role
        connector is constructed FIRST (inside ``_initialize_kv_caches``
        → ``gpu_worker.ensure_kv_transfer_initialized``), and the
        SCHEDULER-role connector is constructed afterwards. Only the
        scheduler-side manager runs ``complete_store`` and populates
        its policy; the worker-side manager's policy stays empty
        forever. With last-wins, the scheduler's (populated) policy is
        the one the source RPC sees, so plan resolves can pin live
        blocks. Returns True (always installs).
        """
        with self._lock:
            previous = self._policy
            self._policy = policy
        logger.info(
            "RemoteG2 registry.set_policy: installed %r (replaced=%s)",
            type(policy).__name__,
            previous is not None,
        )
        return True

    def record_key(self, block_hash_int: int, key: Any) -> None:
        """Remember the OffloadKey associated with a published block_hash.

        Called by the scheduler-side manager in ``complete_store`` so
        the registry can later resolve hash → key → policy block when
        a lease wants to pin the block against eviction.
        """
        with self._lock:
            self._hash_to_key[int(block_hash_int)] = key

    def forget_key(self, block_hash_int: int) -> None:
        with self._lock:
            self._hash_to_key.pop(int(block_hash_int), None)

    def _pin_block_locked(self, block_hash_int: int) -> bool:
        """Bump policy.get(key).ref_cnt for the given hash, atomic
        under ``self._lock`` (caller must hold it).

        Returns True on success, False if the block isn't in the
        policy (e.g. evicted between resolve lookup and pin).
        """
        if self._policy is None:
            # No policy registered yet — POC's pre-store smoke tests
            # exercise this path (registry stands alone without a
            # manager). Pin is a no-op; lease still tracks the hash
            # for unpin balance.
            self.pin_count_total += 1
            return True
        key = self._hash_to_key.get(int(block_hash_int))
        if key is None:
            self.pin_failures += 1
            logger.warning(
                "RemoteG2: pin_failed for hash=%d cause=no_key_registered "
                "(records=%d, keys=%d)",
                int(block_hash_int),
                len(self._records),
                len(self._hash_to_key),
            )
            return False
        block = self._policy.get(key)
        if block is None:
            self.pin_failures += 1
            # Probe the policy's internals to see if the key is there
            # at all under a different form (sanity check for key
            # encoding bugs).
            policy_size = 0
            policy_sample = []
            try:
                # CachePolicy doesn't expose an iterator, but its
                # underlying dict/OrderedDict is `_blocks` for LRU.
                inner = getattr(self._policy, "blocks", None)
                if inner is not None:
                    policy_size = len(inner)
                    policy_sample = list(inner.keys())[:3]
            except Exception:
                pass
            logger.warning(
                "RemoteG2: pin_failed for hash=%d cause=policy_missing_block "
                "(key_len=%d, key_prefix=%r, policy_size=%d, sample_key=%r)",
                int(block_hash_int),
                len(key) if isinstance(key, (bytes, bytearray)) else -1,
                key[:16] if isinstance(key, (bytes, bytearray)) else key,
                policy_size,
                policy_sample[0] if policy_sample else None,
            )
            return False
        if not block.is_ready:
            self.pin_failures += 1
            logger.warning(
                "RemoteG2: pin_failed for hash=%d cause=block_not_ready "
                "(ref_cnt=%d, block_id=%d)",
                int(block_hash_int),
                block.ref_cnt,
                block.block_id,
            )
            return False
        block.ref_cnt += 1
        self.pin_count_total += 1
        return True

    def _unpin_block_locked(self, block_hash_int: int) -> None:
        """Decrement the policy block's ref_cnt for the given hash.
        No-op if the policy isn't registered, the hash isn't known,
        or the block has been removed."""
        self.unpin_count_total += 1
        if self._policy is None:
            return
        key = self._hash_to_key.get(int(block_hash_int))
        if key is None:
            return
        block = self._policy.get(key)
        if block is not None and block.ref_cnt > 0:
            block.ref_cnt -= 1

    def upsert_descriptor(self, record: SourceG2DescriptorRecord) -> None:
        if record.source_worker_id != self.source_worker_id:
            raise ValueError("record source_worker_id does not match registry")
        if record.source_dp_rank != self.source_dp_rank:
            raise ValueError("record source_dp_rank does not match registry")
        if not _is_remote_g2_tier(record.tier):
            raise ValueError("registry accepts remote G2 records only")
        with self._lock:
            self._records[int(record.block_hash)] = record

    def remove_descriptor(self, block_hash: int) -> None:
        with self._lock:
            record = self._records.pop(int(block_hash), None)
            if record is not None:
                record.live = False

    def get_descriptor(self, block_hash: int) -> SourceG2DescriptorRecord | None:
        with self._lock:
            return self._records.get(int(block_hash))

    def _new_lease_id(self, plan_id: str, request_id: str) -> str:
        # Combine clock + monotonic counter so concurrent resolves on
        # the same (plan_id, request_id) within the same clock tick
        # don't collide on the lease_id key. Counter is protected by
        # ``self._lock``, which the caller is expected to hold.
        self._lease_id_counter += 1
        return f"{plan_id}:{request_id}:{self._clock_ms()}:{self._lease_id_counter}"

    def resolve_and_lease(
        self, plan: Mapping[str, Any] | RemoteKvReusePlan
    ) -> RemoteG2ResolveResult:
        """Resolve a plan against the registry and grant a lease.

        Returns descriptors for the maximal prefix of the plan that is
        currently live, plus a lease id holding ``acquire_pin`` references
        so concurrent eviction is suppressed for the lease window.
        """
        now_ms = self._clock_ms()
        try:
            parsed = (
                plan
                if isinstance(plan, RemoteKvReusePlan)
                else RemoteKvReusePlan.from_dict(plan)
            )
        except (TypeError, ValueError):
            return RemoteG2ResolveResult(
                None, (), 0, "invalid_plan", self.source_generation
            )
        if parsed.plan_version != REMOTE_KV_REUSE_PLAN_VERSION:
            return RemoteG2ResolveResult(
                None, (), 0, "unsupported_plan_version", self.source_generation
            )
        if parsed.source_worker_id != self.source_worker_id:
            return RemoteG2ResolveResult(
                None, (), 0, "wrong_source_worker", self.source_generation
            )
        if parsed.source_dp_rank != self.source_dp_rank:
            return RemoteG2ResolveResult(
                None, (), 0, "wrong_source_rank", self.source_generation
            )
        if not parsed.is_remote_g2():
            return RemoteG2ResolveResult(
                None, (), 0, "wrong_source_tier", self.source_generation
            )
        if parsed.is_expired(now_ms):
            return RemoteG2ResolveResult(
                None, (), 0, "plan_expired", self.source_generation
            )

        identity_hashes = parsed.planned_hashes
        kv_hashes = parsed.planned_kv_block_hashes or identity_hashes

        descriptors: list[RemoteG2Descriptor] = []
        per_block_status: list[RemoteG2BlockStatus] = []
        pinned_hashes: list[int] = []
        pin_refs: list[Any] = []

        with self._lock:
            # Generate the lease_id inside the lock so the counter
            # advances atomically with the lease table mutation.
            lease_id = self._new_lease_id(parsed.plan_id, parsed.request_id)
            for i, identity_hash in enumerate(identity_hashes):
                kv_hash = int(kv_hashes[i])
                record = self._records.get(kv_hash)
                if record is None or not record.live:
                    reason = "missing" if record is None else "non_live"
                    per_block_status.append(
                        RemoteG2BlockStatus(int(identity_hash), reason)
                    )
                    break
                if not record.is_resolvable_for(
                    self.source_worker_id, self.source_dp_rank
                ):
                    per_block_status.append(
                        RemoteG2BlockStatus(int(identity_hash), "wrong_owner")
                    )
                    break
                # Atomically pin the policy block so a concurrent
                # CPU-pool eviction can't evict it out from under the
                # in-flight transfer. On failure (e.g. block evicted
                # between record lookup and now) record the status
                # and stop the prefix.
                if not self._pin_block_locked(kv_hash):
                    per_block_status.append(
                        RemoteG2BlockStatus(int(identity_hash), "pin_failed")
                    )
                    break
                # Optional side-channel pin (TRT-LLM POC's
                # find_and_pin_secondary_block_by_hash). Default None
                # in the vLLM POC.
                pin_ref: Any = None
                if self._acquire_pin is not None:
                    try:
                        pin_ref = self._acquire_pin(record, lease_id)
                    except Exception:
                        # Roll back the policy pin we just took.
                        self._unpin_block_locked(kv_hash)
                        per_block_status.append(
                            RemoteG2BlockStatus(int(identity_hash), "pin_failed")
                        )
                        break
                record.lease_count += 1
                pinned_hashes.append(kv_hash)
                pin_refs.append(pin_ref)
                # The descriptor's block_hash must be the hash the TARGET
                # uses to look up its own keys (i.e., the projected vLLM
                # hash equal to `block_hash_to_router_int(target_key)`),
                # not the Router's LocalBlockHash identity. The kv_hash
                # here came from `plan.kv_block_hashes` (chain walk on
                # the host-pinned indexer) which is exactly that
                # projection; for the shim path, kv_hash falls back to
                # identity_hash, so this still works for both sources.
                descriptors.append(
                    RemoteG2Descriptor(
                        block_hash=int(kv_hash),
                        descriptor_generation=record.descriptor_generation,
                        pool_id=record.pool_id,
                        byte_offset=record.byte_offset,
                        byte_length=record.byte_length,
                        metadata=dict(record.metadata),
                    )
                )
                per_block_status.append(
                    RemoteG2BlockStatus(
                        int(identity_hash),
                        "resolved",
                        descriptor_generation=record.descriptor_generation,
                    )
                )

            # If no descriptors at all, release any partial pins we
            # took so they don't leak. (E.g. the first block resolved
            # but the second failed; the for-loop break left the
            # first one pinned even though we won't return a lease.)
            if not descriptors and pinned_hashes:
                for h in pinned_hashes:
                    self._unpin_block_locked(h)
                # release_pin callback for any external pins we held.
                if self._release_pin is not None:
                    for ref in pin_refs:
                        with contextlib.suppress(Exception):
                            self._release_pin(ref)
                pinned_hashes = []
                pin_refs = []

            if not descriptors:
                return RemoteG2ResolveResult(
                    None,
                    (),
                    0,
                    per_block_status[0].status if per_block_status else "empty",
                    self.source_generation,
                    tuple(per_block_status),
                )

            lease = RemoteG2Lease(
                lease_id=lease_id,
                plan_id=parsed.plan_id,
                request_id=parsed.request_id,
                target_worker_id=parsed.target_worker_id,
                target_dp_rank=parsed.target_dp_rank,
                block_hashes=tuple(d.block_hash for d in descriptors),
                descriptor_generations=tuple(
                    d.descriptor_generation for d in descriptors
                ),
                expires_at_ms=now_ms + self.lease_ttl_ms,
                pin_refs=tuple(pin_refs),
            )
            self._leases[lease_id] = lease

        return RemoteG2ResolveResult(
            lease_id=lease_id,
            descriptors=tuple(descriptors),
            num_tokens=len(descriptors) * parsed.block_size_tokens,
            reason="ok",
            source_generation=self.source_generation,
            per_block_status=tuple(per_block_status),
        )

    def release_lease(self, lease_id: str, reason: str = "released") -> bool:
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is None or lease.released:
                return False
            lease.released = True
            lease.release_reason = reason
            for block_hash, pin_ref in zip(lease.block_hashes, lease.pin_refs):
                record = self._records.get(int(block_hash))
                if record is not None and record.lease_count > 0:
                    record.lease_count -= 1
                # Decrement the policy block's ref_cnt — symmetric
                # with the pin we took in resolve_and_lease.
                self._unpin_block_locked(int(block_hash))
                if self._release_pin is not None:
                    with contextlib.suppress(Exception):
                        self._release_pin(pin_ref)
            return True

    def expire_stale_leases(self, now_ms: int | None = None) -> int:
        ts = now_ms if now_ms is not None else self._clock_ms()
        expired: list[str] = []
        with self._lock:
            for lease_id, lease in self._leases.items():
                if not lease.released and lease.expires_at_ms <= ts:
                    expired.append(lease_id)
        for lease_id in expired:
            self.release_lease(lease_id, reason="ttl_expired")
        return len(expired)
