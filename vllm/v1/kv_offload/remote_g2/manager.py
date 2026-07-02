# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remote G2 OffloadingManager.

Extends ``CPUOffloadingManager`` with:

* Concurrent-safe access: an ``RLock`` covers shared state so the
  worker-side ZMQ REP thread can read the registry while the scheduler
  is mutating it.
* Descriptor publishing: on ``complete_store`` the manager looks up
  each newly-stored key's policy block_id, builds a
  ``SourceG2DescriptorRecord`` with the correct mmap-pool byte offset,
  and upserts into the ``SourceG2DescriptorRegistry`` so peers can
  resolve it. Evicted keys are removed.
* Plan-aware lookup: when ``ReqContext.kv_transfer_params`` carries
  ``remote_g2_plan``, ``lookup`` consults a per-request resolve cache
  populated from the target RPC client; matching keys count as hits
  even if the local pool doesn't have them. ``prepare_load`` returns a
  ``RemoteG2LoadSpec`` for those keys instead of a ``CPULoadStoreSpec``.

POC scope: assumes single-tensor canonical KV (one entry in
``CanonicalKVCaches.tensors``). Multi-tensor (multi-group) support
mirrors the same pattern with a per-tensor descriptor; deferred to M4.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.remote_g2.data_model import (
    RemoteG2Descriptor,
    RemoteG2ResolveResult,
    RemoteKvReusePlan,
    SourceG2DescriptorRegistry,
)
from vllm.v1.kv_offload.remote_g2.load_spec import (
    RemoteG2LoadSpec,
    _RemoteBlockHandle,
)

logger = init_logger(__name__)

REMOTE_G2_PLAN_KEY = "remote_g2_plan"
_LoadSource = Literal["local", "remote"]


def block_hash_to_router_int(block_hash_bytes: bytes) -> int:
    """Lossy translation from vLLM BlockHash (XXH3-128 / SHA-256 bytes)
    to a 64-bit int compatible with what the Dynamo Router observes on
    the publisher event stream.

    The projection must match vLLM's ``maybe_convert_block_hash``
    (vllm/v1/core/kv_cache_utils.py), which is what the Router actually
    sees on BlockStored events when
    ``VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=True`` (the default):

        int.from_bytes(hash_bytes, "big") & ((1 << 64) - 1)

    For multi-byte hashes (XXH3-128 → 16 bytes, SHA-256 → 32 bytes),
    masking the full big-endian integer to 64 bits is equivalent to
    keeping the **last 8 bytes**, NOT the first 8 bytes. The earlier
    implementation took the first 8 bytes and therefore produced a
    different int than what the Router indexer stored for the same
    block — every native-plan resolve returned ``missing`` because the
    source's ``_records`` keys and the Router's chain kv_block_hashes
    were two different projections of the same data.
    """
    full = int.from_bytes(block_hash_bytes, "big", signed=False)
    return full & ((1 << 64) - 1)


class RemoteG2OffloadingManager(CPUOffloadingManager):
    """CPUOffloadingManager + Remote G2 source registry + plan path."""

    def __init__(
        self,
        num_blocks: int,
        *,
        source_worker_id: int,
        source_dp_rank: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        lease_ttl_ms: int = 30_000,
        tier: str = "host_pinned",
        pool_id: str = "g2-host-pinned",
        device_id: int = 0,
    ) -> None:
        super().__init__(
            num_blocks=num_blocks,
            cache_policy=cache_policy,
            enable_events=enable_events,
            store_threshold=store_threshold,
            max_tracker_size=max_tracker_size,
        )
        # Shared singleton: both the scheduler-side spec and the
        # worker-side spec resolve the same registry instance via
        # ``get_or_create``, since in vLLM v1 each role constructs its
        # own ``RemoteG2OffloadingSpec``.
        self.registry = SourceG2DescriptorRegistry.get_or_create(
            source_worker_id=source_worker_id,
            source_dp_rank=source_dp_rank,
            lease_ttl_ms=lease_ttl_ms,
        )
        # Share the registry's RLock for all manager-level operations.
        # This is mandatory: the registry's resolve_and_lease path
        # bumps policy.ref_cnt under the registry lock, and our
        # complete_store / prepare_store / lookup paths mutate the
        # same policy under self._rlock. Using a single lock avoids
        # an AB-BA deadlock between the scheduler thread and the
        # SourceG2RpcServer thread.
        self._rlock = self.registry._lock
        # Register our policy + last-wins: in vLLM v1's EngineCore
        # startup, ``_initialize_kv_caches`` runs BEFORE the Scheduler
        # is constructed, so the WORKER-role connector is built first
        # (its policy is empty since ``complete_store`` is only ever
        # run on the SCHEDULER side). Last-wins guarantees the
        # scheduler's populated policy is what the source RPC sees
        # when it pins blocks for an inbound lease.
        self.registry.set_policy(
            self._policy,
            acquire_block_ref=self._acquire_block_ref,
            release_block_ref=self._release_block_ref,
        )
        # Override the medium label on BlockStored / BlockRemoved events so the
        # Dynamo Router classifies our blocks as host-pinned. The inherited
        # CPUOffloadingManager publishes ``self.medium = "CPU"`` (from
        # ``CPULoadStoreSpec.medium()``), but the Rust KV router only
        # recognises ``"CPU_PINNED"`` / ``"CPU_TIER1"`` as the host-pinned
        # tier via ``StorageTier::from_kv_medium``. Without this override the
        # Router's per-worker ``host_pinned blocks`` count stays 0 and
        # ``select_remote_g2_reuse_plan`` never proposes a plan even when our
        # workers have published descriptors.
        self.medium = "CPU_PINNED"
        # Keep these on the manager for reset_cache /tear-down only —
        # the registry holds the authoritative pool layout.
        self._tier = tier
        self._pool_id = pool_id
        self._device_id = int(device_id)

        # Per-request resolve cache: req_id -> ResolveResult.
        self._resolve_cache: dict[str, _ResolveCacheEntry] = {}
        # Successful worker READs release their lease before scheduler
        # completion.  Keep consumed entries only for an idempotent retry at
        # request finish/reset; they are never eligible for another load.
        self._consumed_resolve_entries: dict[str, list[_ResolveCacheEntry]] = {}
        # The scheduler accepts exactly one source LoadStoreSpec per load job.
        # Lock each request to the first source that produced a lookup hit so
        # a local/remote boundary terminates the prefix instead of producing a
        # mixed batch that neither backend can load.
        self._load_source_by_req: dict[str, _LoadSource] = {}
        # Authoritative source of a successfully prepared, in-flight load.
        # request_finished may run before complete_load on cancellation, so
        # this marker (and a remote resolve entry) must outlive lookup state.
        self._prepared_load_source_by_req: dict[str, _LoadSource] = {}
        self._finished_with_pending_load: set[str] = set()
        # Target-side RPC client for plan resolves; populated by the
        # spec once a plan is observed.
        self._target_client_factory: TargetClientFactory | None = None
        # Counters live on the shared registry so the worker-side RPC
        # server (different Spec instance, same registry singleton) can
        # report them.

    # --- pool wiring (worker side) ---

    def set_pool_layout(
        self,
        *,
        layer_pool_base_ptrs: list[int],
        layer_pool_size_bytes: list[int],
        page_size_bytes: int,
        rank: int = 0,
        num_workers: int = 1,
        row_stride_bytes: int | None = None,
    ) -> None:
        """Forward pool layout to the shared registry.

        Required before descriptors can carry a working
        ``nixl_memory_desc``. Called from the worker-side spec with
        one base pointer + size per transformer layer (vLLM v1
        allocates one CPU tensor per layer).
        """
        self.registry.set_pool_layout(
            layer_pool_base_ptrs=layer_pool_base_ptrs,
            layer_pool_size_bytes=layer_pool_size_bytes,
            page_size_bytes=page_size_bytes,
            rank=rank,
            num_workers=num_workers,
            row_stride_bytes=row_stride_bytes,
            tier=self._tier,
            pool_id=self._pool_id,
            device_id=self._device_id,
        )

    def set_target_client_factory(self, factory: TargetClientFactory) -> None:
        self._target_client_factory = factory

    # --- plan-aware lookup ---

    @staticmethod
    def _peek_plan(req_context: ReqContext) -> RemoteKvReusePlan | None:
        params = req_context.kv_transfer_params
        if not params:
            return None
        raw = params.get(REMOTE_G2_PLAN_KEY)
        if raw is None:
            return None
        if isinstance(raw, RemoteKvReusePlan):
            return raw
        if isinstance(raw, Mapping):
            try:
                return RemoteKvReusePlan.from_dict(raw)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "RemoteG2: dropping malformed plan for req %s: %s",
                    req_context.req_id,
                    exc,
                )
                return None
        return None

    def _ensure_resolve(
        self, req_context: ReqContext, plan: RemoteKvReusePlan
    ) -> _ResolveCacheEntry | None:
        cache = self._resolve_cache.get(req_context.req_id)
        if cache is not None:
            return cache
        self.registry.plan_seen_count += 1
        if self._target_client_factory is None:
            logger.warning(
                "RemoteG2: plan present on req %s but no target client "
                "factory configured (peer_endpoints?)",
                req_context.req_id,
            )
            return None
        client = self._target_client_factory(plan)
        if client is None:
            return None
        try:
            result = client.resolve_and_lease(plan)
        except Exception:
            logger.exception(
                "RemoteG2: resolve_and_lease raised for req %s",
                req_context.req_id,
            )
            return None
        entry = _ResolveCacheEntry(
            plan=plan,
            result=result,
            descriptor_by_hash={d.block_hash: d for d in result.descriptors},
            client=client,
        )
        self._resolve_cache[req_context.req_id] = entry
        if result.lease_id is not None and result.descriptors:
            self.registry.plan_resolved_count += 1
            logger.debug(
                "RemoteG2: req %s plan %s resolved: %d descriptors, "
                "lease=%s, reason=%s",
                req_context.req_id,
                plan.plan_id,
                len(result.descriptors),
                result.lease_id,
                result.reason,
            )
        else:
            logger.debug(
                "RemoteG2: req %s plan %s resolve returned no descriptors (reason=%s)",
                req_context.req_id,
                plan.plan_id,
                result.reason,
            )
        return entry

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        req_id = req_context.req_id
        selected = self._load_source_by_req.get(req_id)
        if selected == "local":
            with self._rlock:
                return super().lookup(key, req_context)
        if selected == "remote":
            entry = self._resolve_cache.get(req_id)
            if entry is None:
                plan = self._peek_plan(req_context)
                if plan is None:
                    return LookupResult.MISS
                # A completed remote load consumes its lease.  A request that
                # is resumed/preempted and loads again must resolve fresh
                # descriptors rather than reuse offsets whose pins were
                # already released by the worker's on_load_done callback.
                entry = self._ensure_resolve(req_context, plan)
            if entry is None or entry.result.lease_id is None:
                return LookupResult.MISS
            block_hash_int = block_hash_to_router_int(get_offload_block_hash(key))
            return (
                LookupResult.HIT
                if block_hash_int in entry.descriptor_by_hash
                else LookupResult.MISS
            )

        with self._rlock:
            base = super().lookup(key, req_context)
        # The first positive source wins for the whole request.  A later key
        # that only exists in the other source terminates the maximal prefix.
        if base != LookupResult.MISS:
            # A reverse/sliding lookup may already have resolved a plan for a
            # different key.  Once local wins, that remote lease is unused;
            # release it immediately instead of pinning source capacity until
            # the request eventually finishes.
            self._release_resolve_entry(req_id, reason="local_selected")
            self._load_source_by_req[req_id] = "local"
            return base

        plan = self._peek_plan(req_context)
        if plan is None:
            return base
        entry = self._ensure_resolve(req_context, plan)
        if entry is None or entry.result.lease_id is None:
            return base
        block_hash_int = block_hash_to_router_int(get_offload_block_hash(key))
        if block_hash_int in entry.descriptor_by_hash:
            self._load_source_by_req[req_id] = "remote"
            return LookupResult.HIT
        return LookupResult.MISS

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        req_id = req_context.req_id
        if req_id in self._prepared_load_source_by_req:
            raise RuntimeError(f"RemoteG2: req {req_id} already has a prepared load")

        selected = self._load_source_by_req.get(req_id, "local")
        if selected == "local":
            with self._rlock:
                spec = super().prepare_load(keys, req_context)
            self._prepared_load_source_by_req[req_id] = "local"
            return spec

        entry = self._resolve_cache.get(req_id)
        if entry is None or entry.result.lease_id is None:
            raise RuntimeError(
                f"RemoteG2: req {req_id} selected remote load without a live lease"
            )

        handles: list[_RemoteBlockHandle] = []
        for key in keys:
            block_hash_int = block_hash_to_router_int(get_offload_block_hash(key))
            desc = entry.descriptor_by_hash.get(block_hash_int)
            if desc is None:
                raise RuntimeError(
                    f"RemoteG2: key {key!r} is not covered by the selected "
                    f"remote plan for req {req_id}"
                )
            handles.append(
                _RemoteBlockHandle(
                    block_hash=desc.block_hash,
                    descriptor_generation=desc.descriptor_generation,
                    byte_offset=desc.byte_offset,
                    byte_length=desc.byte_length,
                )
            )
        if not handles:
            raise RuntimeError(f"RemoteG2: req {req_id} prepared an empty remote load")

        self.registry.plan_load_specs_emitted += 1
        self.registry.plan_blocks_loaded += len(handles)
        # Track the bytes this spec will actually READ so ``complete_load``
        # can attribute them to a *completed* transfer (see counter docs in
        # data_model.py). Summed from descriptor byte_length == the exact
        # amount the transfer handler moves per block.
        entry.bytes_emitted += sum(int(h.byte_length) for h in handles)
        self._prepared_load_source_by_req[req_id] = "remote"
        logger.debug(
            "RemoteG2: req %s prepare_load -> RemoteG2LoadSpec "
            "(%d blocks, lease=%s, source_worker_id=%d)",
            req_context.req_id,
            len(handles),
            entry.result.lease_id,
            entry.plan.source_worker_id,
        )
        return RemoteG2LoadSpec(
            peer_name=str(entry.plan.source_tier)
            + ":"
            + str(entry.plan.source_worker_id),
            lease_id=entry.result.lease_id,
            blocks=handles,
            source_worker_id=entry.plan.source_worker_id,
            source_dp_rank=entry.plan.source_dp_rank,
        )

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        req_id = req_context.req_id
        selected = self._prepared_load_source_by_req.pop(req_id, None)
        if selected is None:
            raise RuntimeError(f"RemoteG2: req {req_id} completed an unprepared load")
        if selected == "local":
            with self._rlock:
                super().complete_load(keys, req_context)
        else:
            # Plan-driven load doesn't go through the CPU pool ref_cnt.  The
            # worker releases its source lease after the synchronous READ.
            # Consume the descriptor entry without another RPC on this hot
            # path; request finish/reset retains an idempotent cleanup retry.
            entry = self._resolve_cache.get(req_id)
            if entry is None or entry.result.lease_id is None:
                raise RuntimeError(
                    f"RemoteG2: req {req_id} lost its remote resolve entry"
                )
            if self.registry.record_completed_load(
                entry.bytes_emitted, entry.plan.source_worker_id
            ):
                entry.bytes_emitted = 0
            self._consume_resolve_entry(req_id)

        if req_id in self._finished_with_pending_load:
            self._finalize_request(req_id, reason="req_done_after_load")

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        with self._rlock:
            output = super().prepare_store(keys, req_context)
            if output is None:
                return None
            # Evictions get pulled out of the registry so peers stop
            # trying to resolve dead block hashes, AND drop their
            # hash → key mapping so pin attempts return "missing"
            # cleanly. (The LRU policy can't pick blocks with
            # ref_cnt > 0 in the first place, so evicted_keys here
            # were always unpinned — see policies/lru.py:41.)
            if output.evicted_keys:
                for key in output.evicted_keys:
                    block_hash_int = block_hash_to_router_int(
                        get_offload_block_hash(key)
                    )
                    self.registry.remove_descriptor(block_hash_int)
                    self.registry.forget_key(block_hash_int)
        return output

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        with self._rlock:
            super().complete_store(keys, req_context, success=success)
            if success:
                self._upsert_descriptors_locked(keys)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        with self._rlock:
            super().touch(keys, req_context)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return super().on_new_request(req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        req_id = req_context.req_id
        self._load_source_by_req.pop(req_id, None)
        if req_id in self._prepared_load_source_by_req:
            # The scheduler contract permits request_finished before a
            # pending load completion (e.g. cancellation).  Releasing a
            # remote lease here could expose its source blocks to eviction
            # while the READ is still in flight; retain all completion state.
            self._finished_with_pending_load.add(req_id)
            return
        self._finalize_request(req_id, reason="req_done")

    def _finalize_request(self, req_id: str, *, reason: str) -> None:
        self._load_source_by_req.pop(req_id, None)
        self._finished_with_pending_load.discard(req_id)
        self._release_resolve_entry(req_id, reason=reason)
        for entry in self._consumed_resolve_entries.pop(req_id, []):
            self._release_entry(entry, reason=reason)

    def _consume_resolve_entry(self, req_id: str) -> None:
        entry = self._resolve_cache.pop(req_id, None)
        if entry is not None:
            self._consumed_resolve_entries.setdefault(req_id, []).append(entry)

    def _release_resolve_entry(self, req_id: str, *, reason: str) -> None:
        entry = self._resolve_cache.pop(req_id, None)
        if entry is not None:
            self._release_entry(entry, reason=reason)

    @staticmethod
    def _release_entry(entry: _ResolveCacheEntry, *, reason: str) -> None:
        if entry.result.lease_id is not None:
            try:
                entry.client.release_lease(entry.result.lease_id, reason=reason)
            except Exception:
                logger.warning(
                    "RemoteG2: release_lease cleanup failed; "
                    "source-side cleanup is still required for lease %s",
                    entry.result.lease_id,
                )

    def take_events(self) -> Iterable[OffloadingEvent]:
        with self._rlock:
            events = list(super().take_events())
        yield from events

    def on_schedule_end(self) -> None:
        # A full-attention lookup stops on its first MISS; a sliding-window
        # lookup may inspect more keys but still select no source.  Either way,
        # a resolved-but-unselected entry must not pin remote capacity for the
        # lifetime of a request that will recompute.
        unused = [
            req_id
            for req_id in self._resolve_cache
            if req_id not in self._load_source_by_req
            and req_id not in self._prepared_load_source_by_req
        ]
        for req_id in unused:
            self._release_resolve_entry(req_id, reason="unused_resolve")

    def reset_cache(self) -> None:
        with self._rlock:
            # The connector queues local worker flushes before calling reset,
            # but it has no cross-worker completion/flush acknowledgement.
            # Releasing either side here could let a source buffer be reused
            # while a NIXL READ is still in flight.  Refuse the reset without
            # mutating state; callers may retry after loads and leases drain.
            if self._prepared_load_source_by_req:
                raise RuntimeError(
                    "RemoteG2: cannot reset cache with pending load jobs"
                )
            if self.registry._leases:
                raise RuntimeError(
                    "RemoteG2: cannot reset cache with active inbound leases"
                )

            super().reset_cache()
            # No leases are active, so descriptors and hash mappings can be
            # invalidated atomically with the local policy reset.
            for block_hash in list(self.registry._records.keys()):
                self.registry.remove_descriptor(block_hash)
                self.registry.forget_key(block_hash)

        # Resolved-but-never-submitted and already-consumed outbound entries
        # are safe to release after the local pool is empty: neither has an
        # in-flight READ.  This is outside the registry lock because it can do
        # a synchronous RPC to another worker.
        pending_req_ids = set(self._resolve_cache) | set(self._consumed_resolve_entries)
        for req_id in pending_req_ids:
            self._finalize_request(req_id, reason="reset_cache")
        self._load_source_by_req.clear()
        self._prepared_load_source_by_req.clear()
        self._finished_with_pending_load.clear()
        self._consumed_resolve_entries.clear()

    # --- registry population (must be called with _rlock held) ---

    def _upsert_descriptors_locked(self, keys: Iterable[OffloadKey]) -> None:
        if not self.registry.pool_layout_ready():
            return
        for key in keys:
            block = self._policy.get(key)
            if block is None or not block.is_ready:
                continue
            block_hash_int = block_hash_to_router_int(get_offload_block_hash(key))
            # Record (block_hash → key) so the registry can resolve a
            # pin request back to a policy block. Must happen BEFORE
            # upsert_for_block makes the descriptor visible to peers,
            # otherwise a remote resolve query could see the
            # descriptor but fail to pin its policy block.
            self.registry.record_key(block_hash_int, key)
            self.registry.upsert_for_block(block_hash_int, block.block_id)


# --- supporting types ---


class _TargetClient(Protocol):
    def resolve_and_lease(self, plan: RemoteKvReusePlan) -> RemoteG2ResolveResult: ...

    def release_lease(self, lease_id: str, reason: str = ...) -> bool: ...


TargetClientFactory = Callable[[RemoteKvReusePlan], "_TargetClient | None"]


@dataclass
class _ResolveCacheEntry:
    plan: RemoteKvReusePlan
    result: RemoteG2ResolveResult
    descriptor_by_hash: dict[int, RemoteG2Descriptor]
    client: _TargetClient
    # Bytes for which ``prepare_load`` actually emitted a RemoteG2LoadSpec
    # (a real NIXL READ), summed from descriptor ``byte_length``. Rolled
    # into the registry's post-completion counters by ``complete_load`` and
    # then zeroed, so re-fired completions never double-count.
    bytes_emitted: int = 0
