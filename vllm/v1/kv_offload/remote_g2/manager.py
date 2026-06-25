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
        self.registry.set_policy(self._policy)
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

    def set_target_client_factory(
        self, factory: TargetClientFactory
    ) -> None:
        self._target_client_factory = factory

    # --- plan-aware lookup ---

    @staticmethod
    def _peek_plan(req_context: ReqContext) -> RemoteKvReusePlan | None:
        params = req_context.kv_transfer_params
        import logging as _dbgl_pp
        _dbgl_pp.getLogger(__name__).warning(
            "DBG_PEEK_PLAN: req_id=%s params_type=%s keys=%s",
            req_context.req_id,
            type(params).__name__,
            list(params.keys()) if isinstance(params, dict) else None,
        )
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
            descriptor_by_hash={
                d.block_hash: d for d in result.descriptors
            },
            client=client,
        )
        self._resolve_cache[req_context.req_id] = entry
        if result.lease_id is not None and result.descriptors:
            self.registry.plan_resolved_count += 1
            logger.info(
                "RemoteG2: req %s plan %s resolved: %d descriptors, "
                "lease=%s, reason=%s",
                req_context.req_id,
                plan.plan_id,
                len(result.descriptors),
                result.lease_id,
                result.reason,
            )
        else:
            logger.info(
                "RemoteG2: req %s plan %s resolve returned no descriptors "
                "(reason=%s)",
                req_context.req_id,
                plan.plan_id,
                result.reason,
            )
        return entry

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        with self._rlock:
            base = super().lookup(key, req_context)
        # Any local result (HIT / HIT_PENDING / RETRY) wins over the remote
        # plan; only fall back to the plan on a genuine local MISS.
        if base != LookupResult.MISS:
            return base

        plan = self._peek_plan(req_context)
        if plan is None:
            return base
        entry = self._ensure_resolve(req_context, plan)
        if entry is None or entry.result.lease_id is None:
            return base
        block_hash_int = block_hash_to_router_int(get_offload_block_hash(key))
        if block_hash_int in entry.descriptor_by_hash:
            return LookupResult.HIT
        return LookupResult.MISS

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        # Two cases:
        # 1. All keys are local (legacy CPU path) — defer to super().
        # 2. Any key is plan-resolved — emit a RemoteG2LoadSpec for the
        #    whole batch (POC: we don't split mixed batches; lookup
        #    ordering means a plan covers a contiguous prefix that the
        #    scheduler asks to load as one unit).
        plan = self._peek_plan(req_context)
        entry = self._ensure_resolve(req_context, plan) if plan else None
        if entry is None or entry.result.lease_id is None:
            with self._rlock:
                return super().prepare_load(keys, req_context)

        handles: list[_RemoteBlockHandle] = []
        all_remote = True
        for key in keys:
            block_hash_int = block_hash_to_router_int(
                get_offload_block_hash(key)
            )
            desc = entry.descriptor_by_hash.get(block_hash_int)
            if desc is None:
                all_remote = False
                break
            handles.append(
                _RemoteBlockHandle(
                    block_hash=desc.block_hash,
                    descriptor_generation=desc.descriptor_generation,
                    byte_offset=desc.byte_offset,
                    byte_length=desc.byte_length,
                )
            )
        if not all_remote or not handles:
            with self._rlock:
                return super().prepare_load(keys, req_context)

        self.registry.plan_load_specs_emitted += 1
        self.registry.plan_blocks_loaded += len(handles)
        logger.info(
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
        plan = self._peek_plan(req_context)
        if plan is None:
            with self._rlock:
                super().complete_load(keys, req_context)
            return
        # Plan-driven load doesn't go through the CPU pool ref_cnt, so
        # skip the super() decrement. Lease release happens via the
        # transfer handler's on_load_done callback.
        entry = self._resolve_cache.get(req_context.req_id)
        if entry is None or entry.result.lease_id is None:
            with self._rlock:
                super().complete_load(keys, req_context)

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

    def touch(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        with self._rlock:
            super().touch(keys, req_context)

    def on_new_request(
        self, req_context: ReqContext
    ) -> RequestOffloadingContext:
        return super().on_new_request(req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        entry = self._resolve_cache.pop(req_context.req_id, None)
        if entry is not None and entry.result.lease_id is not None:
            try:
                entry.client.release_lease(entry.result.lease_id, reason="req_done")
            except Exception:
                logger.warning(
                    "RemoteG2: release_lease on req finish failed; "
                    "source TTL will collect lease %s",
                    entry.result.lease_id,
                )

    def take_events(self) -> Iterable[OffloadingEvent]:
        with self._rlock:
            events = list(super().take_events())
        yield from events

    def reset_cache(self) -> None:
        with self._rlock:
            super().reset_cache()
            # Drop every descriptor, hash↔key entry, and active lease.
            # reset_cache is only called when the policy is being
            # fully wiped (e.g. on shutdown / engine restart), so any
            # outstanding lease is meaningless and any peer holding
            # one would error out on its next read regardless.
            for block_hash in list(self.registry._records.keys()):
                self.registry.remove_descriptor(block_hash)
                self.registry.forget_key(block_hash)
            for lease_id in list(self.registry._leases.keys()):
                self.registry.release_lease(lease_id, reason="reset_cache")

    # --- registry population (must be called with _rlock held) ---

    def _upsert_descriptors_locked(
        self, keys: Iterable[OffloadKey]
    ) -> None:
        if not self.registry.pool_layout_ready():
            return
        for key in keys:
            block = self._policy.get(key)
            if block is None or not block.is_ready:
                continue
            block_hash_int = block_hash_to_router_int(
                get_offload_block_hash(key)
            )
            # Record (block_hash → key) so the registry can resolve a
            # pin request back to a policy block. Must happen BEFORE
            # upsert_for_block makes the descriptor visible to peers,
            # otherwise a remote resolve query could see the
            # descriptor but fail to pin its policy block.
            self.registry.record_key(block_hash_int, key)
            self.registry.upsert_for_block(block_hash_int, block.block_id)


# --- supporting types ---


class _TargetClient(Protocol):
    def resolve_and_lease(
        self, plan: RemoteKvReusePlan
    ) -> RemoteG2ResolveResult: ...

    def release_lease(self, lease_id: str, reason: str = ...) -> bool: ...


TargetClientFactory = Callable[[RemoteKvReusePlan], "_TargetClient | None"]


@dataclass
class _ResolveCacheEntry:
    plan: RemoteKvReusePlan
    result: RemoteG2ResolveResult
    descriptor_by_hash: dict[int, RemoteG2Descriptor]
    client: _TargetClient
