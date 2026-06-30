# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remote G2 OffloadingSpec.

Subclass of ``CPUOffloadingSpec`` that:

* Reuses the inherited host pool (pinned CPU tensors) via a
  ``RemoteG2OffloadingWorker`` (a ``CPUOffloadingWorker`` subclass).
* Swaps the manager for ``RemoteG2OffloadingManager``.
* Brings up the source-side bits at first ``get_worker`` call:
  - Reads the host pool base pointers from the inherited worker's
    ``cpu_tensors`` and sets the pool layout on the manager.
  - Builds a NIXL source agent over the pool (or a mock when NIXL is
    unavailable) and registers a bundle provider so peers can fetch
    metadata via ``SourceG2RpcServer.get_metadata``.
  - Spawns the ZMQ REP server bound to the trtllm-compatible socket
    path so the dynamo source bridge can forward to us unchanged.
* Brings up the target-side bits in parallel:
  - Builds a NIXL target adapter over the local GPU pool.
  - Attaches a ``RemoteG2TransferHandler`` to the worker so that
    ``submit_load`` routes ``RemoteG2LoadSpec`` loads to it (NIXL READ).
  - Installs a ``TargetClientFactory`` on the manager that, given a
    plan, returns a ``TargetG2RpcClient`` pointed at the source.
"""

from __future__ import annotations

import contextlib
import os
import threading

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingWorker,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.remote_g2.data_model import RemoteKvReusePlan
from vllm.v1.kv_offload.remote_g2.manager import RemoteG2OffloadingManager
from vllm.v1.kv_offload.remote_g2.nixl_adapter import (
    NixlSourceBundle,
    RawNixlRemoteG2Adapter,
    build_source_agent,
)
from vllm.v1.kv_offload.remote_g2.source_rpc import (
    SourceG2RpcServer,
    default_socket_path,
)
from vllm.v1.kv_offload.remote_g2.target_client import TargetG2RpcClient
from vllm.v1.kv_offload.remote_g2.transfer_handler import (
    RemoteG2OffloadingWorker,
    RemoteG2TransferHandler,
)

logger = init_logger(__name__)


def _resolve_int(extra: dict, key: str, env: str, default: int) -> int:
    if key in extra:
        return int(extra[key])
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


def _resolve_str(extra: dict, key: str, env: str, default: str) -> str:
    if key in extra:
        return str(extra[key])
    return os.environ.get(env, default)


def _resolve_bool(extra: dict, key: str, env: str, default: bool) -> bool:
    if key in extra:
        return bool(extra[key])
    raw = os.environ.get(env)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class _BundleFromPayload:
    """NixlSourceBundle-shaped wrapper around a RemoteG2HandshakePayload.

    The source RPC server's ``get_metadata`` returns a dict serialised
    from a NixlSourceBundle. For the scheduler-side TP>1 path the
    bundle isn't local — we receive each rank's metadata via the
    handshake mechanism and need to expose it through the same
    interface. This tiny adapter wraps the payload so the RPC handler
    code doesn't need to special-case the source.
    """

    def __init__(self, payload):
        self.agent_desc = payload.agent_metadata
        self.layer_pool_base_ptrs = list(payload.layer_pool_base_ptrs)
        self.layer_pool_size_bytes = list(payload.layer_pool_size_bytes)
        self.page_size_bytes = int(payload.page_size_bytes)
        self.source_generation = int(payload.source_generation)
        self.remote_name = payload.agent_name
        self.pool_base_ptr = (
            int(payload.layer_pool_base_ptrs[0])
            if payload.layer_pool_base_ptrs else 0
        )
        self.pool_size_bytes = (
            int(payload.layer_pool_size_bytes[0])
            if payload.layer_pool_size_bytes else 0
        )


class RemoteG2OffloadingSpec(CPUOffloadingSpec):
    """CPUOffloadingSpec + Remote G2 (KV-P2P) capabilities.

    ``kv_connector_extra_config`` keys (env-fallback in parentheses):

    * ``cpu_bytes_to_use`` — inherited; required.
    * ``source_worker_id`` (env ``REMOTE_G2_SOURCE_WORKER_ID``) — required.
    * ``source_dp_rank`` (env ``REMOTE_G2_SOURCE_DP_RANK``) — default 0.
    * ``source_rpc_socket_path`` (env ``REMOTE_G2_SOURCE_RPC_SOCKET_PATH``)
      — defaults to ``/tmp/dynamo_remote_g2_ipc_<dynamo_pid>.sock``.
    * ``peer_endpoints`` — JSON-style ``"<worker_id>=<socket_path>,..."``
      mapping a peer source worker id to its REP socket. Consulted when
      the manager resolves a plan with that ``source_worker_id``. May be
      empty if dynamo will inject endpoints at runtime via
      ``set_peer_endpoint``.
    * ``use_mock_nixl`` (env ``REMOTE_G2_USE_MOCK_NIXL``) — force the
      bytes-memcpy transport even when ``nixl`` is installed (for tests).
    * ``lease_ttl_ms`` — default 30_000.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        # Router-driven KV-P2P reuse depends on self-describing KV events.
        # Since PR #43468 the offloading event stream only carries token_ids
        # and parent_block_hash when BOTH kv_events_config.enable_kv_cache_events
        # and kv_connector_extra_config["self_describing_kv_events"] are set;
        # otherwise BlockStored events use the legacy placeholder payload
        # (empty token_ids, no parent). The Dynamo router needs those fields to
        # index host-pinned offloaded blocks, so without them the router never
        # finds remote candidates and no reuse plan is produced. Warn loudly so
        # this misconfiguration fails visibly instead of silently no-op'ing.
        # NOTE: not fatal — the source-publishing / manually-injected-plan path
        # (e.g. the two_engines eval) does not use the router and works without
        # KV events, so a hard raise here would break that valid use.
        if not (
            self.kv_events_config.enable_kv_cache_events
            and self.kv_events_config.self_describing_kv_events
        ):
            logger.warning(
                "RemoteG2OffloadingSpec: router-driven KV-P2P reuse requires "
                "self-describing KV events, but they are not enabled "
                "(enable_kv_cache_events=%s, self_describing_kv_events=%s). "
                "Offloaded BlockStored events will use the legacy placeholder "
                "payload (no token_ids / parent_block_hash), so the Dynamo "
                "router cannot index host-pinned blocks and no reuse plan will "
                "be produced. Set kv_events_config.enable_kv_cache_events=true "
                "and kv_connector_extra_config['self_describing_kv_events']=true "
                "for the router-driven path.",
                self.kv_events_config.enable_kv_cache_events,
                self.kv_events_config.self_describing_kv_events,
            )
        # TP topology. world_size is known at construction (it's a config
        # value); the actual rank is read lazily from get_tp_group() at
        # get_worker() time because torch.distributed may not be fully
        # initialised here. ``cpu_page_size_per_worker`` is already
        # computed by CPUOffloadingSpec.__init__ above (= the per-rank
        # byte size of one offloaded block, totalled across all layers).
        self.tp_size: int = vllm_config.parallel_config.tensor_parallel_size
        self.tp_rank: int = -1  # resolved in get_worker()/_setup_remote()

        extra = self.extra_config
        self.source_worker_id = _resolve_int(
            extra, "source_worker_id", "REMOTE_G2_SOURCE_WORKER_ID", -1
        )
        if self.source_worker_id < 0:
            raise ValueError(
                "RemoteG2OffloadingSpec requires source_worker_id "
                "(kv_connector_extra_config or REMOTE_G2_SOURCE_WORKER_ID)"
            )
        self.source_dp_rank = _resolve_int(
            extra, "source_dp_rank", "REMOTE_G2_SOURCE_DP_RANK", 0
        )
        self.lease_ttl_ms = _resolve_int(
            extra, "lease_ttl_ms", "REMOTE_G2_LEASE_TTL_MS", 30_000
        )
        self.source_rpc_socket_path = _resolve_str(
            extra,
            "source_rpc_socket_path",
            "REMOTE_G2_SOURCE_RPC_SOCKET_PATH",
            default_socket_path(),
        )
        self.use_mock_nixl = _resolve_bool(
            extra, "use_mock_nixl", "REMOTE_G2_USE_MOCK_NIXL", False
        )
        self.enable_source_rpc: bool = bool(extra.get("enable_source_rpc", True))

        # peer_worker_id -> source rpc socket path. Populated either
        # from extra_config or via set_peer_endpoint() at runtime.
        self._peer_endpoints: dict[int, str] = {}
        for pair in str(extra.get("peer_endpoints", "")).split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            wid, path = pair.split("=", 1)
            try:
                self._peer_endpoints[int(wid.strip())] = path.strip()
            except ValueError:
                continue

        self._rpc_server: SourceG2RpcServer | None = None
        self._source_bundle: NixlSourceBundle | None = None
        self._target_adapter: RawNixlRemoteG2Adapter | None = None
        self._transfer_handler: RemoteG2TransferHandler | None = None
        # Legacy single-rank client cache (TP=1 path); kept for the
        # existing _ensure_peer code that still keys by worker_id only.
        self._target_clients: dict[int, TargetG2RpcClient] = {}
        # TP-aware client cache keyed by (source_worker_id, my_tp_rank).
        self._target_clients_by_rank: dict[
            tuple[int, int], TargetG2RpcClient
        ] = {}
        self._clients_lock = threading.Lock()
        # The remote (NIXL) source/target wiring is brought up lazily on the
        # first get_worker() call, once the CPU pool tensors exist.
        self._remote_initialized = False
        # Handshake-metadata state for TP>1 scheduler-coordinator path.
        # Worker fills _handshake_payload during get_worker()/_setup_remote() and exposes
        # it via get_handshake_metadata. Scheduler caches the merged
        # per-rank dict in _per_rank_handshake when EngineCore calls
        # set_xfer_handshake_metadata, then starts/updates the source
        # RPC server so its get_metadata can answer per-rank queries.
        from vllm.v1.kv_offload.remote_g2.data_model import (
            RemoteG2HandshakePayload as _Payload,
        )
        self._handshake_payload: _Payload | None = None
        self._per_rank_handshake: dict[int, _Payload] = {}

    # --- public helpers ---

    def set_peer_endpoint(self, worker_id: int, socket_path: str) -> None:
        self._peer_endpoints[int(worker_id)] = str(socket_path)

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))
            mgr = RemoteG2OffloadingManager(
                num_blocks=self.num_blocks,
                source_worker_id=self.source_worker_id,
                source_dp_rank=self.source_dp_rank,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=enable_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
                lease_ttl_ms=self.lease_ttl_ms,
            )
            mgr.set_target_client_factory(self._build_target_client)

            # In TP=1 the scheduler and worker are the same process, so
            # the worker's set_pool_layout call (in get_worker()/_setup_remote()) is
            # observable from the scheduler manager's index too. In
            # TP>1 they are *separate processes* — the worker call only
            # populates that worker's local registry singleton, and the
            # scheduler's singleton stays with pool_layout_ready=False,
            # which causes _upsert_descriptors_locked to skip every
            # block: descriptors are never published, and resolve calls
            # return "missing".
            #
            # Set a placeholder pool layout here using the values the
            # scheduler already knows (page_size_per_worker, num_blocks).
            # Pointers are placeholders because they are only used in
            # the descriptor's debug nixl_memory_desc.ptr field; the
            # real READ on the target side uses the source NIXL agent's
            # registered memory (carried in the per-rank NixlSourceBundle).
            # The worker-side get_worker()/_setup_remote() call later overwrites with
            # real pointers within the worker process.
            if self.cpu_page_size_per_worker > 0:
                try:
                    num_layers = sum(
                        len(g.layer_names)
                        for g in self.kv_cache_config.kv_cache_groups
                    )
                except Exception:
                    num_layers = 1
                num_layers = max(num_layers, 1)
                per_layer_page_size = (
                    self.cpu_page_size_per_worker // num_layers
                )
                per_layer_pool_size = (
                    self.num_blocks * per_layer_page_size
                )
                if not mgr.registry.pool_layout_ready():
                    mgr.set_pool_layout(
                        layer_pool_base_ptrs=[1] * num_layers,
                        layer_pool_size_bytes=[per_layer_pool_size]
                            * num_layers,
                        page_size_bytes=per_layer_page_size,
                        rank=0,
                        num_workers=1,
                    )

            self._manager = mgr
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> RemoteG2OffloadingWorker:
        return RemoteG2OffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
        )

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        worker = super().get_worker(kv_caches)
        assert isinstance(worker, RemoteG2OffloadingWorker)
        if not self._remote_initialized:
            self._setup_remote(worker, kv_caches)
            self._remote_initialized = True
        return worker

    def _setup_remote(
        self, worker: RemoteG2OffloadingWorker, kv_caches: CanonicalKVCaches
    ) -> None:
        """Bring up the source + target NIXL wiring and attach the remote
        load handler to ``worker``.

        Runs once, after ``CPUOffloadingWorker.__init__`` has materialised
        the inherited pinned-CPU pool tensors (``worker.cpu_tensors``).
        """
        manager = self.get_manager()
        assert isinstance(manager, RemoteG2OffloadingManager)

        # Resolve our TP rank now that get_tp_group() is reliably
        # initialised (this hook runs after model load + KV cache alloc,
        # well after torch.distributed setup).
        if self.tp_size > 1:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            self.tp_rank = int(get_tensor_model_parallel_rank())
            tp_world = int(get_tensor_model_parallel_world_size())
            assert tp_world == self.tp_size, (
                f"tp_size mismatch: config={self.tp_size} group={tp_world}"
            )
        else:
            self.tp_rank = 0

        cpu_tensors = worker.cpu_tensors
        if not cpu_tensors:
            logger.warning(
                "RemoteG2OffloadingSpec: no CPU tensors materialised; "
                "skipping NIXL agent setup."
            )
            return

        # vLLM v1 allocates one CPU tensor per transformer layer (e.g.
        # 36 for Qwen3-8B). All must be registered with NIXL and
        # transferred per block; otherwise non-layer-0 KV stays
        # uninitialised on the target and the model produces garbage.
        layer_page_sizes = [int(t.shape[-1]) for t in cpu_tensors]
        if len(set(layer_page_sizes)) != 1:
            raise RuntimeError(
                f"RemoteG2OffloadingSpec: heterogeneous per-layer CPU "
                f"page sizes are not supported in this POC: "
                f"{layer_page_sizes!r}. The transfer math assumes a "
                f"uniform stride across layers."
            )
        page_size_bytes = layer_page_sizes[0]
        num_cpu_blocks = int(cpu_tensors[0].shape[0])
        cpu_layer_base_ptrs = [int(t.data_ptr()) for t in cpu_tensors]
        cpu_layer_sizes = [page_size_bytes * int(t.shape[0]) for t in cpu_tensors]

        # Push pool layout to the shared registry; descriptor publish
        # path runs against the per-layer ptrs from now on. For TP>1
        # each worker process has its own independent per-rank CPU
        # pool — the local tensor is already this worker's slice with
        # ``page_size_bytes`` = the per-rank page size and the byte
        # offset within the local pool is just ``block_id *
        # page_size_bytes``. So we pass rank=0 num_workers=1: the
        # registry's formula then evaluates to that.
        # (The shared-mmap interleaved-row formula in set_pool_layout's
        # default — block_id * page_size * num_workers + rank *
        # page_size — only applies if the workers share one pool, which
        # is not the default vLLM allocation path.)
        manager.set_pool_layout(
            layer_pool_base_ptrs=cpu_layer_base_ptrs,
            layer_pool_size_bytes=cpu_layer_sizes,
            page_size_bytes=page_size_bytes,
            rank=0,
            num_workers=1,
        )
        logger.info(
            "RemoteG2OffloadingSpec[tp=%d/%d]: registered %d CPU layer "
            "pools (num_blocks=%d, page_size=%d bytes/block, "
            "total=%.2f GiB)",
            self.tp_rank,
            self.tp_size,
            len(cpu_tensors),
            num_cpu_blocks,
            page_size_bytes,
            sum(cpu_layer_sizes) / (1 << 30),
        )

        # Source NIXL agent (or mock). TP-aware naming so multiple
        # worker processes (one per TP rank) in the same instance do
        # not collide in the NIXL agent namespace.
        agent_name = (
            f"remote-g2-src-{self.source_worker_id}-tp{self.tp_rank}"
        )
        bundle = build_source_agent(
            agent_name,
            cpu_layer_base_ptrs,
            cpu_layer_sizes,
            page_size_bytes,
            source_generation=1,
            backends=() if self.use_mock_nixl else ("UCX",),
        )
        self._source_bundle = bundle
        if bundle is None:
            logger.warning(
                "RemoteG2OffloadingSpec: source NIXL bundle unavailable; "
                "peers cannot fetch metadata via get_metadata."
            )

        # Stash this rank's NIXL bundle as the handshake payload that
        # ``get_handshake_metadata`` will return up the collective_rpc
        # chain to the scheduler.
        if bundle is not None:
            from vllm.v1.kv_offload.remote_g2.data_model import (
                RemoteG2HandshakePayload,
            )
            self._handshake_payload = RemoteG2HandshakePayload(
                tp_rank=self.tp_rank,
                agent_name=agent_name,
                agent_metadata=bundle.agent_desc,
                layer_pool_base_ptrs=list(bundle.layer_pool_base_ptrs),
                layer_pool_size_bytes=list(bundle.layer_pool_size_bytes),
                page_size_bytes=int(bundle.page_size_bytes),
                source_generation=int(bundle.source_generation),
            )

        # ZMQ REP server location depends on TP topology:
        # - TP == 1: worker == scheduler (same Python process), so we
        #   start the RPC server here in the worker path. Legacy code.
        # - TP > 1: worker and scheduler are separate processes. The
        #   scheduler-side spec must own the source RPC server because
        #   the descriptor index lives in the scheduler's registry.
        #   The worker only publishes its NIXL agent via the handshake
        #   payload above; the scheduler-side ``set_xfer_handshake_metadata``
        #   starts the RPC server bound to the canonical (non-rank-scoped)
        #   path and answers ``get_metadata`` per tp_rank from the
        #   cached payloads.
        if self.tp_size == 1 and self.enable_source_rpc and self._rpc_server is None:
            rpc_path = self.source_rpc_socket_path
            self._rpc_server = SourceG2RpcServer(
                manager.registry,
                socket_path=rpc_path,
                manager=manager,
            )
            if bundle is not None:
                self._rpc_server.set_nixl_bundle_provider(lambda b=bundle: b)
            self._rpc_server.start()
            logger.info(
                "RemoteG2OffloadingSpec source RPC at ipc://%s "
                "(source_worker_id=%d, tp=1)",
                rpc_path,
                self.source_worker_id,
            )

        # Target-side adapter. Uses the local GPU pool tensors (one per
        # layer) as the transfer destination.
        gpu_layer_bases = [int(t.tensor.data_ptr()) for t in kv_caches.tensors]
        gpu_layer_sizes = [
            int(t.tensor.numel() * t.tensor.element_size()) for t in kv_caches.tensors
        ]
        if len(gpu_layer_bases) != len(cpu_layer_base_ptrs):
            logger.warning(
                "RemoteG2OffloadingSpec: GPU has %d layer tensors but "
                "CPU pool has %d; target transfers will reject peers "
                "with a different layer count.",
                len(gpu_layer_bases),
                len(cpu_layer_base_ptrs),
            )
        target_agent_name = (
            f"remote-g2-tgt-{self.source_worker_id}-tp{self.tp_rank}"
        )
        try:
            self._target_adapter = RawNixlRemoteG2Adapter(
                target_agent_name,
                gpu_layer_bases,
                gpu_layer_sizes,
                backends=("UCX",),
                use_mock=self.use_mock_nixl,
            )
        except Exception:
            logger.exception(
                "RemoteG2OffloadingSpec: target adapter setup failed; "
                "remote loads will be unavailable until adapter recovers"
            )
            self._target_adapter = None

        if self._target_adapter is not None:
            gpu_page_size = int(kv_caches.tensors[0].page_size_bytes)
            self._transfer_handler = RemoteG2TransferHandler(
                adapter=self._target_adapter,
                gpu_page_size_bytes=gpu_page_size,
                ensure_peer=self._ensure_peer,
                on_load_done=self._on_load_done,
            )
            # The connector drives a single worker per spec; route
            # RemoteG2LoadSpec loads to this handler from submit_load.
            worker.remote_handler = self._transfer_handler

    def shutdown(self) -> None:
        if self._rpc_server is not None:
            self._rpc_server.stop()
            self._rpc_server = None
        with self._clients_lock:
            for client in self._target_clients.values():
                with contextlib.suppress(Exception):
                    client.close()
            for client in self._target_clients_by_rank.values():
                with contextlib.suppress(Exception):
                    client.close()
            self._target_clients.clear()
            self._target_clients_by_rank.clear()

    # ----------------------------------------------------------------
    # Handshake metadata hooks called by OffloadingConnector. The base
    # OffloadingSpec does not declare these; OffloadingConnector calls
    # them via getattr so a spec without them simply opts out.
    # ----------------------------------------------------------------

    def get_handshake_metadata(self) -> Any:
        """Worker side: return this rank's NIXL agent + pool layout.

        Called via collective_rpc("get_kv_connector_handshake_metadata")
        from EngineCore after KV cache registration. The wrapping vLLM
        machinery automatically tags the return value with this
        worker's tp_rank.
        """
        return self._handshake_payload

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, Any]
    ) -> None:
        """Scheduler side: cache the per-rank payloads and start the
        source RPC server bound to the canonical socket path.

        ``metadata`` is keyed by tp_rank with values of type
        ``RemoteG2HandshakePayload`` (or whatever each worker's
        ``get_handshake_metadata`` returned). We re-validate and ignore
        any None or mistyped entries gracefully.
        """
        from vllm.v1.kv_offload.remote_g2.data_model import (
            RemoteG2HandshakePayload,
        )
        per_rank: dict[int, RemoteG2HandshakePayload] = {}
        for tp_rank, payload in metadata.items():
            if not isinstance(payload, RemoteG2HandshakePayload):
                continue
            per_rank[int(tp_rank)] = payload
        if not per_rank:
            logger.warning(
                "RemoteG2: set_xfer_handshake_metadata received no "
                "valid RemoteG2HandshakePayload entries (got %d items)",
                len(metadata),
            )
            return
        self._per_rank_handshake.update(per_rank)
        logger.info(
            "RemoteG2: cached handshake payloads for tp_ranks=%s "
            "(source_worker_id=%d)",
            sorted(per_rank.keys()),
            self.source_worker_id,
        )

        # Start the scheduler-side source RPC server. The registry
        # already has descriptors via the placeholder set_pool_layout
        # call in get_manager(); resolve_and_lease therefore works.
        # get_metadata uses the per-rank cache populated above.
        if self.enable_source_rpc and self._rpc_server is None:
            manager = self.get_manager()
            self._rpc_server = SourceG2RpcServer(
                manager.registry,
                socket_path=self.source_rpc_socket_path,
                manager=manager,
            )
            # Wire a per-rank bundle provider: get_metadata(tp_rank=N)
            # returns rank N's NIXL agent metadata + pool layout.
            self._rpc_server.set_per_rank_bundle_provider(
                self._per_rank_bundle_provider
            )
            self._rpc_server.start()
            logger.info(
                "RemoteG2OffloadingSpec source RPC at ipc://%s "
                "(source_worker_id=%d, tp_size=%d) [scheduler]",
                self.source_rpc_socket_path,
                self.source_worker_id,
                self.tp_size,
            )

    def _per_rank_bundle_provider(self, tp_rank: int):
        """Return a NixlSourceBundle-like dict for the requested rank."""
        payload = self._per_rank_handshake.get(int(tp_rank))
        if payload is None:
            return None
        return _BundleFromPayload(payload)

    # --- TP-aware socket-path helpers ---

    @staticmethod
    def _rank_scoped_socket_path(base_path: str, tp_rank: int) -> str:
        """Compute the per-rank socket path from a base path.

        Convention: append ``_tp{rank}`` before the file extension (if
        any) so that ``/tmp/dynamo_remote_g2_w1.sock`` becomes
        ``/tmp/dynamo_remote_g2_w1_tp0.sock`` for rank 0. This is
        deterministic on both source and target, so the target can
        compute the source's RPC socket path without an extra config
        knob.
        """
        if "." in base_path.rsplit("/", 1)[-1]:
            head, _, tail = base_path.rpartition(".")
            return f"{head}_tp{tp_rank}.{tail}"
        return f"{base_path}_tp{tp_rank}"

    # --- target side wiring ---

    def _build_target_client(self, plan: RemoteKvReusePlan) -> TargetG2RpcClient | None:
        sock = self._peer_endpoints.get(int(plan.source_worker_id))
        if not sock:
            logger.warning(
                "RemoteG2: no peer endpoint registered for worker_id=%d; "
                "configure kv_connector_extra_config.peer_endpoints or "
                "call set_peer_endpoint() at boot",
                plan.source_worker_id,
            )
            return None
        # Cache keyed by (source_worker_id, my_tp_rank). For TP>1 the
        # source-side RPC is in scheduler so all our target ranks talk
        # to the same socket, but each rank uses its own client so
        # subsequent get_metadata calls carry the right tp_rank.
        cache_key = (int(plan.source_worker_id), int(self.tp_rank))
        with self._clients_lock:
            client = self._target_clients_by_rank.get(cache_key)
            if client is None:
                client = TargetG2RpcClient(sock)
                self._target_clients_by_rank[cache_key] = client
        return client

    def _ensure_peer(self, peer_name: str) -> bool:
        """Look up or perform the NIXL metadata handshake with a peer.

        Returns True once ``add_peer`` has been called (or already was).
        """
        if self._target_adapter is None:
            return False
        # peer_name format from manager.prepare_load:
        # f"{plan.source_tier}:{plan.source_worker_id}"
        try:
            _, worker_id_str = peer_name.split(":", 1)
            worker_id = int(worker_id_str)
        except ValueError:
            logger.warning("RemoteG2: bad peer_name %r", peer_name)
            return False
        # Reuse the per-rank client cache. For TP>1 the cache key keeps
        # the connection scoped to this target rank so we can pass our
        # tp_rank to get_metadata; the socket itself is the canonical
        # (non-rank-scoped) source path.
        cache_key = (worker_id, int(self.tp_rank))
        with self._clients_lock:
            client = self._target_clients_by_rank.get(cache_key)
        if client is None:
            sock = self._peer_endpoints.get(worker_id)
            if sock is None:
                return False
            client = TargetG2RpcClient(sock)
            with self._clients_lock:
                self._target_clients_by_rank.setdefault(cache_key, client)
                client = self._target_clients_by_rank[cache_key]
        try:
            bundle = client.get_metadata(
                peer_agent_metadata=self._target_adapter.agent_metadata,
                tp_rank=self.tp_rank,
            )
        except Exception:
            logger.exception("RemoteG2: get_metadata RPC failed for %s", peer_name)
            return False
        if bundle is None:
            return False
        try:
            # Prefer the multi-layer fields when the peer publishes
            # them; fall back to the legacy single-pool fields wrapped
            # as a one-layer list so the adapter API stays uniform.
            layer_bases = bundle.get("layer_pool_base_ptrs")
            layer_sizes = bundle.get("layer_pool_size_bytes")
            if not layer_bases or not layer_sizes:
                layer_bases = [int(bundle.get("pool_base_ptr", 0))]
                layer_sizes = [int(bundle.get("pool_size_bytes", 0))]
            self._target_adapter.add_peer(
                peer_name,
                peer_agent_metadata=bundle.get("agent_metadata", b""),
                peer_layer_pool_base_ptrs=[int(p) for p in layer_bases],
                peer_layer_pool_size_bytes=[int(s) for s in layer_sizes],
            )
            return True
        except Exception:
            logger.exception("RemoteG2: add_peer failed for %s", peer_name)
            return False

    def _on_load_done(self, lease_id: str) -> None:
        # Best-effort release. The plan-driven lookup cache also calls
        # release on request finish; one of the two paths will land.
        # Try per-rank clients first (TP>1 path), then legacy clients.
        candidates = list(self._target_clients_by_rank.values()) + list(
            self._target_clients.values()
        )
        for client in candidates:
            try:
                if client.release_lease(lease_id, reason="load_done"):
                    return
            except Exception:
                continue
