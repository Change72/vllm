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
        self._target_clients: dict[int, TargetG2RpcClient] = {}
        self._clients_lock = threading.Lock()
        # The remote (NIXL) source/target wiring is brought up lazily on the
        # first get_worker() call, once the CPU pool tensors exist.
        self._remote_initialized = False

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
        # path runs against the per-layer ptrs from now on.
        manager.set_pool_layout(
            layer_pool_base_ptrs=cpu_layer_base_ptrs,
            layer_pool_size_bytes=cpu_layer_sizes,
            page_size_bytes=page_size_bytes,
            rank=0,
            num_workers=1,
        )
        logger.info(
            "RemoteG2OffloadingSpec: registered %d CPU layer pools "
            "(num_blocks=%d, page_size=%d bytes/block, total=%.2f GiB)",
            len(cpu_tensors),
            num_cpu_blocks,
            page_size_bytes,
            sum(cpu_layer_sizes) / (1 << 30),
        )

        # Source NIXL agent (or mock).
        agent_name = f"remote-g2-src-{self.source_worker_id}"
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

        # ZMQ REP server (binds the engine-side socket).
        if self.enable_source_rpc and self._rpc_server is None:
            self._rpc_server = SourceG2RpcServer(
                manager.registry,
                socket_path=self.source_rpc_socket_path,
                manager=manager,
            )
            if bundle is not None:
                self._rpc_server.set_nixl_bundle_provider(lambda b=bundle: b)
            self._rpc_server.start()
            logger.info(
                "RemoteG2OffloadingSpec source RPC at ipc://%s (source_worker_id=%d)",
                self.source_rpc_socket_path,
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
        target_agent_name = f"remote-g2-tgt-{self.source_worker_id}"
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
            self._target_clients.clear()

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
        with self._clients_lock:
            client = self._target_clients.get(plan.source_worker_id)
            if client is None:
                client = TargetG2RpcClient(sock)
                self._target_clients[plan.source_worker_id] = client
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
        client = None
        with self._clients_lock:
            client = self._target_clients.get(worker_id)
        if client is None:
            sock = self._peer_endpoints.get(worker_id)
            if sock is None:
                return False
            client = TargetG2RpcClient(sock)
            with self._clients_lock:
                self._target_clients.setdefault(worker_id, client)
                client = self._target_clients[worker_id]
        try:
            bundle = client.get_metadata(
                peer_agent_metadata=self._target_adapter.agent_metadata
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
        for client in list(self._target_clients.values()):
            try:
                if client.release_lease(lease_id, reason="load_done"):
                    return
            except Exception:
                continue
