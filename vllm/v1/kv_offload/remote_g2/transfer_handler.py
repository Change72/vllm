# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remote-read helper that pulls KV from a remote source's host pool.

Driven by ``RemoteG2OffloadingWorker.submit_load`` for the
``RemoteG2LoadSpec -> GPULoadStoreSpec`` direction. Decodes the spec,
ensures the source peer is known to the local NIXL adapter, then issues
one READ per block from the source's pool offset into the target's GPU
pool offset.

For the same-host POC where NIXL is unavailable, the underlying
``RawNixlRemoteG2Adapter`` falls back to a plain memcpy. The handler
code path is unchanged.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.remote_g2.load_spec import RemoteG2LoadSpec
from vllm.v1.kv_offload.remote_g2.nixl_adapter import RawNixlRemoteG2Adapter

if TYPE_CHECKING:
    from collections.abc import Callable

logger = init_logger(__name__)


class RemoteG2TransferHandler:
    """READ blocks from a remote host pool into local GPU blocks.

    This is a plain helper composed by ``RemoteG2OffloadingWorker``: the
    worker routes ``submit_load`` calls whose source is a
    ``RemoteG2LoadSpec`` here, and merges this handler's completions into
    its own ``get_finished``.

    The handler maintains a small registry of NIXL peers; the first
    transfer for an unseen ``peer_name`` triggers a ``get_metadata``
    handshake via the supplied ``ensure_peer`` callback.

    The transfer is performed synchronously inside ``submit_load``
    (POC simplification): blocks are READ in submission order, success
    or failure is queued for ``get_finished``. M4 can switch to a real
    async pump using NIXL's ``post()`` + completion polling.
    """

    def __init__(
        self,
        *,
        adapter: RawNixlRemoteG2Adapter,
        gpu_page_size_bytes: int,
        ensure_peer: Callable[[str], bool],
        on_load_done: Callable[[str], None] | None = None,
    ) -> None:
        self._adapter = adapter
        self._gpu_page_size = int(gpu_page_size_bytes)
        self._ensure_peer = ensure_peer
        self._on_load_done = on_load_done
        self._lock = threading.Lock()
        self._finished: deque[TransferResult] = deque()

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        if not isinstance(src_spec, RemoteG2LoadSpec):
            logger.error(
                "RemoteG2TransferHandler: bad src spec type %r", type(src_spec)
            )
            return False
        if not isinstance(dst_spec, GPULoadStoreSpec):
            logger.error(
                "RemoteG2TransferHandler: bad dst spec type %r", type(dst_spec)
            )
            return False

        gpu_block_ids = dst_spec.block_ids.tolist()
        if len(src_spec.blocks) != len(gpu_block_ids):
            logger.error(
                "RemoteG2TransferHandler: block count mismatch src=%d dst=%d",
                len(src_spec.blocks),
                len(gpu_block_ids),
            )
            return False

        if not self._ensure_peer(src_spec.peer_name):
            logger.warning(
                "RemoteG2 transfer dropped: peer %s metadata not available",
                src_spec.peer_name,
            )
            self._enqueue(job_id, success=False, num_bytes=0, elapsed=0.0)
            return True

        t0 = time.perf_counter()
        total_bytes = 0
        ok = True
        try:
            for handle, gpu_block_id in zip(src_spec.blocks, gpu_block_ids):
                local_offset = int(gpu_block_id) * self._gpu_page_size
                self._adapter.read_block(
                    src_spec.peer_name,
                    peer_byte_offset=handle.byte_offset,
                    local_byte_offset=local_offset,
                    byte_length=handle.byte_length,
                )
                total_bytes += handle.byte_length
        except Exception:
            logger.exception("RemoteG2 NIXL READ failed (job_id=%d)", job_id)
            ok = False

        # Force a device synchronization so the bytes UCX wrote into
        # VRAM are visible to subsequent CUDA kernels on the model's
        # compute stream. NIXL's check_xfer_state == "DONE" confirms
        # UCX completed the transfer, but the data lives on UCX's own
        # stream until we drain — without this sync, the first
        # post-load forward pass can observe stale / partial data on a
        # cached block, producing wrong tokens that the *next* forward
        # pass would generate correctly. The sync cost is amortised
        # across the whole batch (one sync per transfer_async call).
        if ok and total_bytes > 0:
            try:
                import torch

                torch.cuda.synchronize()
            except Exception:
                logger.warning(
                    "RemoteG2: torch.cuda.synchronize after NIXL READ "
                    "failed (job_id=%d); subsequent forward may observe "
                    "stale GPU bytes",
                    job_id,
                    exc_info=True,
                )

        elapsed = time.perf_counter() - t0
        self._enqueue(job_id, success=ok, num_bytes=total_bytes, elapsed=elapsed)

        if ok and self._on_load_done is not None and src_spec.lease_id is not None:
            try:
                self._on_load_done(src_spec.lease_id)
            except Exception:
                logger.exception(
                    "RemoteG2 lease release callback raised "
                    "(lease_id=%s); source TTL will clean up",
                    src_spec.lease_id,
                )
        return True

    def get_finished(self) -> list[TransferResult]:
        with self._lock:
            drained = list(self._finished)
            self._finished.clear()
        return drained

    def wait(self, job_ids: set[int]) -> None:
        # All transfers complete synchronously inside transfer_async, so
        # by the time wait() is called there is nothing in flight.
        return

    def shutdown(self) -> None:
        with self._lock:
            self._finished.clear()

    def _enqueue(
        self, job_id: int, *, success: bool, num_bytes: int, elapsed: float
    ) -> None:
        with self._lock:
            self._finished.append(
                TransferResult(
                    job_id=job_id,
                    success=success,
                    transfer_size=num_bytes,
                    transfer_time=elapsed,
                )
            )


class RemoteG2OffloadingWorker(CPUOffloadingWorker):
    """CPUOffloadingWorker + a remote (KV-P2P) load path.

    Stores and CPU-backed loads use the inherited GPU<->CPU handlers
    unchanged. Loads whose source is a ``RemoteG2LoadSpec`` are routed to
    an attached ``RemoteG2TransferHandler`` (a NIXL READ from a peer's
    host pool). The remote handler is wired up by
    ``RemoteG2OffloadingSpec.get_worker`` once the CPU pool tensors have
    been materialised, via :attr:`remote_handler`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.remote_handler: RemoteG2TransferHandler | None = None

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        if isinstance(src_spec, RemoteG2LoadSpec):
            if self.remote_handler is None:
                logger.error(
                    "RemoteG2OffloadingWorker: remote load requested before "
                    "the NIXL handler was attached (job_id=%d)",
                    job_id,
                )
                return False
            return self.remote_handler.submit_load(job_id, src_spec, dst_spec)
        return super().submit_load(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        finished = super().get_finished()
        if self.remote_handler is not None:
            finished = finished + self.remote_handler.get_finished()
        return finished

    def wait(self, job_ids: set[int]) -> None:
        super().wait(job_ids)
        if self.remote_handler is not None:
            self.remote_handler.wait(job_ids)

    def shutdown(self) -> None:
        super().shutdown()
        if self.remote_handler is not None:
            self.remote_handler.shutdown()
