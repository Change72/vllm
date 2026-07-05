# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remote-read helper that pulls KV from a remote source's host pool.

Driven by ``RemoteG2OffloadingWorker.submit_load`` for the
``RemoteG2LoadSpec -> GPULoadStoreSpec`` direction. Decodes the spec,
ensures the source peer is known to the local NIXL adapter, then issues
bounded multi-block READs from the source's pool into the target's GPU pool.

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
from vllm.v1.kv_offload.remote_g2.host_bounce import (
    HostBounceTransferError,
    RemoteG2HostBounceTransport,
)
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
    (POC simplification): blocks are READ in submission order and successful
    completion is queued for ``get_finished``. Any transfer/completion failure
    raises synchronously so stale KV cannot reach the subsequent forward pass.
    """

    def __init__(
        self,
        *,
        adapter: RawNixlRemoteG2Adapter,
        gpu_page_size_bytes: int,
        ensure_peer: Callable[[str], bool],
        on_load_done: Callable[[str], None] | None = None,
        host_bounce: RemoteG2HostBounceTransport | None = None,
        on_host_bounce_result: Callable[[bool, int], None] | None = None,
    ) -> None:
        self._adapter = adapter
        self._gpu_page_size = int(gpu_page_size_bytes)
        self._ensure_peer = ensure_peer
        self._on_load_done = on_load_done
        self._host_bounce = host_bounce
        self._on_host_bounce_result = on_host_bounce_result
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._shutdown = False
        self._finished: deque[TransferResult] = deque()

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        with self._lifecycle_lock:
            if self._shutdown:
                raise RuntimeError("RemoteG2 transfer handler is shut down")
            return self._submit_load_locked(job_id, src_spec, dst_spec)

    def _submit_load_locked(
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
            message = (
                "RemoteG2TransferHandler: block count mismatch "
                f"src={len(src_spec.blocks)} dst={len(gpu_block_ids)}"
            )
            logger.error(
                "RemoteG2TransferHandler: block count mismatch src=%d dst=%d",
                len(src_spec.blocks),
                len(gpu_block_ids),
            )
            self._release_lease(src_spec.lease_id, success=False)
            if self._host_bounce is not None:
                self._record_host_bounce_result(success=False, num_bytes=0)
            raise RuntimeError(message)
        if not src_spec.blocks:
            self._release_lease(src_spec.lease_id, success=False)
            if self._host_bounce is not None:
                self._record_host_bounce_result(success=False, num_bytes=0)
            raise RuntimeError("RemoteG2 load has no blocks")

        if not self._ensure_peer(src_spec.peer_name):
            message = (
                f"RemoteG2 transfer peer {src_spec.peer_name!r} metadata not available"
            )
            logger.warning(message)
            self._release_lease(src_spec.lease_id, success=False)
            if self._host_bounce is not None:
                self._record_host_bounce_result(success=False, num_bytes=0)
            raise RuntimeError(message)

        t0 = time.perf_counter()
        if self._host_bounce is not None:
            return self._submit_host_bounce(job_id, src_spec, gpu_block_ids, t0)

        total_bytes = 0
        try:
            self._adapter.read_blocks(
                src_spec.peer_name,
                peer_byte_offsets=[handle.byte_offset for handle in src_spec.blocks],
                local_byte_offsets=[
                    int(gpu_block_id) * self._gpu_page_size
                    for gpu_block_id in gpu_block_ids
                ],
                byte_lengths=[handle.byte_length for handle in src_spec.blocks],
            )
            # Logical single-layer bytes. The perf gate multiplies this by the
            # model's layer count when reporting layer-wise wire-equivalent
            # bytes. Only claim bytes after the entire batched load succeeds.
            total_bytes = sum(handle.byte_length for handle in src_spec.blocks)
        except Exception as exc:
            logger.exception("RemoteG2 NIXL READ failed (job_id=%d)", job_id)
            # Do not queue a deferred failed result: start_kv_transfers runs
            # before forward, so a synchronous exception is the only way to
            # guarantee partial/stale KV cannot reach model execution.
            raise RuntimeError(f"RemoteG2 NIXL READ failed (job_id={job_id})") from exc

        # Force a device synchronization so the bytes UCX wrote into
        # VRAM are visible to subsequent CUDA kernels on the model's
        # compute stream. NIXL's check_xfer_state == "DONE" confirms
        # UCX completed the transfer, but the data lives on UCX's own
        # stream until we drain — without this sync, the first
        # post-load forward pass can observe stale / partial data on a
        # cached block, producing wrong tokens that the *next* forward
        # pass would generate correctly. The sync cost is amortised
        # across the whole batch (one sync per transfer_async call).
        if total_bytes > 0 and not self._adapter.use_mock:
            try:
                import torch

                torch.cuda.synchronize()
            except Exception as exc:
                logger.exception(
                    "RemoteG2: torch.cuda.synchronize after NIXL READ "
                    "failed (job_id=%d); rejecting the load",
                    job_id,
                )
                # The synchronous NIXL READ already returned, so source DRAM
                # is no longer in use even though local VRAM completion failed.
                self._release_lease(src_spec.lease_id, success=False)
                raise RuntimeError(
                    f"RemoteG2 CUDA completion failed (job_id={job_id})"
                ) from exc

        elapsed = time.perf_counter() - t0
        self._release_lease(src_spec.lease_id, success=True)
        self._enqueue(job_id, success=True, num_bytes=total_bytes, elapsed=elapsed)
        return True

    def _submit_host_bounce(
        self,
        job_id: int,
        src_spec: RemoteG2LoadSpec,
        gpu_block_ids: list[int],
        started: float,
    ) -> bool:
        assert self._host_bounce is not None
        try:
            stats = self._host_bounce.transfer(
                src_spec.peer_name,
                peer_byte_offsets=[handle.byte_offset for handle in src_spec.blocks],
                gpu_block_ids=gpu_block_ids,
                byte_lengths=[handle.byte_length for handle in src_spec.blocks],
            )
        except Exception as exc:
            # The source lease protects remote DRAM, not the local bounce
            # buffer. Release it once the NIXL READ is known terminal even if
            # local CUDA drain failed; retain it only when remote access itself
            # is uncertain, until explicit cleanup or source process teardown.
            safe_to_release = not isinstance(exc, HostBounceTransferError) or (
                exc.safe_to_release_source_lease
            )
            if safe_to_release:
                self._release_lease(src_spec.lease_id, success=False)
            self._record_host_bounce_result(success=False, num_bytes=0)
            logger.exception(
                "RemoteG2 host-bounce transfer failed before forward (job_id=%d)",
                job_id,
            )
            raise RuntimeError(
                f"RemoteG2 host-bounce transfer failed (job_id={job_id})"
            ) from exc

        elapsed = time.perf_counter() - started
        self._release_lease(src_spec.lease_id, success=True)
        self._record_host_bounce_result(success=True, num_bytes=stats.logical_bytes)
        self._enqueue(
            job_id,
            success=True,
            num_bytes=stats.logical_bytes,
            elapsed=elapsed,
        )
        logger.debug(
            "RemoteG2 host-bounce completed job_id=%d blocks=%d chunks=%d "
            "bytes=%d read=%.6fs enqueue=%.6fs wait=%.6fs total=%.6fs",
            job_id,
            len(src_spec.blocks),
            stats.chunk_count,
            stats.logical_bytes,
            stats.read_seconds,
            stats.copy_enqueue_seconds,
            stats.copy_wait_seconds,
            elapsed,
        )
        return True

    def _release_lease(self, lease_id: str | None, *, success: bool) -> None:
        if self._on_load_done is None or lease_id is None:
            return
        try:
            self._on_load_done(lease_id)
        except Exception:
            logger.exception(
                "RemoteG2 lease release callback raised after %s "
                "(lease_id=%s); source pin remains until explicit cleanup or "
                "process teardown",
                "success" if success else "failure",
                lease_id,
            )

    def _record_host_bounce_result(self, *, success: bool, num_bytes: int) -> None:
        if self._on_host_bounce_result is None:
            return
        try:
            self._on_host_bounce_result(bool(success), int(num_bytes))
        except Exception:
            logger.exception("RemoteG2 host-bounce metrics callback failed")

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
        with self._lifecycle_lock:
            if self._shutdown:
                return
            if self._host_bounce is not None:
                self._host_bounce.close()
            else:
                self._adapter.close()
            self._shutdown = True
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
        if self.remote_handler is not None:
            self.remote_handler.shutdown()
            self.remote_handler = None
        super().shutdown()
