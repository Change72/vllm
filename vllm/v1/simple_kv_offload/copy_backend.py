# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    launch_prepared_copy,
    prepare_copy,
)

logger = init_logger(__name__)


@dataclass
class DmaCopyEvent:
    """A completed/in-flight DMA copy with timing events and its byte size.

    ``start_event``/``end_event`` are ``enable_timing`` events bracketing only
    the DMA launch (host-side prep runs before ``start_event``). ``release``
    returns the event pair to the copy thread's pool once the worker has read
    the timing; it is cleared after the first call to avoid double-release.
    """

    event_idx: int
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    is_store: bool
    release: Callable[[], None] | None = None


class _EventPairPool:
    """Recycle ``enable_timing`` event pairs to avoid per-copy allocation.

    ``acquire`` is called on the copy thread; ``release`` on the worker thread.
    Backed by a thread-safe ``queue.SimpleQueue``.
    """

    def __init__(self, initial_size: int) -> None:
        self._pool: queue.SimpleQueue[tuple[torch.Event, torch.Event]] = (
            queue.SimpleQueue()
        )
        for _ in range(initial_size):
            self._pool.put(self._new_pair())

    @staticmethod
    def _new_pair() -> tuple[torch.Event, torch.Event]:
        return (
            torch.Event(enable_timing=True),
            torch.Event(enable_timing=True),
        )

    def acquire(self) -> tuple[torch.Event, torch.Event]:
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            return self._new_pair()

    def release(self, start_event: torch.Event, end_event: torch.Event) -> None:
        self._pool.put((start_event, end_event))


class DmaCopyBackend:
    """cuMemcpyBatchAsync copy backend (background thread)."""

    _EVENT_POOL_INITIAL_SIZE = 16

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream

        # Stores read the live KV cache -> STREAM (paired with the compute-done
        # wait in get_finished); loads read stable pinned host memory -> ANY.
        self._store_params = build_params(
            gpu_caches,
            cpu_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            cpu_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[DmaCopyEvent],
        wait_event: torch.Event | None = None,
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put(
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            )
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        event_pool = _EventPairPool(DmaCopyBackend._EVENT_POOL_INITIAL_SIZE)
        while True:
            item = q.get()
            if item is None:
                return
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            ) = item
            stream = store_stream if is_store else load_stream
            # #46278: enqueue the compute-done wait FIRST — before the host-side
            # prepare — so it captures the shared compute event's current state
            # and a subsequent step cannot re-record that event during prepare.
            if wait_event is not None:
                stream.wait_event(wait_event)
            # Host-side address/size prep runs before the timing bracket, so
            # kv_offload_*_time measures only the DMA (not the numpy prep, and
            # not the compute-done wait, which is ordered before start_event).
            prepared = prepare_copy(src_blocks, dst_blocks, params)
            start_event, end_event = event_pool.acquire()
            start_event.record(stream)
            if prepared is not None:
                launch_prepared_copy(prepared, params)
            end_event.record(stream)
            events_list.append(
                DmaCopyEvent(
                    event_idx=event_idx,
                    start_event=start_event,
                    end_event=end_event,
                    num_bytes=prepared.num_bytes if prepared is not None else 0,
                    is_store=is_store,
                    release=(
                        lambda s=start_event, e=end_event: event_pool.release(s, e)
                    ),
                )
            )
