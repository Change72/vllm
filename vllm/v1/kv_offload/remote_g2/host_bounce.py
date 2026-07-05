# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-slot host-bounce transport for same-node Remote-G2 reads.

The direct Remote-G2 path asks UCX to READ remote host memory straight into
VRAM.  On systems without a native host-to-device RMA lane, UCX emulates that
operation with small active-message fragments.  This opt-in transport instead
READs one bounded chunk into NIXL-registered local DRAM and scatters it to the
requested GPU blocks with vLLM's existing batched CUDA copy primitive.

``RemoteG2HostBounceTransport`` deliberately keeps the connector's synchronous
``submit_load`` contract: it pipelines chunks *within* one job, then drains all
CUDA events before returning.  Consequently a successful return means the KV
bytes are already visible to the later model forward pass.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import PIN_MEMORY

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.v1.kv_offload.remote_g2.nixl_adapter import (
        RawNixlRemoteG2Adapter,
    )


HOST_BOUNCE_SLOT_COUNT = 2


@dataclass(frozen=True)
class HostBounceTransferStats:
    """Stage timings for one completed host-bounce load.

    ``logical_bytes`` follows the existing Remote-G2 convention: descriptor
    bytes for one canonical layer.  Physical NIXL/H2D bytes multiply this by
    the adapter's canonical layer count.
    """

    logical_bytes: int
    chunk_count: int
    read_seconds: float
    copy_enqueue_seconds: float
    copy_wait_seconds: float


class HostBounceCopyEngine(Protocol):
    """Small interface that keeps the pipeline independently testable."""

    @property
    def layer_pool_base_ptrs(self) -> list[int]: ...

    @property
    def layer_pool_size_bytes(self) -> list[int]: ...

    def wait_slot(self, slot: int) -> float: ...

    def enqueue(
        self,
        slot: int,
        gpu_block_ids: Sequence[int],
        byte_lengths: Sequence[int],
    ) -> float: ...

    def drain(self, *, after_error: bool = False) -> float: ...

    def release(self) -> None: ...


class HostBounceTransferError(RuntimeError):
    """A failed pipeline transfer with explicit source-lease safety."""

    def __init__(self, message: str, *, safe_to_release_source_lease: bool) -> None:
        super().__init__(message)
        self.safe_to_release_source_lease = bool(safe_to_release_source_lease)


@dataclass
class _CudaBounceSlot:
    src_ptrs: torch.Tensor
    dst_ptrs: torch.Tensor
    sizes: torch.Tensor
    event: torch.Event
    active: bool = False


class CudaHostBounceCopyEngine:
    """Own two ordinary host buffers and scatter them to GPU KV blocks.

    The payload tensors are intentionally *not* allocated with
    ``pin_memory=True``.  The target NIXL agent is the sole owner of their host
    registration, avoiding a second ``cudaHostRegister`` of the same address.
    Descriptor tensors are pinned separately and remain private to each slot
    until that slot's CUDA event completes.
    """

    def __init__(
        self,
        gpu_tensors: Sequence[torch.Tensor],
        *,
        page_size_bytes: int,
        blocks_per_slot: int,
        slot_count: int = HOST_BOUNCE_SLOT_COUNT,
    ) -> None:
        if slot_count != HOST_BOUNCE_SLOT_COUNT:
            raise ValueError(
                f"host-bounce currently requires exactly {HOST_BOUNCE_SLOT_COUNT} slots"
            )
        if page_size_bytes <= 0:
            raise ValueError("page_size_bytes must be positive")
        if blocks_per_slot <= 0:
            raise ValueError("blocks_per_slot must be positive")
        if not gpu_tensors:
            raise ValueError("gpu_tensors must be non-empty")
        if not current_platform.is_cuda():
            raise RuntimeError("Remote-G2 host-bounce currently requires CUDA")

        self.page_size_bytes = int(page_size_bytes)
        self.blocks_per_slot = int(blocks_per_slot)
        self.slot_count = int(slot_count)
        self._closed = False
        self._gpu_tensors = list(gpu_tensors)
        self._validate_gpu_tensors()

        rows = self.slot_count * self.blocks_per_slot
        self._bounce_tensors = [
            torch.empty(
                (rows, self.page_size_bytes),
                dtype=torch.int8,
                device="cpu",
                pin_memory=False,
            )
            for _ in self._gpu_tensors
        ]
        # Commit the pages in this worker before NIXL registers them.  Worker
        # launchers can bind the process to the GPU-local NUMA node so this
        # first touch gives the registered pool the intended placement.
        for tensor in self._bounce_tensors:
            tensor.zero_()

        num_ops = self.blocks_per_slot * len(self._gpu_tensors)
        self._slots = [
            _CudaBounceSlot(
                src_ptrs=torch.empty(
                    num_ops, dtype=torch.int64, device="cpu", pin_memory=PIN_MEMORY
                ),
                dst_ptrs=torch.empty(
                    num_ops, dtype=torch.int64, device="cpu", pin_memory=PIN_MEMORY
                ),
                sizes=torch.empty(
                    num_ops, dtype=torch.int64, device="cpu", pin_memory=PIN_MEMORY
                ),
                event=torch.Event(),
            )
            for _ in range(self.slot_count)
        ]
        self._stream = current_platform.Stream()
        if int(self._stream.cuda_stream) == 0:
            raise RuntimeError("host-bounce requires a dedicated CUDA stream")

    def _validate_gpu_tensors(self) -> None:
        for index, tensor in enumerate(self._gpu_tensors):
            if tensor.dtype != torch.int8 or tensor.ndim != 2:
                raise ValueError(
                    f"GPU tensor {index} must be a 2-D int8 canonical cache"
                )
            if not tensor.is_cuda:
                raise ValueError(f"GPU tensor {index} is not CUDA-backed")
            if int(tensor.shape[1]) != self.page_size_bytes:
                raise ValueError(
                    f"GPU tensor {index} page size {tensor.shape[1]} does not "
                    f"match {self.page_size_bytes}"
                )
            if int(tensor.stride(0)) != self.page_size_bytes:
                raise ValueError(
                    f"GPU tensor {index} row stride {tensor.stride(0)} does not "
                    f"match page size {self.page_size_bytes}"
                )
            if int(tensor.stride(1)) != 1:
                raise ValueError(
                    f"GPU tensor {index} inner stride {tensor.stride(1)} is not 1"
                )

    @property
    def layer_pool_base_ptrs(self) -> list[int]:
        return [int(tensor.data_ptr()) for tensor in self._bounce_tensors]

    @property
    def layer_pool_size_bytes(self) -> list[int]:
        return [
            int(tensor.numel() * tensor.element_size())
            for tensor in self._bounce_tensors
        ]

    def _get_slot(self, slot: int) -> _CudaBounceSlot:
        if self._closed:
            raise RuntimeError("host-bounce copy engine is closed")
        if slot < 0 or slot >= self.slot_count:
            raise ValueError(f"invalid host-bounce slot {slot}")
        return self._slots[slot]

    def wait_slot(self, slot: int) -> float:
        state = self._get_slot(slot)
        if not state.active:
            return 0.0
        started = time.perf_counter()
        state.event.synchronize()
        elapsed = time.perf_counter() - started
        state.active = False
        return elapsed

    def enqueue(
        self,
        slot: int,
        gpu_block_ids: Sequence[int],
        byte_lengths: Sequence[int],
    ) -> float:
        state = self._get_slot(slot)
        if state.active:
            raise RuntimeError(f"host-bounce slot {slot} reused before completion")
        if len(gpu_block_ids) != len(byte_lengths):
            raise ValueError("gpu_block_ids and byte_lengths must have equal length")
        block_count = len(gpu_block_ids)
        if block_count == 0 or block_count > self.blocks_per_slot:
            raise ValueError(
                f"host-bounce chunk has {block_count} blocks; expected 1.."
                f"{self.blocks_per_slot}"
            )

        op_count = block_count * len(self._gpu_tensors)
        src = state.src_ptrs[:op_count]
        dst = state.dst_ptrs[:op_count]
        sizes = state.sizes[:op_count]
        src_values = src.numpy()
        dst_values = dst.numpy()
        size_values = sizes.numpy()
        slot_row = slot * self.blocks_per_slot

        op = 0
        for block_index, (gpu_block_id, byte_length) in enumerate(
            zip(gpu_block_ids, byte_lengths)
        ):
            gpu_block_id = int(gpu_block_id)
            byte_length = int(byte_length)
            if gpu_block_id < 0:
                raise ValueError(f"negative GPU block id {gpu_block_id}")
            if byte_length != self.page_size_bytes:
                raise ValueError(
                    f"invalid host-bounce byte length {byte_length}; page size "
                    f"is {self.page_size_bytes}"
                )
            for bounce, gpu in zip(self._bounce_tensors, self._gpu_tensors):
                if gpu_block_id >= int(gpu.shape[0]):
                    raise ValueError(
                        f"GPU block id {gpu_block_id} exceeds cache size {gpu.shape[0]}"
                    )
                src_values[op] = (
                    int(bounce.data_ptr())
                    + (slot_row + block_index) * self.page_size_bytes
                )
                dst_values[op] = (
                    int(gpu.data_ptr()) + gpu_block_id * self.page_size_bytes
                )
                size_values[op] = byte_length
                op += 1

        started = time.perf_counter()
        with current_platform.stream(self._stream):
            ops.swap_blocks_batch(
                src,
                dst,
                sizes,
                is_src_access_order_any=True,
            )
            state.event.record(self._stream)
        state.active = True
        return time.perf_counter() - started

    def drain(self, *, after_error: bool = False) -> float:
        """Wait for all submitted H2D work.

        In an enqueue-error path the event may not have been recorded even
        though the CUDA driver accepted a prefix of the batched copies.  A
        stream synchronization is therefore required in addition to draining
        known slot events.
        """

        if self._closed:
            return 0.0
        started = time.perf_counter()
        first_error: Exception | None = None
        for state in self._slots:
            if not state.active:
                continue
            try:
                state.event.synchronize()
                state.active = False
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if after_error:
            try:
                self._stream.synchronize()
                for state in self._slots:
                    state.active = False
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        elapsed = time.perf_counter() - started
        if first_error is not None:
            raise first_error
        return elapsed

    def release(self) -> None:
        """Drop CUDA and tensor references after NIXL deregistration."""

        if self._closed:
            return
        if any(state.active for state in self._slots):
            raise RuntimeError("cannot release active host-bounce slots")
        self._closed = True
        self._slots.clear()
        self._bounce_tensors.clear()
        self._gpu_tensors.clear()
        self._stream = None  # type: ignore[assignment]


class RemoteG2HostBounceTransport:
    """Pipeline synchronous NIXL READs with asynchronous batched H2D copies."""

    def __init__(
        self,
        *,
        adapter: RawNixlRemoteG2Adapter,
        copy_engine: HostBounceCopyEngine,
        page_size_bytes: int,
        blocks_per_slot: int,
        slot_count: int = HOST_BOUNCE_SLOT_COUNT,
    ) -> None:
        if slot_count != HOST_BOUNCE_SLOT_COUNT:
            raise ValueError(
                f"host-bounce currently requires exactly {HOST_BOUNCE_SLOT_COUNT} slots"
            )
        if page_size_bytes <= 0 or blocks_per_slot <= 0:
            raise ValueError("host-bounce page and chunk sizes must be positive")
        self.adapter = adapter
        self.copy_engine = copy_engine
        self.page_size_bytes = int(page_size_bytes)
        self.blocks_per_slot = int(blocks_per_slot)
        self.slot_count = int(slot_count)
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._closing = False
        self._poisoned = False

    def transfer(
        self,
        peer_name: str,
        *,
        peer_byte_offsets: Sequence[int],
        gpu_block_ids: Sequence[int],
        byte_lengths: Sequence[int],
    ) -> HostBounceTransferStats:
        block_count = len(peer_byte_offsets)
        if len(gpu_block_ids) != block_count or len(byte_lengths) != block_count:
            raise ValueError(
                "peer offsets, GPU block ids, and lengths must have equal length"
            )
        if block_count == 0:
            raise ValueError("host-bounce transfer requires at least one block")
        normalized_lengths = [int(value) for value in byte_lengths]
        invalid_lengths = [
            length for length in normalized_lengths if length != self.page_size_bytes
        ]
        if invalid_lengths:
            raise ValueError(
                "host-bounce requires full-page descriptors before issuing any "
                f"READ: page_size={self.page_size_bytes} invalid={invalid_lengths!r}"
            )

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("host-bounce transport is closed")
            if self._closing:
                raise RuntimeError("host-bounce transport is closing")
            if self._poisoned:
                raise RuntimeError("host-bounce transport is poisoned")

            read_seconds = 0.0
            copy_enqueue_seconds = 0.0
            copy_wait_seconds = 0.0
            chunks = 0
            # Source leases protect only the remote DRAM being READ.  Local
            # CUDA drain governs bounce-buffer lifetime, which is a separate
            # safety axis.  Mark source access uncertain only while the
            # synchronous NIXL call itself is in progress.
            source_access_terminal = True
            try:
                for start in range(0, block_count, self.blocks_per_slot):
                    stop = min(start + self.blocks_per_slot, block_count)
                    slot = chunks % self.slot_count
                    copy_wait_seconds += self.copy_engine.wait_slot(slot)

                    chunk_lengths = normalized_lengths[start:stop]
                    local_offsets = [
                        (slot * self.blocks_per_slot + local_block)
                        * self.page_size_bytes
                        for local_block in range(stop - start)
                    ]
                    read_started = time.perf_counter()
                    source_access_terminal = False
                    self.adapter.read_blocks(
                        peer_name,
                        peer_byte_offsets=peer_byte_offsets[start:stop],
                        local_byte_offsets=local_offsets,
                        byte_lengths=chunk_lengths,
                    )
                    source_access_terminal = True
                    read_seconds += time.perf_counter() - read_started
                    copy_enqueue_seconds += self.copy_engine.enqueue(
                        slot,
                        gpu_block_ids[start:stop],
                        chunk_lengths,
                    )
                    chunks += 1

                copy_wait_seconds += self.copy_engine.drain()
            except Exception as transfer_error:
                # Do not let a partial H2D continue writing GPU blocks while
                # the scheduler recomputes or reuses them.  A failure poisons
                # this transport; the synchronous exception stops forward.
                self._poisoned = True
                try:
                    self.copy_engine.drain(after_error=True)
                except Exception as drain_error:
                    raise HostBounceTransferError(
                        "host-bounce failed and CUDA drain could not be proven",
                        safe_to_release_source_lease=source_access_terminal,
                    ) from drain_error
                raise HostBounceTransferError(
                    "host-bounce transfer failed after draining submitted H2D",
                    safe_to_release_source_lease=source_access_terminal,
                ) from transfer_error

            return HostBounceTransferStats(
                logical_bytes=sum(int(v) for v in byte_lengths),
                chunk_count=chunks,
                read_seconds=read_seconds,
                copy_enqueue_seconds=copy_enqueue_seconds,
                copy_wait_seconds=copy_wait_seconds,
            )

    def close(self) -> None:
        """Drain, deregister, then release buffers; safe to call twice."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closing = True
            # If CUDA completion cannot be proven, intentionally retain both
            # the NIXL registration and tensor references.  Deregistering or
            # freeing a buffer that an unknown DMA operation may still read is
            # less safe than leaking it until process teardown.
            self.copy_engine.drain(after_error=self._poisoned)
            self.adapter.close()
            self.copy_engine.release()
            self._closed = True
