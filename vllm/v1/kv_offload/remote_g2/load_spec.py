# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoadStoreSpec carrying remote source info for a P2P transfer.

Emitted by ``RemoteG2OffloadingManager.prepare_load`` after a successful
plan resolve. The transfer handler registered for
``(RemoteG2LoadSpec, GPULoadStoreSpec)`` decodes this spec and issues
the NIXL READ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.v1.kv_offload.base import LoadStoreSpec


@dataclass
class _RemoteBlockHandle:
    """Per-block transfer coordinates on the remote source."""

    block_hash: int
    descriptor_generation: int
    byte_offset: int
    byte_length: int


@dataclass
class RemoteG2LoadSpec(LoadStoreSpec):
    """Spec for loading KV blocks from a remote source's host pool.

    ``peer_name`` is the NIXL agent name of the source (what
    ``RawNixlRemoteG2Adapter.add_peer`` was called with).
    ``lease_id`` is what the target sends back via ``release_lease``
    when the load completes.
    """

    peer_name: str = ""
    lease_id: str | None = None
    blocks: list[_RemoteBlockHandle] = field(default_factory=list)
    source_worker_id: int = -1
    source_dp_rank: int = -1

    @staticmethod
    def medium() -> str:
        return "REMOTE_G2"

    def __repr__(self) -> str:
        return (
            f"RemoteG2LoadSpec(peer={self.peer_name!r}, "
            f"lease={self.lease_id!r}, blocks={len(self.blocks)})"
        )
