# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remote G2 (host-pinned) KV offloading spec, manager and registry.

POC port of the TRT-LLM RemoteG2 KV-P2P connector to vLLM v1. Adds a
``RemoteG2OffloadingSpec`` (a ``CPUOffloadingSpec`` subclass) that exposes
the host pool to remote workers over NIXL, indexed by the source-side
``SourceG2DescriptorRegistry`` and served by a ZMQ REP loop.
"""
