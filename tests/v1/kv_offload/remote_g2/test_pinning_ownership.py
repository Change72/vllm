# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.kv_offload.remote_g2.spec import RemoteG2OffloadingSpec


@pytest.mark.parametrize(
    ("use_mock_nixl", "expected_background_pinning"),
    [(False, False), (True, True)],
)
def test_nixl_is_the_only_host_registration_owner(
    use_mock_nixl: bool, expected_background_pinning: bool
) -> None:
    spec = object.__new__(RemoteG2OffloadingSpec)
    spec.block_size_factor = 1
    spec.num_blocks = 8
    spec.use_mock_nixl = use_mock_nixl
    kv_caches = MagicMock()

    with patch(
        "vllm.v1.kv_offload.remote_g2.spec.RemoteG2OffloadingWorker"
    ) as worker_cls:
        spec.create_worker(kv_caches)

    worker_cls.assert_called_once_with(
        kv_caches=kv_caches,
        block_size_factor=1,
        num_cpu_blocks=8,
        enable_background_pinning=expected_background_pinning,
    )
