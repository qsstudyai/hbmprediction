from types import SimpleNamespace

import pytest

from cost_eval.allocator_model import AllocatorModel
from cost_eval.specs import HardwareSpec


def _breakdown(**updates):
    values = dict(
        persistent=5, act_live=3, gather_buf=0, grad_buf=0,
        recomp_scratch=0, swap_buf=0, workspace=2,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def test_components_are_mutually_exclusive_and_reconcile():
    hardware = HardwareSpec(
        1000, device_baseline_bytes=7,
        untracked_runtime_point_bytes=11,
        allocator_granularity_bytes=4,
        calibrated_upper_margin_bytes=13,
        ood_margin_bytes=17,
    )
    result = AllocatorModel().estimate(10, _breakdown(), hardware)
    # Rounding (5, 3, 2) to 4-byte buckets contributes 6 bytes.
    assert result.fragmentation_point_bytes == 6
    assert result.total_point_bytes == 7 + 10 + 11 + 6
    assert result.safe_upper_bytes == result.total_point_bytes + 13 + 17


def test_allocator_pool_can_exceed_active_without_double_counting_active():
    hardware = HardwareSpec(
        1000, allocator_pool_point_bytes=20,
        allocator_granularity_bytes=1,
    )
    result = AllocatorModel().estimate(10, _breakdown(), hardware)
    assert result.total_point_bytes == 20


def test_negative_components_are_rejected():
    with pytest.raises(ValueError):
        HardwareSpec(1000, untracked_runtime_point_bytes=-1)
