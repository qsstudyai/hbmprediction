import pytest

from cost_eval.memory_actions import MemoryAction
from cost_eval.memory_ledger import MemoryLedger


def test_peak_snapshot_uses_simultaneous_bucket_values():
    ledger = MemoryLedger()
    actions = (
        MemoryAction("ALLOC", "saved", 100, "act_live", "save"),
        MemoryAction("ALLOC", "workspace", 60, "workspace", "kernel"),
        MemoryAction("FREE", "workspace", owner="kernel_end"),
    )
    peak = ledger.replay(actions, allowed_live={"saved"})
    assert peak.bytes == 160
    assert peak.owner == "kernel"
    assert peak.buckets == {"act_live": 100, "workspace": 60}


def test_duplicate_free_and_leak_are_rejected():
    ledger = MemoryLedger()
    ledger.apply(MemoryAction("ALLOC", "x", 1))
    ledger.apply(MemoryAction("FREE", "x"))
    with pytest.raises(ValueError, match="释放未分配"):
        ledger.apply(MemoryAction("FREE", "x"))

    with pytest.raises(ValueError, match="ledger leak"):
        MemoryLedger().replay((MemoryAction("ALLOC", "leak", 1),))


def test_alias_does_not_allocate_and_blocks_owner_free():
    ledger = MemoryLedger()
    ledger.apply(MemoryAction("ALLOC", "storage", 10, "act_live"))
    ledger.apply(MemoryAction("ALIAS", "view", alias_of="storage"))
    assert ledger.live_bytes == 10
    with pytest.raises(ValueError, match="alias 引用"):
        ledger.apply(MemoryAction("FREE", "storage"))
    ledger.apply(MemoryAction("FREE", "view"))
    ledger.apply(MemoryAction("FREE", "storage"))
    assert ledger.live_bytes == 0
