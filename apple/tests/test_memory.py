import pytest
from exl3_mlx.memory import MemoryBudget


def test_budget_fails_before_more_work(monkeypatch):
    guard = MemoryBudget(16, reserve_gib=0)
    monkeypatch.setattr(guard, "usage", lambda: 17 * 2**30)
    with pytest.raises(MemoryError, match="budget exceeded"):
        guard.check()
    monkeypatch.setattr(guard, "usage", lambda: 8 * 2**30)
    guard.check()


def test_advisory_allocator_limits_and_watchdog_are_restored(monkeypatch):
    import mlx.core as mx

    calls = []
    monkeypatch.setattr(mx, "set_memory_limit", lambda size: calls.append(("memory", size)) or 123)
    monkeypatch.setattr(mx, "set_cache_limit", lambda size: calls.append(("cache", size)) or 456)
    monkeypatch.setattr(MemoryBudget, "usage", lambda self: 0)
    with MemoryBudget(16, cache_mib=64, reserve_gib=0):
        pass
    assert calls == [("memory", 16 * 2**30), ("cache", 64 * 2**20), ("cache", 456), ("memory", 123)]
