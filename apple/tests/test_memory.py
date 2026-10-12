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


def test_system_reserve_is_monitored_after_startup(monkeypatch):
    from types import SimpleNamespace

    import exl3_mlx.memory as memory

    monkeypatch.setattr(memory.sys, "platform", "darwin")

    def available(gib):
        text = (
            f"Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            f"Pages free: {int(gib * 2**30 / 16384)}.\nPages inactive: 0.\nPages speculative: 0.\n"
        )
        monkeypatch.setattr(memory.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=text))

    guard = MemoryBudget(16, reserve_gib=32)
    available(49)
    guard.check_startup_headroom()
    available(40)
    guard.check_system_headroom()
    with pytest.raises(MemoryError, match="budget plus reserve"):
        guard.check_startup_headroom()
    available(31)
    with pytest.raises(MemoryError, match="system reserve"):
        guard.check_system_headroom()
    monkeypatch.setattr(
        memory.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="bad data")
    )
    with pytest.raises(MemoryError, match="Could not measure"):
        guard.check_system_headroom()
