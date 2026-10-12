"""Fail-fast memory budget for the experimental CLI (not an OS hard RAM quota).

MLX's own memory limit is advisory. Check both MLX allocations and process peak
RSS, and keep an independent watchdog alive during long graph evaluations. A
watchdog violation terminates only this process; completed adapter checkpoints
remain on disk. Polling can overshoot a budget during an individual allocation.
"""

import os
import re
import resource
import subprocess
import sys
import threading

import mlx.core as mx


class MemoryBudget:
    def __init__(self, gib=16.0, *, cache_mib=64, reserve_gib=32.0):
        if not 1 <= gib <= 64 or not 0 <= cache_mib <= 1024 or reserve_gib < 0:
            raise ValueError(
                "Use 1..64 GiB process budget, 0..1024 MiB cache and nonnegative reserve"
            )
        self.limit = int(gib * 2**30)
        self.cache_limit = int(cache_mib * 2**20)
        self.reserve = int(reserve_gib * 2**30)
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def usage():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        # Metal allocations can be outside the RSS accounting. Avoid counting
        # overlapping unified-memory allocations twice; check each independently.
        return max(mx.get_active_memory() + mx.get_cache_memory(), rss)

    def check(self):
        used = self.usage()
        if used > self.limit:
            raise MemoryError(
                f"Training memory budget exceeded: {used / 2**30:.2f} GiB > {self.limit / 2**30:.2f} GiB"
            )

    def check_startup_headroom(self):
        if sys.platform != "darwin" or not self.reserve:
            return
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True, timeout=5)
        size = int(re.search(r"page size of (\d+) bytes", result.stdout).group(1))
        counts = {
            k: int(v) for k, v in re.findall(r"^(Pages [^:]+):\s+(\d+)\.", result.stdout, re.M)
        }
        reclaimable = (
            sum(counts[k] for k in ("Pages free", "Pages inactive", "Pages speculative")) * size
        )
        if reclaimable < self.limit + self.reserve:
            raise MemoryError(
                "Insufficient free/reclaimable system memory for the requested budget plus reserve"
            )

    def __enter__(self):
        self.check_startup_headroom()
        self._old_memory = mx.set_memory_limit(self.limit)
        self._old_cache = mx.set_cache_limit(self.cache_limit)
        try:
            self.check()
        except MemoryError:
            mx.set_cache_limit(self._old_cache)
            mx.set_memory_limit(self._old_memory)
            raise

        def watch():
            while not self._stop.wait(0.1):
                try:
                    self.check()
                except MemoryError as error:
                    print(
                        f"\n{error}. Aborting this training process; completed step checkpoints are preserved.",
                        file=sys.stderr,
                        flush=True,
                    )
                    os._exit(75)

        self._thread = threading.Thread(target=watch, name="exl3-memory-budget", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        mx.set_cache_limit(self._old_cache)
        mx.set_memory_limit(self._old_memory)
