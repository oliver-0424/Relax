# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU repetition benchmark.

Run from the repository root with PYTHONPATH=.
"""

import argparse
import json
import platform
import random
import resource
import statistics
import string
import subprocess
import sys
import time
import tracemalloc
import zlib
from pathlib import Path
from typing import Any

from relax.utils.repetition import detect_repetition


def benchmark_case(length: int, pattern: str, runs: int) -> dict[str, Any]:
    if pattern == "repetitive":
        text = "a" * length
    else:
        text = "".join(random.Random(42).choices(string.ascii_letters + string.digits, k=length))
        if pattern == "middle":
            start = (length - 10_000) // 2
            text = text[:start] + "a" * 10_000 + text[start + 10_000 :]
    detect_repetition(text)
    wall_times, cpu_times = [], []
    for _ in range(runs):
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        result = detect_repetition(text)
        wall_times.append((time.perf_counter() - wall_start) * 1_000)
        cpu_times.append((time.process_time() - cpu_start) * 1_000)
    # Separate allocation measurement avoids adding tracing overhead to timings.
    tracemalloc.start()
    detect_repetition(text)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "characters": length,
        "pattern": pattern,
        "window_count": result.window_count,
        "hit_count": len(result.hit_windows),
        "median_wall_ms": round(statistics.median(wall_times), 3),
        "median_cpu_ms": round(statistics.median(cpu_times), 3),
        "peak_traced_bytes": peak,
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1_024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--worker", nargs=2, metavar=("LENGTH", "PATTERN"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.worker:
        report = benchmark_case(int(args.worker[0]), args.worker[1], args.runs)
    else:
        cases = []
        for length in (10_000, 100_000, 1_000_000):
            for pattern in ("control", "middle", "repetitive"):
                worker = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--runs",
                        str(args.runs),
                        "--worker",
                        str(length),
                        pattern,
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                cases.append(json.loads(worker.stdout))
        report = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "zlib": zlib.ZLIB_RUNTIME_VERSION,
            "runs": args.runs,
            "seed": 42,
            "window_size": 10_000,
            "stride": 5_000,
            "threshold": 10.0,
            "memory_notes": "Traced peak is detector-only; RSS is a fresh process including imports and input creation.",
            "cases": cases,
        }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
