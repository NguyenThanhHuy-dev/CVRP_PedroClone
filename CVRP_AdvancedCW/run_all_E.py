#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
run_all_E.py
============
Chạy lần lượt 3 method: gurobi → cplex → pysat cho benchmark E-Set.
Sử dụng: python run_all_E.py
"""

import subprocess
import sys
import time

METHODS = ["gurobi", "cplex", "pysat"]

def main():
    total_start = time.time()
    results = {}

    print("=" * 70)
    print("AUTO RUNNER  |  BENCHMARK E-SET  |  3 METHODS")
    print("=" * 70)

    for method in METHODS:
        print(f"\n{'=' * 70}")
        print(f"[{method.upper()}] BẮT ĐẦU...")
        print(f"{'=' * 70}")

        start = time.time()
        ret = subprocess.run(
            [sys.executable, "run_benchmark_E.py", method],
            check=False,
        )
        elapsed = time.time() - start
        results[method] = {"returncode": ret.returncode, "elapsed": elapsed}

        status = "OK" if ret.returncode == 0 else f"LỖI (code={ret.returncode})"
        print(f"\n[{method.upper()}] KẾT THÚC — {status} | Thời gian: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print("TỔNG KẾT")
    print(f"{'=' * 70}")
    for method, r in results.items():
        status = "OK" if r["returncode"] == 0 else f"LỖI (code={r['returncode']})"
        print(f"  {method.upper():8s} | {status:20s} | {r['elapsed']:.1f}s")
    print(f"  {'TỔNG':8s} | {'':20s} | {total_elapsed:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()