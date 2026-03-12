#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Benchmark Runner - DATASET X (SELECTED INSTANCES)
=================================================

Chỉ chạy các instance tiêu biểu trong bộ X.
Phù hợp cho:
- test solver nhanh
- chạy benchmark cho paper
"""

import os
import sys
import time
import csv
import numpy as np
from datetime import datetime
import importlib.util

TIMEOUT_LIMIT = 1800


# import solver
def import_module_from_file(module_name, file_path):

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


solver = import_module_from_file(
    "route_optimizer_v4",
    "route_optimizer_v4.py"
)


# subset instances
X_SELECTED = [
    "X-n101-k25",
    "X-n106-k14",
    "X-n120-k6",
    "X-n125-k30",
    "X-n134-k13",
    "X-n143-k7",
    "X-n157-k13",
    "X-n176-k26",
    "X-n200-k36"
]


def run_solver(filepath):

    return solver.solve_with_clarke_wright_and_optimize(
        filepath,
        verbose=False
    )


def read_bks(filepath):

    sol = filepath.replace(".vrp", ".sol")

    if not os.path.exists(sol):
        return 0

    with open(sol) as f:
        for line in f:
            if line.startswith("Cost"):
                return int(line.split()[1])

    return 0


def main():

    print("\n" + "=" * 60)
    print("Benchmark X SUBSET")
    print("=" * 60)

    folder = os.path.join("instances", "X")

    instances = []

    for name in X_SELECTED:

        vrp = os.path.join(folder, name + ".vrp")

        if os.path.exists(vrp):

            bks = read_bks(vrp)

            instances.append({
                "name": name,
                "filepath": vrp,
                "bks": bks
            })

    result_file = f"benchmark_X_subset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print("Saving results to:", result_file)

    fields = [
        "Instance",
        "BKS",
        "Best",
        "Avg",
        "Gap(%)",
        "Time(s)"
    ]

    with open(result_file, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for data in instances:

            print("\nInstance:", data["name"])

            costs = []
            times = []

            for run in range(10):

                start = time.time()

                routes, cost, params = run_solver(data["filepath"])

                t = time.time() - start

                costs.append(cost)
                times.append(t)

                print(f"Run {run+1}: cost={cost} time={t:.2f}")

            best = min(costs)
            avg = np.mean(costs)
            avg_t = np.mean(times)

            bks = data["bks"]

            gap = ((best - bks) / bks * 100) if bks else 0

            print(
                f"Best={best}  Avg={avg:.1f}  Gap={gap:.2f}%"
            )

            writer.writerow({
                "Instance": data["name"],
                "BKS": bks,
                "Best": best,
                "Avg": f"{avg:.1f}",
                "Gap(%)": f"{gap:.2f}",
                "Time(s)": f"{avg_t:.2f}"
            })

    print("\nBenchmark finished")


if __name__ == "__main__":
    main()