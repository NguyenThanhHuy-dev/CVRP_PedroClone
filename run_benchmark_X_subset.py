#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Benchmark Runner for Route Optimizer V4 - DATASET X SUBSET
==========================================================
Chạy benchmark cho một tập con các instance khó trong bộ X.
Lặp 10 lần mỗi instance.
"""

import os
import sys
import time
import csv
import numpy as np
from datetime import datetime
import re
import importlib.util

TIMEOUT_LIMIT = 1200


# --- HÀM IMPORT FILE LINH HOẠT ---
def import_module_from_file(module_name, file_path):
    if not os.path.exists(file_path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import solver
route_optimizer_module = import_module_from_file(
    "route_optimizer_v4",
    "route_optimizer_v4.py",
)


# ---- CHỌN CÁC INSTANCE CẦN TEST ----
X_SUBSET = [
    "X-n101-k25",
    "X-n110-k13",
    "X-n115-k10",
    "X-n120-k6",
    "X-n125-k30",
    "X-n129-k18",
    "X-n134-k13",
    "X-n139-k10",
]


# --- ĐỌC THÔNG TIN INSTANCE ---
def get_instance_info(filepath):

    filename = os.path.basename(filepath)
    name = filename.replace(".vrp", "")

    match = re.search(r"n(\d+)", name)
    n = int(match.group(1)) if match else 0

    match = re.search(r"k(\d+)", name)
    k = int(match.group(1)) if match else 0

    bks = 0
    sol_filepath = filepath.replace(".vrp", ".sol")

    if os.path.exists(sol_filepath):
        with open(sol_filepath, "r") as f:
            for line in f:
                if line.startswith("Cost"):
                    try:
                        bks = int(line.split()[1])
                    except ValueError:
                        pass
                    break

    if bks == 0:
        print(f"⚠️  Không tìm thấy BKS trong {name}.sol")

    return name, n, k, bks


# --- GỌI SOLVER ---
def run_solver_wrapper(filepath):

    if not route_optimizer_module:
        raise Exception("File 'route_optimizer_v4.py' không tồn tại!")

    return route_optimizer_module.solve_with_clarke_wright_and_optimize(
        filepath, verbose=False
    )


# --- MAIN ---
def main():

    print("\n" + "=" * 70)
    print("BENCHMARK RUNNER FOR ROUTE OPTIMIZER V4 - DATASET X SUBSET")
    print("Runs per instance: 10")
    print("=" * 70)

    instances_list = []

    folder_path = os.path.join("instances", "X")

    if not os.path.isdir(folder_path):
        print(f"⚠️ Folder '{folder_path}' không tồn tại!")
        return

    print(f"\n📂 Sẽ benchmark {len(X_SUBSET)} instances trong bộ X")

    for name in X_SUBSET:

        filepath = os.path.join(folder_path, name + ".vrp")

        if not os.path.exists(filepath):
            print(f"⚠️ Không tìm thấy file {filepath}")
            continue

        name, n, k, bks = get_instance_info(filepath)

        instances_list.append(
            {"filepath": filepath, "name": name, "n": n, "k": k, "bks": bks}
        )

    result_file = f"benchmark_results_X_subset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"💾 Kết quả sẽ lưu vào: {result_file}")

    fieldnames = [
        "Date",
        "Method",
        "Instance",
        "N",
        "K",
        "BKS",
        "Best",
        "Avg",
        "Gap(%)",
        "Time(s)",
        "Runs",
        "n_customers",
        "n_restarts",
        "time_limit",
        "is_valid",
        "final_k",
    ]

    with open(result_file, mode="w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        print("\n" + "▒" * 70)
        print("▶️ BẮT ĐẦU BENCHMARK...")
        print("▒" * 70)

        for idx, data in enumerate(instances_list):

            print(
                f"\n[{idx+1}/{len(instances_list)}] {data['name']} (N={data['n']}, K={data['k']})"
            )

            filepath = data["filepath"]

            costs = []
            times = []
            params_list = []

            n_runs = 1

            for run in range(n_runs):

                try:

                    start_time = time.time()

                    routes, cost, params = run_solver_wrapper(filepath)

                    elapsed = time.time() - start_time

                    if elapsed > TIMEOUT_LIMIT:
                        print(f"  ⏱️ TIMEOUT after {elapsed:.1f}s")
                        break

                    costs.append(cost)
                    times.append(elapsed)
                    params_list.append(params)

                    print(f"  ✓ Run {run+1}: Cost={cost}, Time={elapsed:.2f}s")

                except Exception as e:

                    print(f"  ❌ Error: {str(e)[:100]}")

                    costs.append(float("inf"))
                    times.append(0)
                    params_list.append({})

            if costs and min(costs) < float("inf"):

                best_c = min(costs)
                avg_c = np.mean(costs)
                avg_t = np.mean(times)

                gap = (
                    ((best_c - data["bks"]) / data["bks"] * 100)
                    if data["bks"] > 0
                    else 0
                )

                gap_str = f"{gap:.2f}" if data["bks"] > 0 else "?"

                best_idx = costs.index(best_c)

                last_params = params_list[best_idx] if best_idx < len(params_list) else {}

                is_valid = last_params.get("is_valid", True)

                valid_icon = "✅ VALID" if is_valid else "⚠️ INVALID"

                print(
                    f"  📊 RESULT: Best={best_c} (Gap {gap_str}%) | Avg={avg_c:.1f} | AvgTime={avg_t:.2f}s | {valid_icon}"
                )

                writer.writerow(
                    {
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Method": "V4_MaxSAT_SoftConstraint",
                        "Instance": data["name"],
                        "N": data["n"],
                        "K": data["k"],
                        "BKS": data["bks"] if data["bks"] > 0 else "?",
                        "Best": best_c,
                        "Avg": f"{avg_c:.1f}",
                        "Gap(%)": gap_str,
                        "Time(s)": f"{avg_t:.2f}",
                        "Runs": n_runs,
                        "n_customers": last_params.get("n_customers", ""),
                        "n_restarts": last_params.get("n_restarts", ""),
                        "time_limit": last_params.get("time_limit", ""),
                        "is_valid": is_valid,
                        "final_k": last_params.get("final_k", ""),
                    }
                )

                csv_file.flush()

            else:
                print("  ❌ ALL RUNS FAILED")

    print("\n" + "▒" * 70)
    print(f"✅ BENCHMARK COMPLETED! Results saved to: {result_file}")
    print("▒" * 70)


if __name__ == "__main__":
    main()