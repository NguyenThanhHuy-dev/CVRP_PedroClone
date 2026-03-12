#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Benchmark Runner for Route Optimizer V4 - DATASET B ONLY
========================================================
Chạy benchmark riêng cho bộ B, lặp 10 lần mỗi instance.
Sử dụng file route_optimizer_v4.py (đã có Soft Constraints).
"""

import os
import sys
import time
import csv
import numpy as np
from datetime import datetime
import re
import importlib.util

TIMEOUT_LIMIT = 1200  # Giới hạn 20 phút cho mỗi instance


# --- HÀM IMPORT FILE LINH HOẠT ---
def import_module_from_file(module_name, file_path):
    if not os.path.exists(file_path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import trực tiếp file v4 mà bạn vừa tối ưu xong
route_optimizer_module = import_module_from_file(
    "route_optimizer_v4",
    "route_optimizer_v4.py",
)


# Database BKS (Best Known Solutions)
def get_instance_info(filepath):
    """
    Trích xuất thông tin N, K từ tên file
    Và ĐỌC TRỰC TIẾP file .sol để lấy Best Known Solution (BKS)
    """
    filename = os.path.basename(filepath)
    name = filename.replace(".vrp", "")

    # Parse N and K từ filename
    match = re.search(r"n(\d+)", name)
    n = int(match.group(1)) if match else 0

    match = re.search(r"k(\d+)", name)
    k = int(match.group(1)) if match else 0

    # --- TỰ ĐỘNG ĐỌC FILE .SOL ĐỂ LẤY BKS ---
    bks = 0
    sol_filepath = filepath.replace(".vrp", ".sol")

    if os.path.exists(sol_filepath):
        with open(sol_filepath, "r") as f:
            for line in f:
                # File chuẩn CVRPLIB luôn có định dạng ví dụ: "Cost 1496"
                if line.startswith("Cost"):
                    try:
                        bks = int(line.split()[1])
                    except ValueError:
                        pass
                    break

    # Cảnh báo nếu file .sol bị thiếu
    if bks == 0:
        print(f"⚠️  Không tìm thấy BKS (Cost) trong file {name}.sol!")

    return name, n, k, bks


def run_solver_wrapper(filepath):
    """
    Hàm gọi solver và trả về kết quả.
    """
    if not route_optimizer_module:
        raise Exception("File 'route_optimizer_v4.py' không tồn tại hoặc lỗi import!")

    # Tắt chế độ in chi tiết (verbose=False) để màn hình benchmark sạch sẽ
    return route_optimizer_module.solve_with_clarke_wright_and_optimize(
        filepath, verbose=False
    )


def main():
    print("\n" + "=" * 70)
    print("BENCHMARK RUNNER FOR ROUTE OPTIMIZER V4 - DATASET B")
    print("Runs per instance: 10")
    print("=" * 70)

    # Chỉ thu thập file từ folder B
    instances_list = []
    folder_path = os.path.join("instances", "B")

    if not os.path.isdir(folder_path):
        print(f"⚠️  Folder '{folder_path}' không tồn tại. Vui lòng kiểm tra lại!")
        return

    vrp_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".vrp")])
    print(f"\n📂 Đã tìm thấy {len(vrp_files)} instances trong thư mục B")

    for vrp_file in vrp_files:
        filepath = os.path.join(folder_path, vrp_file)
        name, n, k, bks = get_instance_info(filepath)
        instances_list.append(
            {"filepath": filepath, "name": name, "n": n, "k": k, "bks": bks}
        )

    # File CSV đầu ra mang tên riêng cho bộ B
    result_file = f"benchmark_results_B_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"💾 Dữ liệu sẽ được lưu tại: {result_file}")

    file_exists = os.path.exists(result_file)
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

    with open(result_file, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        print("\n" + "▒" * 70)
        print("▶️  BẮT ĐẦU CHẠY BENCHMARK...")
        print("▒" * 70)

        for idx, data in enumerate(instances_list):
            print(
                f"\n[{idx+1}/{len(instances_list)}] {data['name']} (N={data['n']}, K={data['k']})"
            )

            filepath = data["filepath"]
            costs = []
            times = []
            params_list = []

            n_runs = 10

            for run in range(n_runs):
                try:
                    start_time = time.time()
                    routes, cost, params = run_solver_wrapper(filepath)
                    elapsed = time.time() - start_time

                    if elapsed > TIMEOUT_LIMIT:
                        print(f"  ⏱️  TIMEOUT after {elapsed:.1f}s")
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
                last_params = (
                    params_list[best_idx] if best_idx < len(params_list) else {}
                )

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
                print(f"  ❌ ALL RUNS FAILED")

    print("\n" + "▒" * 70)
    print(f"✅ BENCHMARK COMPLETED! Results saved to: {result_file}")
    print("▒" * 70)


if __name__ == "__main__":
    main()
