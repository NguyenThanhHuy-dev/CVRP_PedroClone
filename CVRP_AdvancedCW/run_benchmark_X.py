#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Benchmark Runner for Augerat X-Set Instances
============================================
Sử dụng: python run_benchmark_X.py [METHOD]
  METHOD: gurobi | cplex | pysat (mặc định: pysat)

Ví dụ:
  python run_benchmark_X.py gurobi
  python run_benchmark_X.py cplex
  python run_benchmark_X.py pysat
"""

import os
import sys
import time
import csv
import re
import logging

VALID_METHODS = ("gurobi", "cplex", "pysat")

def parse_method() -> str:
    if len(sys.argv) < 2:
        print(f"[INFO] Không có METHOD, dùng mặc định: pysat")
        print(f"       Cách dùng: python {sys.argv[0]} [{'|'.join(VALID_METHODS)}]")
        return "pysat"

    method = sys.argv[1].strip().lower()
    if method not in VALID_METHODS:
        print(f"[LỖI] METHOD không hợp lệ: '{method}'")
        print(f"       Chọn một trong: {', '.join(VALID_METHODS)}")
        sys.exit(1)

    return method

METHOD = parse_method()

print(f"[INFO] Đang nạp solver: {METHOD.upper()}...")

if METHOD == "gurobi":
    try:
        from advanced_optimizer_gurobi import solve_advanced
    except ImportError as e:
        print(f"[LỖI] Không import được advanced_optimizer_gurobi: {e}")
        sys.exit(1)

elif METHOD == "cplex":
    try:
        from advanced_optimizer_cplex import solve_advanced
    except ImportError as e:
        print(f"[LỖI] Không import được advanced_optimizer_cplex: {e}")
        sys.exit(1)

else:  # pysat
    try:
        from advanced_optimizer_pysat import solve_advanced
    except ImportError as e:
        print(f"[LỖI] Không import được advanced_optimizer_pysat: {e}")
        sys.exit(1)

print(f"[INFO] Solver {METHOD.upper()} đã sẵn sàng.")

INSTANCE_DIR = os.path.join("instances", "X")
RESULTS_DIR  = "results"
RESULT_FILE  = os.path.join(RESULTS_DIR, f"benchmark_X_{METHOD}.csv")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

BASE_CONFIGS = {
    "gurobi": {"single_timeout": 5.0, "patience": 10},
    "cplex":  {"single_timeout": 5.0, "patience": 10},
    "pysat":  {"single_timeout": 5.0, "patience": 10},
}

MAX_ITERATIONS = 80


def build_dynamic_config(n: int, k: int) -> dict:
    """
    Tính Dynamic Config dựa trên N (số khách) và K (số xe) của instance.

    avg_route_len  = N / K
    max_single     = max(floor, avg * 1.5)   -- đủ rộng cho tuyến bị phình
    max_pairwise   = max(floor, avg * 2.5)   -- chứa được ~2.5 tuyến trung bình
    pair_timeout   = clamp(avg * 1.0, 15s, 40s)
    n_pairs        = min(k, 15)
    """
    base = BASE_CONFIGS[METHOD]
    avg  = n / k if k > 0 else float(n)

    if METHOD == "gurobi":
        max_single   = max(40, int(avg * 1.5))
        max_pairwise = max(25, int(avg * 2.5))
        pair_timeout = min(40.0, max(15.0, avg * 1.0))
    elif METHOD == "cplex":
        max_single   = max(30, int(avg * 1.5))
        max_pairwise = max(20, int(avg * 2.5))
        pair_timeout = min(40.0, max(15.0, avg * 1.0))
    else:  # pysat — solver chậm hơn, giữ ngưỡng thấp
        max_single   = max(11, int(avg * 1.2))
        max_pairwise = max(10, int(avg * 2.0))
        pair_timeout = min(20.0, max(8.0, avg * 0.8))

    return {
        "max_single_size":   max_single,
        "single_timeout":    base["single_timeout"],
        "max_pairwise_size": max_pairwise,
        "pairwise_timeout":  pair_timeout,
        "n_closest_pairs":   min(k, 15),
        "patience":          base["patience"],
    }

def get_bks_from_sol(sol_filepath: str) -> int:
    if not os.path.exists(sol_filepath):
        return 0
    try:
        with open(sol_filepath, 'r') as f:
            for line in f:
                if line.strip().lower().startswith('cost'):
                    parts = line.replace(':', '').split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        pass
    return 0


def get_instance_info(filename: str):
    name    = filename.replace(".vrp", "")
    n_match = re.search(r'-n(\d+)', name)
    k_match = re.search(r'-k(\d+)', name)
    n = int(n_match.group(1)) if n_match else 0
    k = int(k_match.group(1)) if k_match else 0
    return name, n, k


def suppress_logging_to_console():
    root_logger = logging.getLogger()
    suppressed = []
    for h in root_logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.CRITICAL)
            suppressed.append(h)
    return suppressed


def restore_logging_to_console(suppressed_handlers):
    for h in suppressed_handlers:
        h.setLevel(logging.INFO)

def get_completed_instances(csv_path):
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row['Instance'])
    return done

def run_x_benchmark():
    if not os.path.exists(INSTANCE_DIR):
        print(f"[LỖI] Không tìm thấy thư mục '{INSTANCE_DIR}'")
        print(f"       Hãy tạo thư mục 'instances/X' và copy file .vrp + .sol vào đó.")
        return

    x_files = sorted(
        [f for f in os.listdir(INSTANCE_DIR) if f.startswith("X-") and f.endswith(".vrp")],
        key=lambda f: get_instance_info(f)[1]
    )

    completed = get_completed_instances(RESULT_FILE)

    x_files = [
        f for f in x_files
        if get_instance_info(f)[0] not in completed
    ]
    
    print(f"[INFO] Đã chạy: {len(completed)} instances")
    print(f"[INFO] Còn lại: {len(x_files)} instances sẽ chạy")

    if not x_files:
        print(f"[INFO] Không tìm thấy file .vrp nào trong '{INSTANCE_DIR}'.")
        return

    print("\n" + "=" * 70)
    print(f"BENCHMARK X-SET  |  METHOD: {METHOD.upper()}  |  {len(x_files)} instances")
    print("CHẾ ĐỘ: Dynamic Config (tính per-instance theo N và K)")
    print(f"  single_timeout : {BASE_CONFIGS[METHOD]['single_timeout']}s (cố định)")
    print(f"  patience       : {BASE_CONFIGS[METHOD]['patience']} (cố định)")
    print(f"  max_iterations : {MAX_ITERATIONS}")
    print(f"  Các tham số còn lại được tính động dựa trên N/K mỗi instance")
    print(f"KẾT QUẢ LƯU VÀO: {RESULT_FILE}")
    print("=" * 70)

    fieldnames = [
        'Instance', 'N', 'K', 'BKS', 'Cost_Found', 'Gap(%)', 'Time(s)',
        'Max_Single', 'Max_Pair', 'Num_Pairs', 'Patience', 'Max_Iter',
        'Single_Imp_Count', 'Pair_Imp_Count',
        'Method',
    ]

    file_exists = os.path.exists(RESULT_FILE)
    with open(RESULT_FILE, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for idx, filename in enumerate(x_files):
            name, n, k = get_instance_info(filename)
            filepath     = os.path.join(INSTANCE_DIR, filename)
            sol_filepath = os.path.join(INSTANCE_DIR, filename.replace(".vrp", ".sol"))

            bks     = get_bks_from_sol(sol_filepath)
            bks_str = str(bks) if bks > 0 else "N/A"

            print(f"\n[{idx+1:02d}/{len(x_files)}] {name}  (N={n}, K={k}, BKS={bks_str})")

            # --- Tính Dynamic Config cho instance này ---
            cfg = build_dynamic_config(n, k)
            print(f"  [Config] single={cfg['max_single_size']} pair={cfg['max_pairwise_size']}"
                  f" ptimeout={cfg['pairwise_timeout']:.0f}s pairs={cfg['n_closest_pairs']}")

            start_t    = time.time()
            suppressed = []
            try:
                suppressed = suppress_logging_to_console()

                opt_routes, opt_cost, stats = solve_advanced(
                    filepath,
                    config=cfg,
                    max_iterations=MAX_ITERATIONS,
                )

                restore_logging_to_console(suppressed)
                elapsed = time.time() - start_t

                gap = ((opt_cost - bks) / bks) * 100 if bks > 0 else 0.0

                print(f"  -> Cost: {opt_cost} | Gap: {gap:+.2f}% | Time: {elapsed:.1f}s")
                print(f"  -> Single cải thiện: {stats.get('single_imp_count', 0)} lần"
                      f" | Pairwise cải thiện: {stats.get('pairwise_imp_count', 0)} lần")

                writer.writerow({
                    'Instance':         name,
                    'N':                n,
                    'K':                k,
                    'BKS':              bks_str,
                    'Cost_Found':       opt_cost,
                    'Gap(%)':           f"{gap:.2f}" if bks > 0 else "N/A",
                    'Time(s)':          f"{elapsed:.2f}",
                    'Max_Single':       cfg['max_single_size'],
                    'Max_Pair':         cfg['max_pairwise_size'],
                    'Num_Pairs':        cfg['n_closest_pairs'],
                    'Patience':         cfg['patience'],
                    'Max_Iter':         MAX_ITERATIONS,
                    'Single_Imp_Count': stats.get('single_imp_count', 0),
                    'Pair_Imp_Count':   stats.get('pairwise_imp_count', 0),
                    'Method':           METHOD,
                })
                csv_file.flush()

            except Exception as e:
                restore_logging_to_console(suppressed)
                print(f"  -> LỖI: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"HOÀN THÀNH. Kết quả: {RESULT_FILE}")


if __name__ == "__main__":
    run_x_benchmark()