#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Benchmark Runner for Augerat B-Set Instances
=============================================
Sử dụng: python run_benchmark_B.py [METHOD]
  METHOD: gurobi | cplex | pysat (mặc định: pysat)

Ví dụ:
  python run_benchmark_B.py gurobi
  python run_benchmark_B.py cplex
  python run_benchmark_B.py pysat
"""

import os
import sys
import time
import csv
import re
import logging

# =====================================================================
# PHÂN TÍCH THAM SỐ DÒNG LỆNH
# =====================================================================
VALID_METHODS = ("gurobi", "cplex", "pysat")

def parse_method() -> str:
    """Đọc METHOD từ argv, validate và trả về."""
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

# =====================================================================
# IMPORT SOLVER TƯƠNG ỨNG
# =====================================================================
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

# =====================================================================
# CẤU HÌNH THƯ MỤC & FILE KẾT QUẢ (tách riêng theo method)
# =====================================================================
INSTANCE_DIR = os.path.join("instances", "B")
RESULTS_DIR  = "results"

# Mỗi method có file CSV riêng: benchmark_B_gurobi.csv, ...
RESULT_FILE = os.path.join(RESULTS_DIR, f"benchmark_B_{METHOD}.csv")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# =====================================================================
# SIÊU THAM SỐ - Mỗi method có config tối ưu riêng
# =====================================================================
CONFIGS = {
    "gurobi": {
        "max_single_size":   40,
        "single_timeout":     2.0,
        "max_pairwise_size": 25,
        "pairwise_timeout":  10.0,
        "n_closest_pairs":    5,
        "patience":           5,
    },
    "cplex": {
        "max_single_size":   30,
        "single_timeout":     5.0,
        "max_pairwise_size": 20,
        "pairwise_timeout":  10.0,
        "n_closest_pairs":    5,
        "patience":           5,
    },
    "pysat": {
        "max_single_size":   11,
        "single_timeout":     5.0,
        "max_pairwise_size": 10,
        "pairwise_timeout":   8.0,
        "n_closest_pairs":    3,
        "patience":           5,
    },
}

TUNING_CONFIG = CONFIGS[METHOD]
MAX_ITERATIONS = 50

# =====================================================================
# HELPERS
# =====================================================================
def get_bks_from_sol(sol_filepath: str) -> int:
    """Đọc file .sol để lấy Best Known Solution cost."""
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
    """Trích xuất tên, N, K từ tên file VRP."""
    name    = filename.replace(".vrp", "")
    n_match = re.search(r'-n(\d+)', name)
    k_match = re.search(r'-k(\d+)', name)
    n = int(n_match.group(1)) if n_match else 0
    k = int(k_match.group(1)) if k_match else 0
    return name, n, k


def suppress_logging_to_console():
    """Tắt StreamHandler của root logger (giữ FileHandler)."""
    root_logger = logging.getLogger()
    suppressed = []
    for h in root_logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.CRITICAL)
            suppressed.append(h)
    return suppressed


def restore_logging_to_console(suppressed_handlers):
    """Khôi phục level INFO cho các handler đã tắt."""
    for h in suppressed_handlers:
        h.setLevel(logging.INFO)

# =====================================================================
# HÀM CHÍNH
# =====================================================================
def run_b_benchmark():
    if not os.path.exists(INSTANCE_DIR):
        print(f"[LỖI] Không tìm thấy thư mục '{INSTANCE_DIR}'")
        print(f"       Hãy tạo thư mục 'instances/B' và copy file .vrp + .sol vào đó.")
        return

    b_files = sorted(
        [f for f in os.listdir(INSTANCE_DIR) if f.startswith("B-") and f.endswith(".vrp")],
        key=lambda f: get_instance_info(f)[1]   # sắp xếp theo N
    )

    if not b_files:
        print(f"[INFO] Không tìm thấy file .vrp nào trong '{INSTANCE_DIR}'.")
        return

    # --- In tiêu đề ---
    print("\n" + "=" * 70)
    print(f"BENCHMARK B-SET  |  METHOD: {METHOD.upper()}  |  {len(b_files)} instances")
    print("CẤU HÌNH THAM SỐ:")
    for k, v in TUNING_CONFIG.items():
        print(f"  {k}: {v}")
    print(f"  max_iterations: {MAX_ITERATIONS}")
    print(f"KẾT QUẢ LƯU VÀO: {RESULT_FILE}")
    print("=" * 70)

    # --- Chuẩn bị CSV ---
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

        for idx, filename in enumerate(b_files):
            name, n, k = get_instance_info(filename)
            filepath     = os.path.join(INSTANCE_DIR, filename)
            sol_filepath = os.path.join(INSTANCE_DIR, filename.replace(".vrp", ".sol"))

            bks     = get_bks_from_sol(sol_filepath)
            bks_str = str(bks) if bks > 0 else "N/A"

            print(f"\n[{idx+1:02d}/{len(b_files)}] {name}  (N={n}, K={k}, BKS={bks_str})")

            start_t    = time.time()
            suppressed = []
            try:
                suppressed = suppress_logging_to_console()

                opt_routes, opt_cost, stats = solve_advanced(
                    filepath,
                    config=TUNING_CONFIG,
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
                    'Max_Single':       TUNING_CONFIG['max_single_size'],
                    'Max_Pair':         TUNING_CONFIG['max_pairwise_size'],
                    'Num_Pairs':        TUNING_CONFIG['n_closest_pairs'],
                    'Patience':         TUNING_CONFIG['patience'],
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
    run_b_benchmark()