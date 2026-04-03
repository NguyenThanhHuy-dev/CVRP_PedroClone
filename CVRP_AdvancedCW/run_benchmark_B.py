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
from csv_upsert import load_csv, save_csv, upsert_row, FIELDNAMES

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

INSTANCE_DIR = os.path.join("instances", "B")
RESULTS_DIR  = "results"

# Mỗi method có file CSV riêng: benchmark_B_gurobi.csv, ...
RESULT_FILE = os.path.join(RESULTS_DIR, f"benchmark_B_{METHOD}.csv")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# =====================================================================
# THAM SỐ
# =====================================================================
MAX_ITERATIONS = 150
USE_TEST_CONFIG = True
TEST_CONFIG = {
    "max_single_size":    11,
    "single_timeout":     60.0,
    "max_pairwise_size":  10,
    "pairwise_timeout":  500.0,
    "n_closest_pairs":     12,
    "patience":           20,
    "global_timeout":   1200.0,
}
BASE_CONFIGS = {
    "gurobi": {"single_timeout":  5.0, "patience": 20},
    "cplex":  {"single_timeout":  5.0, "patience": 20},
    "pysat":  {"single_timeout": 10.0, "patience": 20},
}

def build_dynamic_config(n: int, k: int) -> dict:
    base = BASE_CONFIGS[METHOD]
    avg  = n / k if k > 0 else float(n)

    if METHOD == "gurobi":
        max_single   = max(40, int(avg * 1.5))
        max_pairwise = max(25, int(avg * 2.5))
        pair_timeout = min(40.0, max(15.0, avg * 1.0))
        n_pairs      = 999 # <-- GUROBI QUÉT FULL CẶP
    elif METHOD == "cplex":
        max_single   = max(30, int(avg * 1.5))
        max_pairwise = max(20, int(avg * 2.5))
        pair_timeout = min(40.0, max(15.0, avg * 1.0))
        n_pairs      = 999 # <-- CPLEX QUÉT FULL CẶP
    else:  # pysat — kích thước bài toán con bị giới hạn cứng
        max_single   = min(11, max(8,  int(avg * 1.0)))
        max_pairwise = min(12, max(8,  int(avg * 1.5)))
        pair_timeout = min(20.0, max(15.0, avg * 1.5))
        n_pairs      = min(k, 5) # <-- PYSAT BỊ GIỚI HẠN CHẶT ĐỂ BẢO VỆ CPU

    return {
        "max_single_size":   max_single,
        "single_timeout":    base["single_timeout"],
        "max_pairwise_size": max_pairwise,
        "pairwise_timeout":  pair_timeout,
        "n_closest_pairs":   n_pairs, # <-- TRUYỀN BIẾN ĐỘNG VÀO ĐÂY
        "patience":          base["patience"],
        "global_timeout":    1800.0,
    }
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
    if USE_TEST_CONFIG:
        print("CHẾ ĐỘ: TEST CONFIG (cố định, dễ so sánh)")
        for k_cfg, v_cfg in TEST_CONFIG.items():
            print(f"  {k_cfg}: {v_cfg}")
    else:
        print("CHẾ ĐỘ: Dynamic Config (tính per-instance theo N và K)")
        print(f"  single_timeout : {BASE_CONFIGS[METHOD]['single_timeout']}s")
        print(f"  patience       : {BASE_CONFIGS[METHOD]['patience']}")
    print(f"  max_iterations : {MAX_ITERATIONS}")
    print(f"KẾT QUẢ LƯU VÀO: {RESULT_FILE}")
    print("=" * 70)

    # --- Load toàn bộ CSV hiện tại vào memory (upsert mode) ---
    csv_data = load_csv(RESULT_FILE)
    print(f"[INFO] Đã load {len(csv_data)} kết quả cũ từ {RESULT_FILE}")

    for idx, filename in enumerate(b_files):
            name, n, k = get_instance_info(filename)
            filepath     = os.path.join(INSTANCE_DIR, filename)
            sol_filepath = os.path.join(INSTANCE_DIR, filename.replace(".vrp", ".sol"))

            bks     = get_bks_from_sol(sol_filepath)
            bks_str = str(bks) if bks > 0 else "N/A"

            # Thông báo nếu instance này đã từng chạy trước đó
            existing_key = (name, METHOD)
            if existing_key in csv_data:
                old_runs = csv_data[existing_key].get("Runs", "0")
                old_best = csv_data[existing_key].get("Best_Cost", "?")
                print(f"\n[{idx+1:02d}/{len(b_files)}] {name}  (N={n}, K={k}, BKS={bks_str})"
                      f"  ← đã có {old_runs} lần chạy, best={old_best}")
            else:
                print(f"\n[{idx+1:02d}/{len(b_files)}] {name}  (N={n}, K={k}, BKS={bks_str})"
                      f"  ← lần đầu chạy")

            # --- Chọn config ---
            if USE_TEST_CONFIG:
                cfg = TEST_CONFIG.copy()
            else:
                cfg = build_dynamic_config(n, k)
                
            # [ÉP ĐÈ AN TOÀN]: Bất chấp dùng cấu hình Test hay Dynamic, 
            # Gurobi và CPLEX luôn được quyền quét toàn bộ các cặp tuyến.
            if METHOD in ("gurobi", "cplex"):
                cfg["n_closest_pairs"] = 999

            print(f"  [Config] single={cfg['max_single_size']} pair={cfg['max_pairwise_size']}"
                  f" stimeout={cfg['single_timeout']:.0f}s ptimeout={cfg['pairwise_timeout']:.0f}s"
                  f" pairs={cfg['n_closest_pairs']} patience={cfg['patience']}")

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
                      f" | Pairwise cải thiện: {stats.get('pairwise_imp_count', 0)} lần"
                      f" | S-timeout: {stats.get('single_timeouts', 0)}"
                      f" | P-timeout: {stats.get('pairwise_timeouts', 0)}")

                # Upsert vào dict (không tạo row mới nếu đã tồn tại)
                csv_data = upsert_row(
                    data=csv_data,
                    instance=name,
                    n=n, k=k,
                    bks=bks,
                    new_cost=opt_cost,
                    elapsed=elapsed,
                    cfg=cfg,
                    stats=stats,
                    method=METHOD,
                    max_iterations=MAX_ITERATIONS,
                )

                # Ghi ngay sau mỗi instance để không mất dữ liệu nếu crash
                save_csv(RESULT_FILE, csv_data)

                # In tóm tắt tổng hợp sau upsert
                row = csv_data[(name, METHOD)]
                print(f"  -> Runs={row['Runs']} | Best={row['Best_Cost']}"
                      f" | Avg={row['Avg_Cost']}")

            except Exception as e:
                restore_logging_to_console(suppressed)
                print(f"  -> LỖI: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"HOÀN THÀNH. Kết quả: {RESULT_FILE}")


if __name__ == "__main__":
    run_b_benchmark()