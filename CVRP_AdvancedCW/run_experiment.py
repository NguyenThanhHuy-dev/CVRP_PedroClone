#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Thực nghiệm tìm thông số tối ưu cho 1 instance
===============================================
Mục đích: Chạy nhiều lần với các bộ thông số khác nhau trên
          instance B-n31-k5, mỗi lần chạy INSERT một row mới
          vào CSV (không upsert) để so sánh tất cả các lần chạy.

Cách dùng:
  python run_experiment.py                    # chạy tất cả config trong EXPERIMENT_CONFIGS
  python run_experiment.py B-n51-k7.vrp       # chạy instance khác nếu muốn
"""

import os
import sys
import time
import csv
import re
import logging

# =====================================================================
# IMPORT SOLVER
# =====================================================================
try:
    from advanced_optimizer_pysat import solve_advanced
except ImportError as e:
    print(f"[LỖI] Không import được advanced_optimizer_pysat: {e}")
    sys.exit(1)

# =====================================================================
# CẤU HÌNH
# =====================================================================
INSTANCE_DIR  = os.path.join("instances", "B")
RESULTS_DIR   = "results"
RESULT_FILE   = os.path.join(RESULTS_DIR, "experiment_tuning.csv")
DEFAULT_INSTANCE = "B-n31-k5.vrp"

# MAX_ITERATIONS chung cho tất cả các lần chạy
MAX_ITERATIONS = 150

# =====================================================================
# DANH SÁCH CÁC BỘ THÔNG SỐ CẦN THỬ
# Thêm/bớt dict vào list này để thêm/bớt thực nghiệm.
# Mỗi dict sẽ thành 1 row trong CSV.
# =====================================================================
EXPERIMENT_CONFIGS = [
    # --- Baseline: config cũ (max_single=14, timeout ngắn) ---
    {
        "_label": "baseline_old",
        "max_single_size":   11,
        "single_timeout":     60.0,
        "max_pairwise_size":  12,
        "pairwise_timeout":  600.0,
        "n_closest_pairs":   12,
        "patience":          20,
        "global_timeout":  1200.0,
    },
    # --- Config mới đề xuất (max_single nhỏ hơn, timeout dài hơn) ---
    # {
    #     "_label": "proposed_v1",
    #     "max_single_size":   11,
    #     "single_timeout":    10.0,
    #     "max_pairwise_size":  9,
    #     "pairwise_timeout":  20.0,
    #     "n_closest_pairs":    5,
    #     "patience":          10,
    #     "global_timeout":  1800.0,
    # },
    # # --- Thử PBEnc (max_single giảm, pair timeout rất dài) ---
    # {
    #     "_label": "pbenc_long_pair",
    #     "max_single_size":   11,
    #     "single_timeout":    20.0,
    #     "max_pairwise_size":  9,
    #     "pairwise_timeout": 1000.0,
    #     "n_closest_pairs":   15,
    #     "patience":          20,
    #     "global_timeout":  1200.0,
    # },
    # # --- Thử tăng single lên 12, timeout vừa phải ---
    # {
    #     "_label": "single12_balanced",
    #     "max_single_size":   12,
    #     "single_timeout":    15.0,
    #     "max_pairwise_size":  9,
    #     "pairwise_timeout":  30.0,
    #     "n_closest_pairs":    5,
    #     "patience":          10,
    #     "global_timeout":  1800.0,
    # },
    # # --- Chỉ dùng Single MaxSAT (pair size = 0 → bỏ qua pairwise) ---
    # {
    #     "_label": "single_only",
    #     "max_single_size":   11,
    #     "single_timeout":    20.0,
    #     "max_pairwise_size":  9,
    #     "pairwise_timeout":  10.0,
    #     "n_closest_pairs":    10,
    #     "patience":          10,
    #     "global_timeout":  1200.0,
    # },
]

# =====================================================================
# FIELDNAMES — không có Best/Avg, mỗi row = 1 lần chạy độc lập
# =====================================================================
FIELDNAMES = [
    "Run_ID",           # số thứ tự lần chạy trong file CSV
    "Label",            # tên config để dễ nhận biết
    "Instance",
    "N", "K", "BKS",
    "Cost_Found",
    "Gap(%)",
    "Time(s)",
    "Max_Single", "Single_Timeout",
    "Max_Pair",   "Pair_Timeout",
    "Num_Pairs",
    "Patience",
    "Max_Iter",
    "Global_Timeout",
    "Single_Imp_Count", "Single_Timeouts",
    "Pair_Imp_Count",   "Pair_Timeouts",
    "Solver",
]

# =====================================================================
# HELPERS
# =====================================================================
def get_bks_from_sol(sol_filepath: str) -> int:
    if not os.path.exists(sol_filepath):
        return 0
    try:
        with open(sol_filepath, "r") as f:
            for line in f:
                if line.strip().lower().startswith("cost"):
                    parts = line.replace(":", "").split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        pass
    return 0


def get_instance_info(filename: str):
    name    = filename.replace(".vrp", "")
    n_match = re.search(r"-n(\d+)", name)
    k_match = re.search(r"-k(\d+)", name)
    n = int(n_match.group(1)) if n_match else 0
    k = int(k_match.group(1)) if k_match else 0
    return name, n, k


def next_run_id(result_file: str) -> int:
    """Đọc file CSV hiện tại để lấy Run_ID tiếp theo."""
    if not os.path.exists(result_file):
        return 1
    try:
        with open(result_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if not rows:
                return 1
            return max(int(r.get("Run_ID", 0)) for r in rows) + 1
    except Exception:
        return 1


def suppress_logging():
    root_logger = logging.getLogger()
    suppressed = []
    for h in root_logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.CRITICAL)
            suppressed.append(h)
    return suppressed


def restore_logging(suppressed):
    for h in suppressed:
        h.setLevel(logging.INFO)


# =====================================================================
# HÀM CHÍNH
# =====================================================================
def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Chọn instance ---
    filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INSTANCE
    if not filename.endswith(".vrp"):
        filename += ".vrp"

    filepath     = os.path.join(INSTANCE_DIR, filename)
    sol_filepath = os.path.join(INSTANCE_DIR, filename.replace(".vrp", ".sol"))

    if not os.path.exists(filepath):
        print(f"[LỖI] Không tìm thấy instance: {filepath}")
        sys.exit(1)

    name, n, k = get_instance_info(filename)
    bks         = get_bks_from_sol(sol_filepath)
    bks_str     = str(bks) if bks > 0 else "N/A"

    print("\n" + "=" * 70)
    print(f"THỰC NGHIỆM TÌM THÔNG SỐ TỐI ỨU")
    print(f"Instance  : {name}  (N={n}, K={k}, BKS={bks_str})")
    print(f"Số config : {len(EXPERIMENT_CONFIGS)}")
    print(f"Kết quả   : {RESULT_FILE}")
    print("=" * 70)

    # --- Chuẩn bị CSV (append mode, tạo header nếu chưa có) ---
    file_exists = os.path.exists(RESULT_FILE)
    run_id      = next_run_id(RESULT_FILE)

    with open(RESULT_FILE, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for cfg_idx, raw_cfg in enumerate(EXPERIMENT_CONFIGS):
            label = raw_cfg.get("_label", f"config_{cfg_idx+1}")
            # Tách _label ra khỏi config thực sự truyền vào solver
            cfg = {k: v for k, v in raw_cfg.items() if not k.startswith("_")}

            print(f"\n[{cfg_idx+1}/{len(EXPERIMENT_CONFIGS)}] Label: {label}")
            print(f"  single={cfg['max_single_size']} stimeout={cfg['single_timeout']:.0f}s"
                  f" | pair={cfg['max_pairwise_size']} ptimeout={cfg['pairwise_timeout']:.0f}s"
                  f" | pairs={cfg['n_closest_pairs']} patience={cfg['patience']}"
                  f" | global={cfg['global_timeout']:.0f}s")

            start_t    = time.time()
            suppressed = []
            try:
                suppressed = suppress_logging()

                opt_routes, opt_cost, stats = solve_advanced(
                    filepath,
                    config=cfg,
                    max_iterations=MAX_ITERATIONS,
                )

                restore_logging(suppressed)
                elapsed = time.time() - start_t
                gap     = ((opt_cost - bks) / bks * 100) if bks > 0 else 0.0
                gap_str = f"{gap:.2f}" if bks > 0 else "N/A"

                print(f"  -> Cost: {opt_cost} | Gap: {gap:+.2f}% | Time: {elapsed:.1f}s")
                print(f"  -> Single imp: {stats.get('single_imp_count', 0)}"
                      f" (timeout={stats.get('single_timeouts', 0)})"
                      f" | Pair imp: {stats.get('pairwise_imp_count', 0)}"
                      f" (timeout={stats.get('pairwise_timeouts', 0)})")

                writer.writerow({
                    "Run_ID":            run_id,
                    "Label":             label,
                    "Instance":          name,
                    "N":                 n,
                    "K":                 k,
                    "BKS":               bks_str,
                    "Cost_Found":        opt_cost,
                    "Gap(%)":            gap_str,
                    "Time(s)":           f"{elapsed:.2f}",
                    "Max_Single":        cfg["max_single_size"],
                    "Single_Timeout":    cfg["single_timeout"],
                    "Max_Pair":          cfg["max_pairwise_size"],
                    "Pair_Timeout":      cfg["pairwise_timeout"],
                    "Num_Pairs":         cfg["n_closest_pairs"],
                    "Patience":          cfg["patience"],
                    "Max_Iter":          MAX_ITERATIONS,
                    "Global_Timeout":    cfg["global_timeout"],
                    "Single_Imp_Count":  stats.get("single_imp_count",   0),
                    "Single_Timeouts":   stats.get("single_timeouts",    0),
                    "Pair_Imp_Count":    stats.get("pairwise_imp_count", 0),
                    "Pair_Timeouts":     stats.get("pairwise_timeouts",  0),
                    "Solver":            stats.get("solver_name", "MaxSAT-RC2"),
                })
                csv_file.flush()
                run_id += 1

            except Exception as e:
                restore_logging(suppressed)
                print(f"  -> LỖI: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"HOÀN THÀNH. Kết quả: {RESULT_FILE}")

    # --- In bảng tóm tắt cuối ---
    print("\nTÓM TẮT KẾT QUẢ:")
    print(f"  {'Label':<25} {'Cost':>8} {'Gap(%)':>8} {'Time(s)':>9}"
          f" {'S_imp':>6} {'S_to':>5} {'P_imp':>6} {'P_to':>5}")
    print("  " + "-" * 75)
    try:
        with open(RESULT_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Instance") == name:
                    print(f"  {row['Label']:<25} {row['Cost_Found']:>8}"
                          f" {row['Gap(%)']:>8} {row['Time(s)']:>9}"
                          f" {row['Single_Imp_Count']:>6} {row['Single_Timeouts']:>5}"
                          f" {row['Pair_Imp_Count']:>6} {row['Pair_Timeouts']:>5}")
    except Exception:
        pass


if __name__ == "__main__":
    run_experiment()