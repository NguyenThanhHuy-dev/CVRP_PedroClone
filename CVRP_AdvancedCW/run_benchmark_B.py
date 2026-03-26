#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Benchmark Runner for Augerat B-Set Instances (With Hyperparameter Tuning)
=========================================================================
Chạy thực nghiệm cho bộ dữ liệu B-Set, cho phép cấu hình tham số động,
đọc BKS từ file .sol và lưu kết quả (kèm tham số) vào thư mục 'results/'.
"""

import os
import sys
import io
import time
import csv
import re
import logging

# Import advanced optimizer của bạn
try:
    from advanced_optimizer import solve_advanced
except ImportError:
    print("Lỗi: Không tìm thấy 'advanced_optimizer.py'. Đảm bảo nó nằm cùng thư mục.")
    exit(1)

# --- CẤU HÌNH THƯ MỤC ---
INSTANCE_DIR = os.path.join("instances", "B")
RESULTS_DIR = "results"
RESULT_FILE = os.path.join(RESULTS_DIR, "benchmark_B_results.csv")

# Tự động tạo thư mục results nếu chưa có
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- CẤU HÌNH SIÊU THAM SỐ (HYPERPARAMETERS) CHO LẦN CHẠY NÀY ---
TUNING_CONFIG = {
    "max_single_size": 40,     # TĂNG MẠNH: Gurobi giải bài toán TSP < 40 đỉnh chưa tới 0.1s.
    "single_timeout": 2.0,     # GIẢM: Gurobi giải rất nhanh, 2s là quá đủ.
    
    "max_pairwise_size": 11,   # GIẢM XUỐNG: Phần này vẫn dùng PySAT. N > 11 sẽ gây bùng nổ tổ hợp, 100% dính timeout vô ích.
    "pairwise_timeout": 10.0,  # GIỮ NGUYÊN: Cho RC2 10s để cố gắng giải quyết.
    
    "n_closest_pairs": 5,      # TỐT: Mở rộng không gian tìm kiếm giữa các tuyến lân cận.
    "patience": 5
}
MAX_ITERATIONS = 50

def get_bks_from_sol(sol_filepath):
    """Đọc file .sol để lấy Best Known Solution (Cost)."""
    if not os.path.exists(sol_filepath):
        return 0
    try:
        with open(sol_filepath, 'r') as f:
            for line in f:
                if line.strip().lower().startswith('cost'):
                    parts = line.replace(':', '').split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception as e:
        print(f"Lỗi khi đọc BKS từ {sol_filepath}: {e}")
    return 0

def get_instance_info(filename):
    """Trích xuất N và K từ tên file."""
    name = filename.replace(".vrp", "")
    n_match = re.search(r'-n(\d+)', name)
    k_match = re.search(r'-k(\d+)', name)
    n = int(n_match.group(1)) if n_match else 0
    k = int(k_match.group(1)) if k_match else 0
    return name, n, k

def suppress_logging_to_console():
    """
    Tắt StreamHandler của root logger để console không bị ngập log khi chạy batch.
    Các FileHandler vẫn hoạt động bình thường.
    Trả về list handler đã tắt để có thể khôi phục sau.
    """
    root_logger = logging.getLogger()
    suppressed = []
    for h in root_logger.handlers:
        # Chỉ tắt StreamHandler thuần (không phải FileHandler)
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.CRITICAL)  # Chỉ in lỗi nghiêm trọng
            suppressed.append(h)
    return suppressed

def restore_logging_to_console(suppressed_handlers):
    """Khôi phục lại level INFO cho các handler đã tắt."""
    for h in suppressed_handlers:
        h.setLevel(logging.INFO)

def run_b_benchmark():
    if not os.path.exists(INSTANCE_DIR):
        print(f"Lỗi: Không tìm thấy thư mục '{INSTANCE_DIR}'")
        print(f"Vui lòng tạo thư mục 'instances/B' và copy các file .vrp, .sol vào đó.")
        return

    b_files = [f for f in os.listdir(INSTANCE_DIR) if f.startswith("B-") and f.endswith(".vrp")]
    b_files.sort(key=lambda f: get_instance_info(f)[1])

    if not b_files:
        print(f"Không tìm thấy instance bộ B nào trong thư mục '{INSTANCE_DIR}'.")
        return

    print("\n" + "="*70)
    print(f"BENCHMARK B-SET STARTING ({len(b_files)} files found)")
    print("CẤU HÌNH THAM SỐ (TUNING PARAMS):")
    for k, v in TUNING_CONFIG.items():
        print(f"  - {k}: {v}")
    print(f"  - max_iterations: {MAX_ITERATIONS}")
    print("="*70)

    file_exists = os.path.exists(RESULT_FILE)
    with open(RESULT_FILE, mode='a', newline='') as csv_file:
        fieldnames = [
            'Instance', 'N', 'K', 'BKS', 'Cost_Found', 'Gap(%)', 'Time(s)',
            'Max_Single', 'Max_Pair', 'Num_Pairs', 'Patience', 'Max_Iter',
            'Single_Imp_Count', 'Pair_Imp_Count'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for idx, filename in enumerate(b_files):
            name, n, k = get_instance_info(filename)
            filepath = os.path.join(INSTANCE_DIR, filename)
            sol_filepath = os.path.join(INSTANCE_DIR, filename.replace(".vrp", ".sol"))

            bks = get_bks_from_sol(sol_filepath)
            bks_str = str(bks) if bks > 0 else "N/A"

            print(f"\n[{idx+1}/{len(b_files)}] Đang chạy {name} (N={n}, K={k}, BKS={bks_str})...")

            start_t = time.time()
            try:
                # FIX: Thay vì redirect sys.stdout (gây xung đột với multiprocessing),
                # chỉ tắt console logging. FileHandler vẫn ghi log đầy đủ.
                suppressed = suppress_logging_to_console()

                opt_routes, opt_cost, stats = solve_advanced(
                    filepath,
                    config=TUNING_CONFIG,
                    max_iterations=MAX_ITERATIONS
                )

                restore_logging_to_console(suppressed)

                elapsed = time.time() - start_t

                if bks > 0:
                    gap = ((opt_cost - bks) / bks) * 100
                else:
                    gap = 0.0

                print(f" -> Xong! Cost: {opt_cost} | Gap: {gap:.2f}% | Time: {elapsed:.2f}s")
                print(f" -> Single Cải thiện: {stats.get('single_imp_count', 0)} lần | Pairwise Cải thiện: {stats.get('pairwise_imp_count', 0)} lần")

                writer.writerow({
                    'Instance': name,
                    'N': n,
                    'K': k,
                    'BKS': bks_str,
                    'Cost_Found': opt_cost,
                    'Gap(%)': f"{gap:.2f}" if bks > 0 else "N/A",
                    'Time(s)': f"{elapsed:.2f}",
                    'Max_Single': TUNING_CONFIG['max_single_size'],
                    'Max_Pair': TUNING_CONFIG['max_pairwise_size'],
                    'Num_Pairs': TUNING_CONFIG['n_closest_pairs'],
                    'Patience': TUNING_CONFIG['patience'],
                    'Max_Iter': MAX_ITERATIONS,
                    'Single_Imp_Count': stats.get('single_imp_count', 0),
                    'Pair_Imp_Count': stats.get('pairwise_imp_count', 0)
                })
                csv_file.flush()

            except Exception as e:
                restore_logging_to_console(suppressed)
                print(f" -> LỖI khi chạy {name}: {str(e)}")
                import traceback
                traceback.print_exc()  # In full traceback để dễ debug

    print("\n" + "="*70)
    print(f"ĐÃ HOÀN THÀNH. Kết quả được lưu tại: {RESULT_FILE}")

if __name__ == "__main__":
    run_b_benchmark()