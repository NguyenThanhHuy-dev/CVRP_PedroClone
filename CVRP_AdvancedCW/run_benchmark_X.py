#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Benchmark Runner for Augerat X-Set Instances
============================================
Chạy thực nghiệm riêng cho bộ dữ liệu X-Set, 
đọc BKS trực tiếp từ file .sol và lưu kết quả ra CSV.
"""

import os
import time
import csv
import re

# Import advanced optimizer của bạn
try:
    from advanced_optimizer import solve_advanced
except ImportError:
    print("Lỗi: Không tìm thấy 'advanced_optimizer.py'. Đảm bảo nó nằm cùng thư mục.")
    exit(1)

# Trỏ tới thư mục chứa bộ X
INSTANCE_DIR = os.path.join("instances", "X")
RESULT_FILE = "benchmark_X_results.csv"

def get_bks_from_sol(sol_filepath):
    """Đọc file .sol để lấy Xest Known Solution (Cost)."""
    if not os.path.exists(sol_filepath):
        return 0
    try:
        with open(sol_filepath, 'r') as f:
            for line in f:
                # Tìm dòng bắt đầu bằng "Cost" (không phân biệt hoa thường)
                if line.strip().lower().startswith('cost'):
                    # Xử lý cả trường hợp "Cost 672" hoặc "Cost: 672"
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

def run_b_benchmark():
    if not os.path.exists(INSTANCE_DIR):
        print(f"Lỗi: Không tìm thấy thư mục '{INSTANCE_DIR}'")
        print(f"Vui lòng tạo thư mục 'instances/X' và copy các file .vrp, .sol vào đó.")
        return

    # Chỉ lấy các file .vrp (bắt đầu bằng X-)
    b_files = [f for f in os.listdir(INSTANCE_DIR) if f.startswith("X-") and f.endswith(".vrp")]
    
    # Sắp xếp theo số lượng Node (N)
    b_files.sort(key=lambda f: get_instance_info(f)[1])

    if not b_files:
        print(f"Không tìm thấy instance bộ X nào trong thư mục '{INSTANCE_DIR}'.")
        return

    print("\n" + "="*60)
    print(f"BENCHMARK X-SET STARTING ({len(b_files)} files found)")
    print("="*60)

    # Chuẩn bị file CSV
    file_exists = os.path.exists(RESULT_FILE)
    with open(RESULT_FILE, mode='a', newline='') as csv_file:
        fieldnames = ['Instance', 'N', 'K', 'BKS', 'Cost_Found', 'Gap(%)', 'Time(s)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        for idx, filename in enumerate(b_files):
            name, n, k = get_instance_info(filename)
            
            filepath = os.path.join(INSTANCE_DIR, filename)
            sol_filepath = os.path.join(INSTANCE_DIR, filename.replace(".vrp", ".sol"))
            
            # Tự động đọc BKS từ file .sol
            bks = get_bks_from_sol(sol_filepath)
            bks_str = str(bks) if bks > 0 else "N/A"
            
            print(f"\n[{idx+1}/{len(b_files)}] Đang chạy {name} (N={n}, K={k}, BKS={bks_str})...")
            
            start_t = time.time()
            try:
                # Chặn print của advanced_optimizer để console gọn gàng hơn
                import sys, io
                old_stdout = sys.stdout
                sys.stdout = io.StringIO() 
                
                # Gọi hàm tối ưu
                opt_routes, opt_cost = solve_advanced(filepath)
                
                # Phục hồi print
                sys.stdout = old_stdout
                
                elapsed = time.time() - start_t
                
                # Tính Gap
                if bks > 0:
                    gap = ((opt_cost - bks) / bks) * 100
                else:
                    gap = 0.0

                print(f" -> Xong! Cost: {opt_cost} | Gap: {gap:.2f}% | Time: {elapsed:.2f}s")
                
                # Ghi ra CSV
                writer.writerow({
                    'Instance': name,
                    'N': n,
                    'K': k,
                    'BKS': bks_str,
                    'Cost_Found': opt_cost,
                    'Gap(%)': f"{gap:.2f}" if bks > 0 else "N/A",
                    'Time(s)': f"{elapsed:.2f}"
                })
                csv_file.flush()
                
            except Exception as e:
                sys.stdout = old_stdout # Đảm bảo phục hồi stdout nếu có lỗi
                print(f" -> LỖI khi chạy {name}: {str(e)}")

    print("\n" + "="*60)
    print(f"ĐÃ HOÀN THÀNH. Kết quả được lưu tại: {RESULT_FILE}")

if __name__ == "__main__":
    run_b_benchmark()