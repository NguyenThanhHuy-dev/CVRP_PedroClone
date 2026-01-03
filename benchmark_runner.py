#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Universal Benchmark Runner
==========================
Công cụ chạy thực nghiệm so sánh đa phương pháp:
1. Hỗ trợ chọn Method 1 (Cũ/Pipeline) hoặc Method 2 (Mới/Hybrid LNS).
2. Tự động cảnh báo rủi ro treo máy với phương pháp cũ.
3. Lưu kết quả vào CSV chung hoặc riêng tùy chọn.

Author: Adapted for CVRP Project
"""

import os
import sys
import time
import csv
import numpy as np
from datetime import datetime
import re

# --- IMPORT CÁC MODULE THUẬT TOÁN ---
# Đảm bảo bạn đã có 2 file này trong cùng thư mục
try:
    import route_optimizer      # File cũ (Pipeline)
    import route_optimizer_v2   # File mới (Hybrid LNS)
except ImportError as e:
    print(f"Lỗi Import: {e}")
    print("Vui lòng đảm bảo 'route_optimizer.py' và 'route_optimizer_v2.py' nằm cùng thư mục.")
    sys.exit(1)

# Cấu hình mặc định
INSTANCE_DIR = "instances"
DEFAULT_RESULT_FILE = "benchmark_results.csv"

# Database BKS (Best Known Solutions)
BKS_DB = {
    "P-n19-k2": 212, "P-n22-k2": 216, "E-n31-k7": 379,
    "A-n32-k5": 784, "A-n33-k6": 661, "A-n37-k5": 669,
    "B-n39-k5": 549, "F-n45-k4": 724, "P-n45-k5": 510,
    "E-n51-k5": 521, "P-n55-k7": 568, "A-n60-k9": 1354,
    "P-n101-k4": 681, "X-n502-k39": 69226, "X-n1001-k43": 72355
}

def get_instance_info(filename):
    """Trích xuất thông tin N và K từ tên file."""
    name = filename.replace(".vrp", "")
    n_match = re.search(r'-n(\d+)', name)
    k_match = re.search(r'-k(\d+)', name)
    n = int(n_match.group(1)) if n_match else 0
    k = int(k_match.group(1)) if k_match else 0
    bks = BKS_DB.get(name, 0)
    return name, n, k, bks

def run_solver_wrapper(method_choice, filepath):
    """Hàm bọc để gọi đúng phương pháp dựa trên lựa chọn."""
    if method_choice == '1':
        # Gọi phương pháp cũ (route_optimizer.py)
        # Hàm: solve_with_clarke_wright_and_optimize -> returns (routes, cost)
        return route_optimizer.solve_with_clarke_wright_and_optimize(filepath)
    else:
        # Gọi phương pháp mới (route_optimizer_v2.py)
        # Hàm: solve -> returns (routes, cost)
        return route_optimizer_v2.solve(filepath)

def run_benchmark():
    # --- 1. SETUP KHỞI ĐẦU ---
    if not os.path.exists(INSTANCE_DIR):
        print(f"Lỗi: Không tìm thấy thư mục '{INSTANCE_DIR}'")
        return

    files = [f for f in os.listdir(INSTANCE_DIR) if f.endswith(".vrp")]
    files.sort()

    if not files:
        print("Không có file .vrp nào trong thư mục instances.")
        return

    # --- 2. CHỌN PHƯƠNG PHÁP ---
    print("\n" + "="*50)
    print("CHỌN PHƯƠNG PHÁP CHẠY")
    print("="*50)
    print("1. Phương pháp CŨ (route_optimizer.py - Pipeline)")
    print("   -> Ưu điểm: Nhanh với bài nhỏ (N < 50)")
    print("   -> Nhược điểm: Dễ treo/crash với bài lớn (N > 100)")
    print("2. Phương pháp MỚI (route_optimizer_v2.py - Hybrid LNS)")
    print("   -> Ưu điểm: Ổn định, chạy được bài siêu lớn (N=1000)")
    print("-" * 50)
    
    method_choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()
    if method_choice not in ['1', '2']:
        print("Lựa chọn không hợp lệ. Mặc định dùng Method 2 (Mới).")
        method_choice = '2'

    method_name = "Old (Pipeline)" if method_choice == '1' else "New (Hybrid LNS)"

    # --- 3. CHỌN FILE INSTANCE ---
    print("\n" + "="*50)
    print(f"DANH SÁCH INSTANCES ({len(files)} files)")
    print("="*50)
    print(f"{'ID':<5} {'Instance Name':<20} {'Size (N)':<10} {'BKS':<10}")
    print("-" * 50)
    
    instances_data = []
    for idx, f in enumerate(files):
        name, n, k, bks = get_instance_info(f)
        bks_str = str(bks) if bks > 0 else "N/A"
        print(f"{idx+1:<5} {name:<20} {n:<10} {bks_str:<10}")
        instances_data.append({'file': f, 'name': name, 'bks': bks, 'n': n})

    print("-" * 50)
    selection = input("Chọn ID để chạy (Ví dụ: 1, 3 hoặc 'all'): ").strip()
    
    selected_indices = []
    if selection.lower() == 'all':
        selected_indices = range(len(files))
    else:
        try:
            selected_indices = [int(x.strip()) - 1 for x in selection.split(',') if x.strip()]
        except ValueError:
            print("Lỗi nhập liệu.")
            return

    # --- CẢNH BÁO AN TOÀN CHO METHOD CŨ ---
    if method_choice == '1':
        risky_files = [instances_data[i]['name'] for i in selected_indices 
                       if 0 <= i < len(instances_data) and instances_data[i]['n'] > 100]
        if risky_files:
            print("\n⚠️  CẢNH BÁO: Bạn đã chọn Method 1 (Cũ) cho các file lớn:", risky_files)
            print("    Phương pháp này có thể bị TREO MÁY (Hang) hàng giờ.")
            confirm = input("    Bạn có chắc chắn muốn tiếp tục không? (y/n): ").lower()
            if confirm != 'y':
                print("Đã hủy bỏ.")
                return

    # --- 4. CẤU HÌNH CHẠY ---
    try:
        n_runs = int(input("Số lần chạy mỗi instance (Runs): ") or 1)
    except ValueError:
        n_runs = 1

    custom_csv = input(f"Tên file kết quả (Enter để dùng '{DEFAULT_RESULT_FILE}'): ").strip()
    result_file = custom_csv if custom_csv else DEFAULT_RESULT_FILE
    if not result_file.endswith('.csv'):
        result_file += '.csv'

    # --- 5. BẮT ĐẦU CHẠY ---
    file_exists = os.path.exists(result_file)
    
    with open(result_file, mode='a', newline='') as csv_file:
        fieldnames = [
            'Date', 'Method', 'Instance', 'N', 'K', 'BKS', 
            'Best_Cost', 'Avg_Cost', 'Worst_Cost', 
            'Best_Gap(%)', 'Avg_Time(s)', 'Runs'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Chỉ viết header nếu file mới tạo
        if not file_exists:
            writer.writeheader()

        print("\n" + "="*60)
        print(f"STARTING BENCHMARK | Method: {method_name} | Log: {result_file}")
        print("="*60)

        for idx in selected_indices:
            if idx < 0 or idx >= len(instances_data): continue
            
            data = instances_data[idx]
            filepath = os.path.join(INSTANCE_DIR, data['file'])
            
            print(f"\n>>> Running: {data['name']} (Size: {data['n']})")
            
            costs = []
            times = []
            
            for run in range(n_runs):
                print(f"    Run {run + 1}/{n_runs}...", end=" ", flush=True)
                start_t = time.time()
                try:
                    # Gọi hàm wrapper
                    _, cost = run_solver_wrapper(method_choice, filepath)
                    
                    elapsed = time.time() - start_t
                    costs.append(cost)
                    times.append(elapsed)
                    print(f"Done. Cost: {cost} | Time: {elapsed:.2f}s")
                    
                except Exception as e:
                    print(f"\n    ❌ ERROR: {e}")
                    # Nếu lỗi (ví dụ file cũ crash), dừng instance này
                    break
                except KeyboardInterrupt:
                    print("\n    🛑 Interrupted by User!")
                    return
            
            if not costs: continue

            # Tính toán thống kê
            best_c = min(costs)
            avg_c = np.mean(costs)
            worst_c = max(costs)
            avg_t = np.mean(times)
            gap = ((best_c - data['bks']) / data['bks'] * 100) if data['bks'] > 0 else 0.0
            
            # Ghi vào CSV
            row = {
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Method': method_name,
                'Instance': data['name'],
                'N': data['n'],
                'K': get_instance_info(data['file'])[2],
                'BKS': data['bks'],
                'Best_Cost': best_c,
                'Avg_Cost': f"{avg_c:.2f}",
                'Worst_Cost': worst_c,
                'Best_Gap(%)': f"{gap:.2f}",
                'Avg_Time(s)': f"{avg_t:.2f}",
                'Runs': n_runs
            }
            writer.writerow(row)
            csv_file.flush()
            
            print(f"    -> Stats: Best={best_c} (Gap {gap:.2f}%) | AvgT={avg_t:.2f}s")

    print("\n" + "="*60)
    print(f"DONE! Results saved to: {result_file}")

if __name__ == "__main__":
    run_benchmark()