#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Universal Benchmark Runner v2.0
===============================
Cải tiến:
1. Hỗ trợ chọn Folder nguồn (instances vs X).
2. Giao diện chọn bài thông minh (hỗ trợ cú pháp 1-5, 7, 9).
3. Tích hợp cả 2 phương pháp giải (Cũ & Mới).

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
try:
    # Kiểm tra xem file tồn tại không để tránh lỗi crash ngay lập tức
    has_v1 = os.path.exists("route_optimizer.py")
    has_v2 = os.path.exists("route_optimizer_v2.py")
    
    if has_v1: import route_optimizer
    if has_v2: import route_optimizer_v2
    
    if not has_v1 and not has_v2:
        print("❌ LỖI: Không tìm thấy cả 'route_optimizer.py' lẫn 'route_optimizer_v2.py'.")
        sys.exit(1)
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

# Database BKS (Best Known Solutions) - Mở rộng thêm cho bộ X
BKS_DB = {
    # Bộ A, B, E, F, P, M (Cũ)
    "P-n19-k2": 212, "P-n22-k2": 216, "E-n31-k7": 379,
    "A-n32-k5": 784, "A-n33-k6": 661, "A-n37-k5": 669,
    "B-n39-k5": 549, "F-n45-k4": 724, "P-n45-k5": 510,
    "E-n51-k5": 521, "P-n55-k7": 568, "A-n60-k9": 1354,
    "P-n101-k4": 681, 
    # Bộ X (Uchoa et al.) - Ví dụ một số bài nhỏ/vừa
    "X-n101-k25": 27591, "X-n106-k14": 26362, "X-n110-k13": 14971,
    "X-n115-k10": 12747, "X-n120-k6": 13332, "X-n125-k30": 55539,
    "X-n129-k18": 28940, "X-n134-k13": 10916, "X-n139-k10": 10594,
    "X-n143-k7": 15700, "X-n148-k46": 43448, "X-n153-k22": 21220,
    "X-n502-k39": 69226, "X-n1001-k43": 72355
}

def get_instance_info(filename):
    """Trích xuất thông tin N và K từ tên file."""
    name = filename.replace(".vrp", "")
    n_match = re.search(r'-n(\d+)', name)
    k_match = re.search(r'-k(\d+)', name)
    n = int(n_match.group(1)) if n_match else 0
    k = int(k_match.group(1)) if k_match else 0
    # Tìm BKS, nếu không có trả về 0
    bks = BKS_DB.get(name, 0)
    return name, n, k, bks

def run_solver_wrapper(method_choice, filepath):
    """Hàm bọc để gọi đúng phương pháp."""
    if method_choice == '1':
        if not has_v1: raise Exception("File route_optimizer.py không tồn tại!")
        return route_optimizer.solve_with_clarke_wright_and_optimize(filepath)
    else:
        if not has_v2: raise Exception("File route_optimizer_v2.py không tồn tại!")
        return route_optimizer_v2.solve(filepath)

def parse_selection(selection_str, max_len):
    """Xử lý chuỗi nhập phức tạp: '1, 2, 5-8' -> [0, 1, 4, 5, 6, 7]"""
    if selection_str.lower() == 'all':
        return range(max_len)
    
    selected = set()
    parts = selection_str.split(',')
    try:
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                # User nhập 1-based, convert sang 0-based
                for i in range(start, end + 1):
                    if 1 <= i <= max_len: selected.add(i - 1)
            else:
                if part:
                    i = int(part)
                    if 1 <= i <= max_len: selected.add(i - 1)
    except ValueError:
        return []
    return sorted(list(selected))

def run_benchmark():
    print("\n" + "="*60)
    print("       🚀 CVRP BENCHMARK RUNNER - UNIVERSAL TOOL 🚀")
    print("="*60)

    # --- 1. CHỌN FOLDER DỮ LIỆU ---
    print("\n[BƯỚC 1] Chọn Nguồn Dữ Liệu (Instances Folder):")
    print("   1. Folder 'instances' (Mặc định - Các bộ cũ A, B, P...)")
    print("   2. Folder 'X' (Uchoa et al. - Cùng cấp thư mục)")
    
    folder_choice = input("👉 Nhập lựa chọn (1 hoặc 2, Enter = 1): ").strip()
    
    if folder_choice == '2':
        target_dir = "X"
        default_csv = "benchmark_results_X.csv"
    else:
        target_dir = "instances"
        default_csv = "benchmark_results.csv"

    if not os.path.exists(target_dir):
        print(f"\n❌ LỖI: Không tìm thấy thư mục '{target_dir}'.")
        print(f"   Vui lòng tạo folder '{target_dir}' và copy file .vrp vào đó.")
        return

    files = [f for f in os.listdir(target_dir) if f.endswith(".vrp")]
    files.sort() # Sắp xếp tên file cho dễ nhìn
    
    # Sắp xếp lại theo kích thước (Size N) để dễ chọn
    # Logic: Lấy N từ tên file, sort theo N tăng dần
    files.sort(key=lambda x: get_instance_info(x)[1])

    if not files:
        print(f"⚠️  Cảnh báo: Folder '{target_dir}' trống rỗng!")
        return

    # --- 2. CHỌN PHƯƠNG PHÁP ---
    print("\n[BƯỚC 2] Chọn Phương Pháp Giải (Algorithm):")
    print("   1. Method CŨ (route_optimizer.py) - Pipeline Clarke-Wright + LocalSearch")
    print("   2. Method MỚI (route_optimizer_v2.py) - Hybrid LNS + MaxSAT (Khuyên dùng)")
    
    method_choice = input("👉 Nhập lựa chọn (1 hoặc 2, Enter = 2): ").strip()
    if method_choice not in ['1', '2']: method_choice = '2'
    
    method_name = "V1_Old_Pipeline" if method_choice == '1' else "V2_New_HybridLNS"

    # --- 3. CHỌN FILE INSTANCE (GIAO DIỆN MỚI) ---
    print("\n" + "="*65)
    print(f"📂 DANH SÁCH FILE TRONG '{target_dir}' (Sắp xếp theo Size)")
    print("="*65)
    print(f"{'ID':<4} | {'Instance Name':<20} | {'Size':<6} | {'Xe(K)':<6} | {'BKS':<8}")
    print("-" * 65)
    
    instances_data = []
    for idx, f in enumerate(files):
        name, n, k, bks = get_instance_info(f)
        bks_str = str(bks) if bks > 0 else "?"
        print(f"{idx+1:<4} | {name:<20} | {n:<6} | {k:<6} | {bks_str:<8}")
        instances_data.append({'file': f, 'name': name, 'bks': bks, 'n': n, 'k': k})

    print("-" * 65)
    print("💡 HƯỚNG DẪN CHỌN:")
    print("   - Nhập 'all' để chạy tất cả.")
    print("   - Nhập số lẻ: '1, 3, 5'")
    print("   - Nhập theo dải: '1-5' (Chạy từ bài 1 đến bài 5)")
    print("   - Kết hợp: '1-3, 10' (Chạy bài 1,2,3 và bài 10)")
    
    selection = input("\n👉 Nhập các ID muốn chạy: ").strip()
    selected_indices = parse_selection(selection, len(files))
    
    if not selected_indices:
        print("❌ Không có bài nào được chọn hoặc nhập sai.")
        return

    # --- CẢNH BÁO AN TOÀN ---
    if method_choice == '1':
        risky_files = [instances_data[i]['name'] for i in selected_indices if instances_data[i]['n'] > 150]
        if risky_files:
            print(f"\n⚠️  CẢNH BÁO NGUY HIỂM: Method 1 (Cũ) có thể treo với N > 150.")
            print(f"   Các file rủi ro: {risky_files}")
            confirm = input("   Tiếp tục? (y/n): ").lower()
            if confirm != 'y': return

    # --- 4. CẤU HÌNH CUỐI ---
    try:
        n_runs_input = input("👉 Số lần chạy mỗi bài (Mặc định 1): ")
        n_runs = int(n_runs_input) if n_runs_input else 1
    except ValueError:
        n_runs = 1

    custom_csv = input(f"👉 Tên file lưu kết quả (Mặc định '{default_csv}'): ").strip()
    result_file = custom_csv if custom_csv else default_csv
    if not result_file.endswith('.csv'): result_file += '.csv'

    # --- 5. EXECUTION LOOP ---
    # Check if header exists
    file_exists = os.path.exists(result_file)
    
    with open(result_file, mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = [
            'Date', 'Source_Dir', 'Method', 'Instance', 'N', 'K', 'BKS', 
            'Best_Cost', 'Avg_Cost', 'Gap(%)', 'Avg_Time(s)', 'Runs'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()

        print("\n" + "▒"*60)
        print(f"▶️  ĐANG CHẠY... (Kết quả lưu tại: {result_file})")
        print("▒"*60)

        for i, idx in enumerate(selected_indices):
            data = instances_data[idx]
            filepath = os.path.join(target_dir, data['file'])
            
            print(f"\n[{i+1}/{len(selected_indices)}] Testing: {data['name']} (N={data['n']}, K={data['k']})")
            
            costs = []
            times = []
            
            for run in range(n_runs):
                print(f"    ↳ Run {run+1}/{n_runs}...", end=" ", flush=True)
                start_t = time.time()
                try:
                    _, cost = run_solver_wrapper(method_choice, filepath)
                    elapsed = time.time() - start_t
                    costs.append(cost)
                    times.append(elapsed)
                    print(f"✅ OK. Cost: {cost} | Time: {elapsed:.2f}s")
                except KeyboardInterrupt:
                    print("\n🛑 Dừng bởi người dùng!")
                    return
                except Exception as e:
                    print(f"\n❌ LỖI: {e}")
                    break # Skip to next instance
            
            if costs:
                best_c = min(costs)
                avg_c = np.mean(costs)
                avg_t = np.mean(times)
                # Tính Gap
                if data['bks'] > 0:
                    gap = ((best_c - data['bks']) / data['bks']) * 100
                    gap_str = f"{gap:.2f}"
                else:
                    gap_str = "N/A"
                
                # In tóm tắt ngay lập tức
                print(f"    📊 KẾT QUẢ: Best={best_c} (Gap {gap_str}%) | AvgTime={avg_t:.2f}s")
                
                # Ghi CSV
                writer.writerow({
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'Source_Dir': target_dir,
                    'Method': method_name,
                    'Instance': data['name'],
                    'N': data['n'],
                    'K': data['k'],
                    'BKS': data['bks'] if data['bks'] > 0 else '?',
                    'Best_Cost': best_c,
                    'Avg_Cost': f"{avg_c:.1f}",
                    'Gap(%)': gap_str,
                    'Avg_Time(s)': f"{avg_t:.2f}",
                    'Runs': n_runs
                })
                csv_file.flush()

    print("\n" + "="*60)
    print("🎉 HOÀN TẤT! Kiểm tra file CSV để xem báo cáo.")

if __name__ == "__main__":
    run_benchmark()