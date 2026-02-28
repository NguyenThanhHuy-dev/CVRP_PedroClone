#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Universal Benchmark Runner v3.2
===============================
Hỗ trợ:
1. Chọn Folder nguồn (instances vs X).
2. Chọn thuật toán:
   - Method 1: route_optimizer.py (Cũ/V1)
   - Method 2: route_optimizer_v3.py (Mới/V3 - Updated Logic)
3. Chọn bài chạy linh hoạt (1-5, 7, 9).
"""

import os
import sys
import time
import csv
import numpy as np
from datetime import datetime
import re
import importlib.util
import multiprocessing

TIMEOUT_LIMIT = 1200 # 20 minutes limit per instance

# --- HÀM IMPORT FILE LINH HOẠT ---
def import_module_from_file(module_name, file_path):
    if not os.path.exists(file_path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import 2 version optimizer
# Lưu ý: Đảm bảo file route_optimizer.py và route_optimizer_v3.py nằm cùng thư mục
route_optimizer = import_module_from_file("route_optimizer", "route_optimizer.py")
route_optimizer_v4 = import_module_from_file("route_optimizer_v4", "route_optimizer_v4.py")

# Database BKS (Best Known Solutions)
BKS_DB = {
    # --- Bộ Cũ (A, B, E, F, P, M) - Giữ nguyên để tham khảo ---
    "P-n19-k2": 212, "P-n22-k2": 216, "E-n31-k7": 379,
    "A-n32-k5": 784, "A-n33-k6": 661, "A-n37-k5": 669,
    "B-n39-k5": 549, "F-n45-k4": 724, "P-n45-k5": 510,
    "E-n51-k5": 521, "P-n55-k7": 568, "A-n60-k9": 1354,
    "P-n101-k4": 681, 

    # --- Bộ X (Uchoa et al.) - FULL 100 Instances ---
    "X-n101-k25": 27591, 
    "X-n106-k14": 26362, 
    "X-n110-k13": 14971, 
    "X-n115-k10": 12747, 
    "X-n120-k6": 13332, 
    "X-n125-k30": 55539, 
    "X-n129-k18": 28940, 
    "X-n134-k13": 10916, 
    "X-n139-k10": 13590, 
    "X-n143-k7": 15700, 
    "X-n148-k46": 43448, 
    "X-n153-k22": 21220, 
    "X-n157-k13": 16876, 
    "X-n162-k11": 14138, 
    "X-n167-k10": 20557, 
    "X-n172-k51": 45607, 
    "X-n176-k26": 47812, 
    "X-n181-k23": 25569, 
    "X-n186-k15": 24145, 
    "X-n190-k8": 16980, 
    "X-n195-k51": 44225, 
    "X-n200-k36": 58578, 
    "X-n204-k19": 19565, 
    "X-n209-k16": 30656, 
    "X-n214-k11": 10856, 
    "X-n219-k73": 117595, 
    "X-n223-k34": 40437, 
    "X-n228-k23": 25742, 
    "X-n233-k16": 19230, 
    "X-n237-k14": 27042, 
    "X-n242-k48": 82751, 
    "X-n247-k50": 37274, 
    "X-n251-k28": 38684, 
    "X-n256-k16": 18839, 
    "X-n261-k13": 26558, 
    "X-n266-k58": 75478, 
    "X-n270-k35": 35291, 
    "X-n275-k28": 21245, 
    "X-n280-k17": 33503, 
    "X-n284-k15": 20215, 
    "X-n289-k60": 95151, 
    "X-n294-k50": 47161, 
    "X-n298-k31": 34231, 
    "X-n303-k21": 21736, 
    "X-n308-k13": 25859, 
    "X-n313-k71": 94043, 
    "X-n317-k53": 78355, 
    "X-n322-k28": 29834, 
    "X-n327-k20": 27532, 
    "X-n331-k15": 31102, 
    "X-n336-k84": 139111, 
    "X-n344-k43": 42050, 
    "X-n351-k40": 25896, 
    "X-n359-k29": 51505, 
    "X-n367-k17": 22814, 
    "X-n376-k94": 147713, 
    "X-n384-k52": 65928, 
    "X-n393-k38": 38260, 
    "X-n401-k29": 66154, 
    "X-n411-k19": 19712, 
    "X-n420-k130": 107798, 
    "X-n429-k61": 65449, 
    "X-n439-k37": 36391, 
    "X-n449-k29": 55233, 
    "X-n459-k26": 24139, 
    "X-n469-k138": 221824, 
    "X-n480-k70": 89449, 
    "X-n491-k59": 66483, 
    "X-n502-k39": 69226, 
    "X-n513-k21": 24201, 
    "X-n524-k153": 154593, 
    "X-n536-k96": 94846, 
    "X-n548-k50": 86700, 
    "X-n561-k42": 42717, 
    "X-n573-k30": 50673, 
    "X-n586-k159": 190316, 
    "X-n599-k92": 108451, 
    "X-n613-k62": 59535, 
    "X-n627-k43": 62164, 
    "X-n641-k35": 63682, 
    "X-n655-k131": 106780, 
    "X-n670-k130": 146332, 
    "X-n685-k75": 68205, 
    "X-n701-k44": 81923, 
    "X-n716-k35": 43373, 
    "X-n733-k159": 136187, 
    "X-n749-k98": 77269, 
    "X-n766-k71": 114417, 
    "X-n783-k48": 72386, 
    "X-n801-k40": 73305, 
    "X-n819-k171": 158121, 
    "X-n837-k142": 193737, 
    "X-n856-k95": 88965, 
    "X-n876-k59": 99299, 
    "X-n895-k37": 53860, 
    "X-n916-k207": 329179, 
    "X-n936-k151": 132715, 
    "X-n957-k87": 85465, 
    "X-n979-k58": 118976, 
    "X-n1001-k43": 72355
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
    """
    Hàm gọi solver dựa trên lựa chọn của người dùng.
    Trả về: (routes, cost, params)
    """
    if method_choice == '1':
        if not route_optimizer:
            raise Exception("File 'route_optimizer.py' không tồn tại!")
        routes, cost = route_optimizer.solve_with_clarke_wright_and_optimize(filepath, verbose=False)
        return routes, cost, {}
    else:
        if not route_optimizer_v4:
            raise Exception("File 'route_optimizer_v4.py' không tồn tại!")
        return route_optimizer_v4.solve_with_clarke_wright_and_optimize(filepath, verbose=False)

def parse_selection(selection_str, max_len):
    """Xử lý chuỗi nhập: '1, 2, 5-8' -> [0, 1, 4, 5, 6, 7]"""
    if not selection_str: return []
    if selection_str.lower() == 'all': return range(max_len)
    
    selected = set()
    parts = selection_str.split(',')
    try:
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
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

    print("\n[BƯỚC 1] Chọn Nguồn Dữ Liệu:")
    print("   1. Folder 'instances' (Bộ cũ: A, B, P...)")
    print("   2. Folder 'X' (Bộ mới: Uchoa et al.)")
    
    folder_choice = input("👉 Nhập lựa chọn (1/2, Enter=1): ").strip()
    
    if folder_choice == '2':
        target_dir = "X"
        default_csv = "results_X_v3_validated.csv"
    else:
        target_dir = "instances"
        default_csv = "results_instances_v3_validated.csv"

    if not os.path.exists(target_dir):
        print(f"\n❌ LỖI: Không tìm thấy thư mục '{target_dir}'.")
        return

    files = [f for f in os.listdir(target_dir) if f.endswith(".vrp")]
    files.sort(key=lambda x: get_instance_info(x)[1])

    if not files:
        print(f"⚠️  Folder '{target_dir}' trống!")
        return

    print("\n[BƯỚC 2] Chọn Phương Pháp Giải:")
    print("   1. Method 1: route_optimizer.py (Cũ - V1)")
    print("   2. Method 2: route_optimizer_v4.py (Mới/V4)")
    
    method_choice = input("👉 Nhập lựa chọn (1/2, Enter=2): ").strip()
    if method_choice not in ['1', '2']: method_choice = '2'
    
    method_name = "V1_Old" if method_choice == '1' else "V4_New"

    print("\n" + "="*65)
    print(f"📂 DANH SÁCH FILE TRONG '{target_dir}'")
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
    print("💡 VÍ DỤ NHẬP:")
    print("   - 'all' : Chạy hết")
    print("   - '1-5' : Chạy bài 1 đến 5")
    print("   - '1, 3, 5' : Chạy bài lẻ 1, 3, 5")
    
    selection = input("\n👉 Nhập các ID muốn chạy: ").strip()
    selected_indices = parse_selection(selection, len(files))
    
    if not selected_indices:
        print("❌ Chưa chọn bài nào.")
        return

    try:
        n_runs_input = input("👉 Số lần chạy mỗi bài (Mặc định 1): ")
        n_runs = int(n_runs_input) if n_runs_input else 1
    except: n_runs = 1

    custom_csv = input(f"👉 Tên file CSV kết quả (Mặc định '{default_csv}'): ").strip()
    result_file = custom_csv if custom_csv else default_csv
    if not result_file.endswith('.csv'): result_file += '.csv'

    file_exists = os.path.exists(result_file)
    with open(result_file, mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Date', 'Method', 'Instance', 'N', 'K', 'BKS', 'Best', 'Avg', 'Gap(%)', 'Time(s)', 'Runs',
                      'n_customers', 'n_restarts', 'time_limit', 'is_valid', 'final_k']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        
        print("\n" + "▒"*60)
        print(f"▶️  BẮT ĐẦU CHẠY {method_name}... (Lưu tại: {result_file})")
        print("▒"*60)

        for i, idx in enumerate(selected_indices):
            data = instances_data[idx]
            filepath = os.path.join(target_dir, data['file'])
            
            print(f"\n[{i+1}/{len(selected_indices)}] Bài: {data['name']} (N={data['n']})")
            
            costs = []
            times = []
            
            for run in range(n_runs):
                print(f"    ↳ Run {run+1}...", end=" ", flush=True)
                
                # --- TIMEOUT HANDLING WITH MULTIPROCESSING ---
                # Wrapper function for worker process
                def solver_worker(method_choice, filepath, return_dict):
                    try:
                        res = run_solver_wrapper(method_choice, filepath)
                        return_dict['result'] = res
                    except Exception as e:
                        return_dict['error'] = str(e)

                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                
                p = multiprocessing.Process(target=solver_worker, args=(method_choice, filepath, return_dict))
                
                start_t = time.time()
                p.start()
                p.join(timeout=TIMEOUT_LIMIT) # Timeout 1200s

                if p.is_alive():
                    # TIMEOUT OCCURRED
                    p.terminate()
                    p.join()
                    elapsed = TIMEOUT_LIMIT
                    print(f"❌ TIMEOUT (> {TIMEOUT_LIMIT}s) | Killing process...")
                    
                    # Record failure
                    costs.append(-1) # Indicator for timeout/fail
                    times.append(elapsed)
                    last_params = {'is_valid': False, 'note': 'TIMEOUT'}
                else:
                    # FINISHED IN TIME
                    elapsed = time.time() - start_t
                    
                    if 'error' in return_dict:
                        print(f"\n❌ LỖI: {return_dict['error']}")
                        last_params = {}
                        # Option: không append cost hoặc append cost vô cùng lớn
                    elif 'result' in return_dict:
                        results = return_dict['result']
                        # routes = results[0] # unused here
                        cost = results[1]
                        params = results[2]
                        
                        costs.append(cost)
                        times.append(elapsed)
                        last_params = params
                        print(f"✅ Cost: {cost} | Time: {elapsed:.2f}s")
                    else:
                         print("\n❌ LỖI: Không nhận được kết quả (Unknown Error)")
                         last_params = {}
            
            if costs:
                best_c = min(costs)
                avg_c = np.mean(costs)
                avg_t = np.mean(times)
                gap = ((best_c - data['bks']) / data['bks'] * 100) if data['bks'] > 0 else 0
                gap_str = f"{gap:.2f}" if data['bks'] > 0 else "?"
                
                
                # Check validation status from params
                is_valid = last_params.get('is_valid', True)
                valid_icon = "✅" if is_valid else "⚠️ INVALID"
                
                print(f"    📊 KẾT QUẢ: Best={best_c} (Gap {gap_str}%) | AvgTime={avg_t:.2f}s | {valid_icon}")
                
                writer.writerow({
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'Method': method_name,
                    'Instance': data['name'],
                    'N': data['n'], 'K': data['k'],
                    'BKS': data['bks'] if data['bks'] > 0 else '?',
                    'Best': best_c, 'Avg': f"{avg_c:.1f}",
                    'Gap(%)': gap_str, 'Time(s)': f"{avg_t:.2f}",
                    'Runs': n_runs,
                    'n_customers': last_params.get('n_customers', ''),
                    'n_restarts': last_params.get('n_restarts', ''),
                    'time_limit': last_params.get('time_limit', ''),
                    'is_valid': is_valid,
                    'final_k': last_params.get('final_k', '')
                })
                csv_file.flush()

    print("\n🎉 Đã chạy xong!")

if __name__ == "__main__":
    run_benchmark()