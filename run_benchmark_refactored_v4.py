#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Benchmark Runner for Refactored V4 (Full Suite)
===============================================
Chạy full benchmark cho bộ A, B, E, F, P sử dụng route_optimizer_v4_refactored
Lưu kết quả vào CSV với đầy đủ thông tin như bản gốc.
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
from pathlib import Path

TIMEOUT_LIMIT = 1200  # 20 minutes limit per instance

# --- HÀM IMPORT FILE LINH HOẠT ---
def import_module_from_file(module_name, file_path):
    if not os.path.exists(file_path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import refactored optimizer
# Sửa lại dòng import module:
route_optimizer_v4_refactored = import_module_from_file(
    "route_optimizer_v4_refactored", 
    "route_optimizer_v4_refactored/main.py"  # Trỏ đúng vào file main
)

# Database BKS (Best Known Solutions)
BKS_DB = {
    # --- Bộ Cũ (A, B, E, F, P) ---
    "A-n32-k5": 784, "A-n33-k5": 643, "A-n33-k6": 661, "A-n34-k5": 778,
    "A-n36-k5": 799, "A-n37-k5": 669, "A-n37-k6": 949, "A-n38-k5": 730,
    "A-n39-k5": 822, "A-n39-k6": 831, "A-n44-k6": 937, "A-n45-k6": 944,
    "A-n45-k7": 1146, "A-n46-k7": 1044, "A-n48-k7": 1073, "A-n53-k7": 1010,
    "A-n54-k7": 1167, "A-n55-k9": 1073, "A-n60-k9": 1354, "A-n61-k9": 1034,
    "A-n62-k8": 1288, "A-n63-k9": 1616, "A-n63-k10": 1314, "A-n64-k9": 1401,
    "A-n65-k9": 1174, "A-n69-k9": 1159, "A-n80-k10": 1763,
    
    "B-n31-k5": 672, "B-n34-k5": 788, "B-n35-k5": 955, "B-n38-k6": 805,
    "B-n39-k5": 549, "B-n41-k6": 829, "B-n43-k6": 742, "B-n44-k7": 909,
    "B-n45-k5": 751, "B-n45-k6": 678, "B-n50-k7": 741, "B-n50-k8": 1312,
    "B-n51-k7": 1032, "B-n52-k7": 747, "B-n56-k7": 898, "B-n57-k7": 1099,
    "B-n57-k9": 1595, "B-n63-k10": 1534, "B-n64-k9": 861, "B-n66-k9": 1316,
    "B-n67-k10": 1032, "B-n78-k10": 1221,
    
    "E-n31-k7": 379, "E-n51-k5": 521, "E-n76-k7": 682, "E-n76-k8": 735,
    "E-n76-k10": 830, "E-n76-k14": 1021, "E-n101-k8": 815, "E-n101-k14": 1071,
    "E-n76-k15": 1299, "E-n101-k5": 855,
    
    "F-n45-k4": 724, "F-n72-k4": 237, "F-n135-k7": 1162,
    
    "P-n19-k2": 212, "P-n22-k2": 216, "P-n45-k5": 510, "P-n55-k7": 568,
    "P-n101-k4": 681,
}

def get_instance_info(filepath):
    """Extract instance information from filename."""
    filename = os.path.basename(filepath)
    name = filename.replace('.vrp', '')
    
    # Parse N and K from filename
    match = re.search(r'n(\d+)', name)
    n = int(match.group(1)) if match else 0
    
    match = re.search(r'k(\d+)', name)
    k = int(match.group(1)) if match else 0
    
    bks = BKS_DB.get(name, 0)
    return name, n, k, bks

def run_solver_wrapper(filepath):
    """
    Hàm gọi solver (refactored v4) và trả về kết quả.
    Trả về: (routes, cost, params)
    """
    if not route_optimizer_v4_refactored:
        raise Exception("Folder 'route_optimizer_v4_refactored' không tồn tại!")
    return route_optimizer_v4_refactored.solve_with_clarke_wright_and_optimize(filepath, verbose=False)

def main():
    print("\n" + "="*70)
    print("BENCHMARK RUNNER FOR ROUTE_OPTIMIZER_V4_REFACTORED")
    print("Full Suite: A, B, E, F, P")
    print("="*70)
    
    # Collect all instances from A, B, E, F, P folders
    instances_list = []
    instances_dir = "instances"
    
    for folder_name in ['A', 'B', 'E', 'F', 'P']:
        folder_path = os.path.join(instances_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"⚠️  Folder '{folder_path}' không tồn tại, skip.")
            continue
        
        vrp_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.vrp')])
        print(f"\n📂 Folder {folder_name}: {len(vrp_files)} instances")
        
        for vrp_file in vrp_files:
            filepath = os.path.join(folder_path, vrp_file)
            name, n, k, bks = get_instance_info(filepath)
            instances_list.append({
                'filepath': filepath,
                'name': name,
                'n': n,
                'k': k,
                'bks': bks
            })
            print(f"  ✓ {name} (N={n}, K={k}, BKS={bks if bks > 0 else '?'})")
    
    print(f"\n📊 Tổng cộng: {len(instances_list)} instances")
    
    # Setup output CSV
    result_file = f"results_refactored_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"\n💾 CSV output: {result_file}")
    
    file_exists = os.path.exists(result_file)
    fieldnames = ['Date', 'Method', 'Instance', 'N', 'K', 'BKS', 'Best', 'Avg', 'Gap(%)', 'Time(s)', 'Runs',
                  'n_customers', 'n_restarts', 'time_limit', 'is_valid', 'final_k']
    
    with open(result_file, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        print("\n" + "▒"*70)
        print("▶️  BẮT ĐẦU CHẠY BENCHMARK...")
        print("▒"*70)
        
        for idx, data in enumerate(instances_list):
            print(f"\n[{idx+1}/{len(instances_list)}] {data['name']} (N={data['n']}, K={data['k']})")
            
            filepath = data['filepath']
            
            costs = []
            times = []
            params_list = []
            
            n_runs = 10  # Run 10 times per instance for experimental results
            
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
                    costs.append(float('inf'))
                    times.append(0)
                    params_list.append({})
            
            if costs and min(costs) < float('inf'):
                best_c = min(costs)
                avg_c = np.mean(costs)
                avg_t = np.mean(times)
                gap = ((best_c - data['bks']) / data['bks'] * 100) if data['bks'] > 0 else 0
                gap_str = f"{gap:.2f}" if data['bks'] > 0 else "?"
                
                # Get params from best run
                best_idx = costs.index(best_c)
                last_params = params_list[best_idx] if best_idx < len(params_list) else {}
                
                is_valid = last_params.get('is_valid', True)
                valid_icon = "✅" if is_valid else "⚠️  INVALID"
                
                print(f"  📊 RESULT: Best={best_c} (Gap {gap_str}%) | AvgTime={avg_t:.2f}s | {valid_icon}")
                
                writer.writerow({
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'Method': "V4_Refactored",
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
                    'final_k': last_params.get('final_k', ''),
                })
                csv_file.flush()
            else:
                print(f"  ❌ ALL RUNS FAILED")
    
    print("\n" + "▒"*70)
    print(f"✅ BENCHMARK COMPLETED! Results saved to: {result_file}")
    print("▒"*70)

if __name__ == "__main__":
    main()
