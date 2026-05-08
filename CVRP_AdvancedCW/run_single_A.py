#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Single Instance Tester for Augerat A-Set
========================================
Sử dụng: python run_single_A.py [METHOD] [FILENAME] [ALNS_ITERS] [PAIR_SIZE]
Ví dụ 1 (Chạy nhanh): 
    python run_single_A.py pysat A-n37-k5.vrp
Ví dụ 2 (Ép RC2 chạy sâu): 
    python run_single_A.py pysat A-n37-k5.vrp 200 20
"""

import os
import sys
import time
import math
import logging
from run_benchmark_A import (
    VALID_METHODS, get_instance_info, get_bks_from_sol, 
    build_dynamic_config, RESULT_FILE, INSTANCE_DIR,
    suppress_logging_to_console, restore_logging_to_console
)
from csv_upsert import load_csv, save_csv, upsert_row

def main():
    if len(sys.argv) < 3:
        print("Cú pháp: python run_single_A.py [METHOD] [FILENAME] [OPTIONAL: ALNS_ITER] [OPTIONAL: PAIR_SIZE]")
        print("Ví dụ: python run_single_A.py pysat A-n37-k5.vrp")
        sys.exit(1)

    method = sys.argv[1].strip().lower()
    filename = sys.argv[2].strip()
    
    # Custom Overrides
    custom_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    custom_pair  = int(sys.argv[4]) if len(sys.argv) > 4 else -1

    if method not in VALID_METHODS:
        print(f"[LỖI] Method {method} không hỗ trợ.")
        sys.exit(1)

    filepath = os.path.join(INSTANCE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[LỖI] Không tìm thấy file {filepath}")
        sys.exit(1)

    print(f"[INFO] Nạp solver: {method.upper()}...")
    if method == "gurobi":
        from advanced_optimizer_gurobi import solve_advanced
    elif method == "cplex":
        from advanced_optimizer_cplex import solve_advanced
    else:  
        from advanced_optimizer_pysat import solve_advanced

    name, n, k = get_instance_info(filename)
    sol_filepath = filepath.replace(".vrp", ".sol")
    bks = get_bks_from_sol(sol_filepath)

    cfg = build_dynamic_config(n, k)
    if custom_pair > 0:
        print(f"[WARNING] Override Pair_Size = {custom_pair} (Gốc: {cfg['max_pairwise_size']})")
        cfg['max_pairwise_size'] = custom_pair
        cfg['pairwise_timeout'] = 120.0 # Nếu ép chạy to thì nới time ra cho nó chạy

    print("\n" + "=" * 70)
    print(f"TEST ĐƠN: {name} (N={n}, K={k}, BKS={bks})")
    print(f"  [Config] single={cfg['max_single_size']} pair={cfg['max_pairwise_size']} "
          f"s_to={cfg['single_timeout']:.0f}s p_to={cfg['pairwise_timeout']:.0f}s")
    print(f"  [ALNS Iters] {custom_iters}")
    print("=" * 70)

    csv_data = load_csv(RESULT_FILE)
    
    start_t = time.time()
    try:
        # Chúng ta không chặn console ở Single test để bạn xem log RC2 nhảy như nào
        opt_routes, opt_cost, stats = solve_advanced(
            filepath, config=cfg, max_iterations=custom_iters, target_cost=float(bks)
        )
        
        elapsed = time.time() - start_t
        gap = ((opt_cost - bks) / bks) * 100 if bks > 0 else 0.0

        print(f"\n[KẾT QUẢ CUỐI CÙNG]")
        print(f"  -> Cost: {opt_cost} | Gap: {gap:+.2f}% | Time: {elapsed:.1f}s")
        print(f"  -> S-Imp: {stats.get('single_imp_count', 0)} | P-Imp: {stats.get('pairwise_imp_count', 0)}")

        # Upsert (Cập nhật điểm cao nhất) vào file CSV hệ thống!
        csv_data = upsert_row(
            data=csv_data, instance=name, n=n, k=k, bks=bks,
            new_cost=opt_cost, elapsed=elapsed, cfg=cfg, stats=stats,
            method=method, max_iterations=custom_iters,
        )
        save_csv(RESULT_FILE, csv_data)
        
        row = csv_data[(name, method)]
        print(f"  -> Lịch sử (CSV): Runs={row['Runs']} | Best={row['Best_Cost']} ({row['Best_Gap(%)']}%) | Avg={row['Avg_Cost']} ({row['Avg_Gap(%)']}%)")

    except KeyboardInterrupt:
        print("\n[STOP] Bạn đã bấm Ctrl+C ngắt tiến trình.")
    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG] {e}")

if __name__ == "__main__":
    main()