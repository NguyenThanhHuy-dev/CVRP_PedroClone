#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Route Optimizer using MaxSAT (Refactored)
=========================================
Tối ưu CVRP sử dụng MaxSAT với các chiến lược:
1. Single Route Optimization: Tối ưu thứ tự trong mỗi route
2. Pair-wise Route Optimization: Gộp 2 routes, tối ưu phân hoạch + ordering

Author: Adapted from RTSS + CVRP MaxSAT paper
"""

import sys
import os
import time
import numpy as np
import traceback
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from classes.instance import Instance
from classes.clarke_wright import ClarkeWright
from classes.two_opt import TwoOpt

from core.cost_utils import calculate_route_cost
from search.guided_local_search import guided_local_search
from search.iterated_local_search import multi_start_ils
from optimization.route_optimization import optimize_all_routes, pairwise_optimize

def solve_with_clarke_wright_and_optimize(filepath: str, verbose: bool = True):
    if verbose:
        print("=" * 60)
        print("CVRP Solver: Clarke-Wright + MaxSAT Optimization")
        print("=" * 60)
    
    # Load instance
    cvrp = Instance(filepath)
    cvrp.load()
    
    n_vehicles = int(re.search(r'-k(\d+)', filepath).group(1))
    
    if verbose:
        print(f"\nInstance: {filepath}")
        print(f"  Dimension: {cvrp.dimension}")
        print(f"  Vehicles: {n_vehicles}")
        print(f"  Capacity: {cvrp.capacity}")
    
    # step 1: CW
    if verbose: print("\n=== Step 1: Clarke-Wright Heuristic ===")
    try:
        cw_time, cw_routes = ClarkeWright.run(cvrp, n_vehicles)
    except Exception as e:
        print("\n" + "!"*60)
        print(f"⚠️ WARNING: Clarke-Wright failed with k={n_vehicles}. Retrying with k=999 (Unlimited)...")
        print(f"Error: {e}")
        try:
            cw_time, cw_routes = ClarkeWright.run(cvrp, 999)
            print("Fallback successful with k=999")
        except Exception as e2:
             print(f"CRITICAL ERROR during fallback: {e2}")
             traceback.print_exc()
             return {}, float('inf'), {}
    
    cw_cost = sum(route.cost for route in cw_routes.values())
    if verbose:
        print(f"  Time: {cw_time:.3f}s")
        print(f"  Cost: {cw_cost}")
    
    # Convert to format
    routes = {i: list(route.value) for i, (_, route) in enumerate(cw_routes.items())}
    
    # step2: 2-opt
    if verbose: print("\n=== Step 2: Two-Opt Local Search ===")
    two_opt_time, two_opt_routes = TwoOpt.run(cw_routes)
    two_opt_cost = sum(route.cost for route in two_opt_routes.values())
    if verbose:
        print(f"  Time: {two_opt_time:.3f}s")
        print(f"  Cost: {two_opt_cost}")
    
    routes = {i: list(route.value) for i, (_, route) in enumerate(two_opt_routes.items())}
    demands = np.array(cvrp.demands)
    
    # step 3: Multi-Start ILS
    if verbose: print("\n=== Step 3: Multi-Start ILS ===")
    start_time = time.time()
    
    # Adaptive parameters based on instance size - GREEDY settings
    n_customers = len(demands) - 1 
    if n_customers > 80:
        n_restarts = 5
        time_limit = 60.0
    elif n_customers > 50:
        n_restarts = 10
        time_limit = 60.0
    else:
        n_restarts = 20
        time_limit = 30.0
    
    ils_routes, ils_cost = multi_start_ils(
        routes.copy(), cvrp.distances, demands, cvrp.capacity,
        n_restarts=n_restarts, time_limit=time_limit
    )
    ils_time = time.time() - start_time
    if verbose:
        print(f"\n  Time: {ils_time:.3f}s")
        print(f"  Cost: {ils_cost}")
    
    # step 4: Guided Local Search (GLS)
    if verbose: print("\n=== Step 4: Guided Local Search ===")
    start_time = time.time()
    
    # GLS time based on instance size
    gls_time_limit = 60.0 if n_customers > 50 else 30.0
    gls_routes, gls_cost = guided_local_search(
        ils_routes.copy(), cvrp.distances, demands, cvrp.capacity,
        max_iterations=200, time_limit=gls_time_limit
    )
    gls_time = time.time() - start_time
    if verbose:
        print(f"  Time: {gls_time:.3f}s")
        print(f"  Cost: {gls_cost}")
    
    # step 5: MaxSAT single route optimization
    if verbose: print("\n=== Step 5: MaxSAT Single Route Optimization ===")
    start_time = time.time()
    
    opt_routes, opt_cost = optimize_all_routes(gls_routes, cvrp.distances, demands, cvrp.capacity)
    single_time = time.time() - start_time
    if verbose:
        print(f"  Time: {single_time:.3f}s")
        print(f"  Cost: {opt_cost}")
    
    # step 6: MaxSAT pair-wise optimization (Inter-route)
    if verbose: print("\n=== Step 6: MaxSAT Pair-wise Optimization ===")
    start_time = time.time()
    
    pair_routes, pair_cost = pairwise_optimize(
        opt_routes.copy(), cvrp.distances, demands, cvrp.capacity, max_iterations=5
    )
    pair_time = time.time() - start_time
    if verbose:
        print(f"\n  Time: {pair_time:.3f}s")
        print(f"  Cost: {pair_cost}")
    
    # Get optimal from solution file
    sol_file = filepath.replace('.vrp', '.sol')
    optimal = None
    if os.path.exists(sol_file):
        with open(sol_file) as f:
            for line in f:
                if line.startswith('Cost'):
                    optimal = int(line.split()[1])
                    break
    
    # Validation Logic
    final_k = len(pair_routes)
    is_valid = (final_k == n_vehicles)
    
    # Summary
    if verbose:
        total_time = cw_time + two_opt_time + ils_time + gls_time + single_time + pair_time
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Clarke-Wright:    {cw_cost}")
        print(f"  + Two-Opt:        {two_opt_cost} (improved {cw_cost - two_opt_cost})")
        print(f"  + ILS:            {ils_cost} (improved {two_opt_cost - ils_cost})")
        print(f"  + GLS:            {gls_cost} (improved {ils_cost - gls_cost})")
        print(f"  + MaxSAT Single:  {opt_cost} (improved {gls_cost - opt_cost})")
        print(f"  + MaxSAT Pair:    {pair_cost} (improved {opt_cost - pair_cost})")
        print(f"  Total time:       {total_time:.3f}s")
        print(f"  Vehicles:         {final_k} / {n_vehicles} ({'VALID' if is_valid else 'INVALID'})")
        
        if optimal:
            print(f"\n  Optimal (from file): {optimal}")
            print(f"  Gap: {(pair_cost - optimal) / optimal * 100:.1f}%")
        print("\nFinal Routes:")
        for v, route in pair_routes.items():
            demand = sum(demands[c] for c in route)
            cost = calculate_route_cost(route, cvrp.distances)
            print(f"  Route {v}: {route} (demand={demand}, cost={cost})")
    
    # Return parameters for logging
    params = {
        'n_customers': n_customers,
        'n_restarts': n_restarts,
        'time_limit': time_limit,
        'is_valid': is_valid,
        'final_k': final_k
    }
    
    return pair_routes, pair_cost, params

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        
        # Sửa lỗi relative path nếu execute từ trong folder:
        # sys.argv[1] nếu từ outside thì là 'instances/X.vrp'. Nếu script run inside thì cần chỉnh lại đường dẫn
        if not os.path.exists(filepath) and os.path.exists(os.path.join(parent_dir, filepath)):
            filepath = os.path.join(parent_dir, filepath)
    else:
        filepath = os.path.join(parent_dir, "instances/E-n31-k7.vrp")
    
    solve_with_clarke_wright_and_optimize(filepath)
