#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Route Optimizer V2 (Fixed Logic)
================================
Fixes applied:
1. SingleRouteOptimizer: Trả về cost thực tế thay vì 0 khi không tìm thấy nghiệm (Fallback).
2. Added Final Validator: Tính toán lại tổng cost độc lập để kiểm tra kết quả.
"""

import sys
import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from itertools import combinations
import math
import random
import re

# Đảm bảo import được các module từ thư mục hiện tại
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from classes.instance import Instance
    from classes.clarke_wright import ClarkeWright
except ImportError:
    print("Lỗi: Không tìm thấy thư mục 'classes' hoặc module Instance.")
    sys.exit(1)

# ============================================================================
# PART 1: MAX-SAT OPTIMIZERS
# ============================================================================

class SingleRouteOptimizer:
    """Tối ưu thứ tự các điểm trong một route (TSP nhỏ)."""
    def __init__(self, customers: List[int], distances: np.ndarray):
        self.customers = customers
        self.n = len(customers) + 1
        self.distances = distances
        self.local_to_global = {0: 0}
        for i, c in enumerate(customers): self.local_to_global[i + 1] = c
        self.local_dist = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                gi, gj = self.local_to_global[i], self.local_to_global[j]
                self.local_dist[i, j] = distances[gi, gj]
        self.var_id = 0
        self.conNet = [[0] * self.n for _ in range(self.n)]
        self.rchNet = [[0] * self.n for _ in range(self.n)]
        self.wcnf = WCNF()

    def new_var_id(self) -> int:
        self.var_id += 1; return self.var_id

    def gen_model(self):
        # Variables
        for i in range(self.n):
            for j in range(self.n):
                if i != j: self.conNet[i][j] = self.new_var_id()
        for i in range(1, self.n):
            for j in range(i + 1, self.n):
                var = self.new_var_id()
                self.rchNet[i][j] = var; self.rchNet[j][i] = -var
        
        # Soft clauses
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.local_dist[i, j] > 0:
                    self.wcnf.append([-self.conNet[i][j]], weight=self.local_dist[i, j])
        
        # Hard clauses (Simplified TSP)
        for i in range(self.n):
            out_v = [self.conNet[i][j] for j in range(self.n) if i!=j]
            in_v = [self.conNet[j][i] for j in range(self.n) if i!=j]
            self.wcnf.append(out_v); self.wcnf.append(in_v) # At least one
            for x in range(len(out_v)):
                for y in range(x+1, len(out_v)): self.wcnf.append([-out_v[x], -out_v[y]])
            for x in range(len(in_v)):
                for y in range(x+1, len(in_v)): self.wcnf.append([-in_v[x], -in_v[y]])
        
        # Subtour elimination
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i!=j and self.rchNet[i][j]!=0:
                    self.wcnf.append([-self.conNet[i][j], self.rchNet[i][j]])
        
        for a in range(1, self.n):
            for b in range(1, self.n):
                if a==b: continue
                for c in range(1, self.n):
                    if c==a or c==b: continue
                    if self.rchNet[a][b] and self.rchNet[b][c]:
                        self.wcnf.append([-self.rchNet[a][b], -self.rchNet[b][c], self.rchNet[a][c]])
                        self.wcnf.append([-self.rchNet[a][b], -self.rchNet[b][c], -self.conNet[a][c]])

    def decode(self, model):
        pos = set(v for v in model if v > 0)
        route = []
        curr, visited = 0, {0}
        for _ in range(self.n - 1):
            for j in range(self.n):
                if j not in visited and self.conNet[curr][j] in pos:
                    if self.local_to_global[j] != 0: route.append(self.local_to_global[j])
                    visited.add(j); curr = j; break
        return route
    
    def calculate_initial_cost(self):
        # Tính cost của route ban đầu để làm fallback
        c = 0
        full_route = [0] + self.customers + [0] # 0 là local index của depot trong class này
        # Lưu ý: self.customers chứa global index, nhưng self.local_dist dùng local index
        # Ta dùng local_dist cho chính xác
        
        # Map global customers back to local indices: 1, 2, 3...
        local_route = [0] + list(range(1, self.n)) + [0] 
        # Cần chú ý: self.customers đầu vào đã theo thứ tự. 
        # Trong init: local_to_global[1] = customers[0]
        # Nghĩa là thứ tự 0 -> 1 -> 2 ... -> 0 chính là thứ tự của route ban đầu
        
        for k in range(len(local_route)-1):
            c += self.local_dist[local_route[k], local_route[k+1]]
        return c

    def optimize(self):
        # [FIX QUAN TRỌNG] Tính cost ban đầu trước khi chạy
        fallback_cost = self.calculate_initial_cost()
        
        self.gen_model()
        with RC2(self.wcnf, verbose=0) as solver:
            model = solver.compute()
            if model: 
                return self.decode(model), solver.cost
            
            # [FIX QUAN TRỌNG] Nếu fail, trả về route cũ và cost cũ (thay vì 0)
            return self.customers, fallback_cost

class PairwiseRouteOptimizer:
    """Tối ưu 2 routes: Assignment + Ordering."""
    def __init__(self, c1, c2, dists, dems, cap):
        self.c1, self.c2 = c1, c2
        self.all_c = c1 + c2
        self.n = len(self.all_c) + 1
        self.dists, self.dems, self.cap = dists, dems, cap
        self.l2g = {0:0}
        for i, c in enumerate(self.all_c): self.l2g[i+1] = c
        self.ldist = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                self.ldist[i,j] = dists[self.l2g[i], self.l2g[j]]
        self.wcnf = WCNF()
        self.var_id = 0
        self.con = [[[0]*self.n for _ in range(self.n)] for _ in range(2)]
        self.rch = [[[0]*self.n for _ in range(self.n)] for _ in range(2)]
        self.assign = [0]*self.n

    def nid(self): self.var_id+=1; return self.var_id

    def optimize(self):
        # Gen vars
        for i in range(1, self.n): self.assign[i] = self.nid()
        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i!=j: self.con[v][i][j] = self.nid()
        for v in range(2):
            for i in range(1, self.n):
                for j in range(i+1, self.n):
                    x = self.nid(); self.rch[v][i][j]=x; self.rch[v][j][i]=-x
        
        # Soft clauses
        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i!=j and self.ldist[i,j]>0:
                        self.wcnf.append([-self.con[v][i][j]], weight=self.ldist[i,j])
        
        # Hard clauses
        for i in range(1, self.n):
            for j in range(self.n):
                if i!=j:
                    self.wcnf.append([self.assign[i], -self.con[1][i][j]])
                    self.wcnf.append([self.assign[i], -self.con[1][j][i]])
                    self.wcnf.append([-self.assign[i], -self.con[0][i][j]])
                    self.wcnf.append([-self.assign[i], -self.con[0][j][i]])

        for c in range(1, self.n):
            in_v = [self.con[v][i][c] for v in range(2) for i in range(self.n) if i!=c]
            out_v = [self.con[v][c][j] for v in range(2) for j in range(self.n) if j!=c]
            self.wcnf.append(in_v); self.wcnf.append(out_v)
            for i in range(len(in_v)):
                for j in range(i+1, len(in_v)): self.wcnf.append([-in_v[i], -in_v[j]])
            for i in range(len(out_v)):
                for j in range(i+1, len(out_v)): self.wcnf.append([-out_v[i], -out_v[j]])

        for v in range(2):
            d_out = [self.con[v][0][j] for j in range(1, self.n)]
            d_in = [self.con[v][i][0] for i in range(1, self.n)]
            self.wcnf.append(d_out); self.wcnf.append(d_in)
            for i in range(len(d_out)):
                for j in range(i+1, len(d_out)): self.wcnf.append([-d_out[i], -d_out[j]])
            for i in range(len(d_in)):
                for j in range(i+1, len(d_in)): self.wcnf.append([-d_in[i], -d_in[j]])

        for v in range(2):
            for i in range(1, self.n):
                for j in range(1, self.n):
                    if i!=j and self.rch[v][i][j]!=0:
                        self.wcnf.append([-self.con[v][i][j], self.rch[v][i][j]])
            for a in range(1, self.n):
                for b in range(1, self.n):
                    if a==b: continue
                    for c in range(1, self.n):
                        if c==a or c==b: continue
                        if self.rch[v][a][b] and self.rch[v][b][c]:
                            self.wcnf.append([-self.rch[v][a][b], -self.rch[v][b][c], self.rch[v][a][c]])
                            self.wcnf.append([-self.rch[v][a][b], -self.rch[v][b][c], -self.con[v][a][c]])

        with RC2(self.wcnf, verbose=0) as solver:
            model = solver.compute()
            if model:
                pos = set(m for m in model if m>0)
                routes = [[], []]
                for v in range(2):
                    curr, visited = 0, {0}
                    for _ in range(self.n):
                        for j in range(self.n):
                            if j not in visited and self.con[v][curr][j] in pos:
                                if self.l2g[j]!=0: routes[v].append(self.l2g[j])
                                visited.add(j); curr=j; break
                
                if sum(self.dems[c] for c in routes[0]) <= self.cap and \
                   sum(self.dems[c] for c in routes[1]) <= self.cap:
                    return routes[0], routes[1], solver.cost
        
        return self.c1, self.c2, float('inf')

def calculate_route_cost(route, distances):
    if not route: return 0
    c = distances[0, route[0]]
    for i in range(len(route)-1): c += distances[route[i], route[i+1]]
    c += distances[route[-1], 0]
    return int(c)

# ============================================================================
# PART 2: METAHEURISTIC (LNS, GLS, ILS)
# ============================================================================

def lns_perturbation(routes, distances, demands, capacity, destroy_pct=0.3):
    routes = {k: list(v) for k, v in routes.items()}
    all_c = []
    for r in routes.values(): all_c.extend(r)
    if len(all_c) < 5: return routes
    
    n_rem = max(2, int(len(all_c) * destroy_pct))
    removed = random.sample(all_c, n_rem)
    
    for k in routes:
        routes[k] = [c for c in routes[k] if c not in removed]
        
    for c in removed:
        best_cost, best_k, best_pos = float('inf'), -1, -1
        for k, r in routes.items():
            if sum(demands[x] for x in r) + demands[c] > capacity: continue
            for i in range(len(r)+1):
                p = r[i-1] if i>0 else 0
                n = r[i] if i<len(r) else 0
                cost = distances[p,c] + distances[c,n] - distances[p,n]
                if cost < best_cost: best_cost, best_k, best_pos = cost, k, i
        
        if best_k != -1:
            routes[best_k].insert(best_pos, c)
        else:
            new_k = max(routes.keys()) + 1 if routes else 0
            routes[new_k] = [c]
    return routes

def inter_route_local_search(routes, distances, demands, capacity):
    improved = True
    while improved:
        improved = False
        for r1 in list(routes.keys()):
            for i, c in enumerate(routes[r1]):
                if not routes[r1]: continue
                for r2 in list(routes.keys()):
                    if r1 == r2: continue
                    if sum(demands[x] for x in routes[r2]) + demands[c] > capacity: continue
                    
                    r1_vec = routes[r1]
                    p1 = r1_vec[i-1] if i>0 else 0
                    n1 = r1_vec[i+1] if i<len(r1_vec)-1 else 0
                    rem_gain = distances[p1,c] + distances[c,n1] - distances[p1,n1]
                    
                    best_ins, best_pos = float('inf'), -1
                    r2_vec = routes[r2]
                    for j in range(len(r2_vec)+1):
                        p2 = r2_vec[j-1] if j>0 else 0
                        n2 = r2_vec[j] if j<len(r2_vec) else 0
                        ins_cost = distances[p2,c] + distances[c,n2] - distances[p2,n2]
                        if ins_cost < best_ins: best_ins, best_pos = ins_cost, j
                    
                    if best_ins < rem_gain:
                        routes[r1].pop(i)
                        routes[r2].insert(best_pos, c)
                        improved = True; break
                if improved: break
            if improved: break
    
    total = sum(calculate_route_cost(r, distances) for r in routes.values())
    return routes, total

def guided_local_search(routes, distances, demands, capacity, time_limit=30):
    print("\n=== Guided Local Search (GLS) ===")
    start = time.time()
    curr_routes, best_cost = inter_route_local_search(routes, distances, demands, capacity)
    best_routes = {k:list(v) for k,v in curr_routes.items()}
    
    n = distances.shape[0]
    penalties = np.zeros((n, n))
    
    iteration = 0
    while time.time() - start < time_limit:
        iteration += 1
        max_util, edges = -1, []
        for r in curr_routes.values():
            if not r: continue
            tour = [0] + r + [0]
            for i in range(len(tour)-1):
                u, v = tour[i], tour[i+1]
                util = distances[u,v] / (1 + penalties[u,v])
                if util > max_util: max_util, edges = util, [(u,v)]
                elif abs(util - max_util) < 1e-6: edges.append((u,v))
        
        for u,v in edges: penalties[u,v] += 1; penalties[v,u] += 1
        
        curr_routes = lns_perturbation(curr_routes, distances, demands, capacity, 0.2)
        curr_routes, cost = inter_route_local_search(curr_routes, distances, demands, capacity)
        
        if cost < best_cost:
            print(f"  GLS Iter {iteration}: New Best {cost}")
            best_cost = cost
            best_routes = {k:list(v) for k,v in curr_routes.items()}
            
    return best_routes, best_cost

# ============================================================================
# PART 3: MAIN LOGIC
# ============================================================================

def optimize_all_routes(routes, distances, demands, capacity):
    print("\n=== MaxSAT Single Route Optimization ===")
    opt_routes = {}
    total = 0
    for v, r in routes.items():
        if not r: continue
        # Skip if too large
        if len(r) > 11: 
            opt_routes[v] = r
            total += calculate_route_cost(r, distances)
            print("!", end="", flush=True) 
            continue
            
        print(".", end="", flush=True) 
        
        opt = SingleRouteOptimizer(r, distances)
        new_r, cost = opt.optimize()
        
        # [FIX] Kiểm tra chặt chẽ cost trả về
        if sum(demands[c] for c in new_r) <= capacity and cost > 0:
            opt_routes[v] = new_r
            total += cost
        else:
            opt_routes[v] = r
            total += calculate_route_cost(r, distances)
            
    print("\n  Single Route Optimization done.")
    return opt_routes, total

def pairwise_optimize(routes, distances, demands, capacity):
    print("\n=== MaxSAT Pairwise Optimization ===")
    improved = True
    start_time = time.time()
    TIME_LIMIT = 60 
    
    while improved:
        if time.time() - start_time > TIME_LIMIT:
            print("  [TIMEOUT] Pairwise optimization limit reached.")
            break
            
        improved = False
        ids = list(routes.keys())
        pairs = list(combinations(ids, 2))
        
        for i, j in pairs:
            if time.time() - start_time > TIME_LIMIT: break
            if i not in routes or j not in routes: continue
            if not routes[i] or not routes[j]: continue
            if len(routes[i]) + len(routes[j]) > 9: continue
            
            print(".", end="", flush=True)
            
            curr_cost = calculate_route_cost(routes[i], distances) + calculate_route_cost(routes[j], distances)
            opt = PairwiseRouteOptimizer(routes[i], routes[j], distances, demands, capacity)
            r1, r2, new_cost = opt.optimize()
            
            if new_cost < curr_cost:
                print(f"\n  Merged {i}&{j}: {curr_cost}->{new_cost}")
                routes[i], routes[j] = r1, r2
                improved = True
                break
    
    print("\n  Pairwise done.")
    return routes

def validate_and_print_result(routes, distances, demands, capacity):
    print("\n" + "="*30)
    print("      FINAL VALIDATION")
    print("="*30)
    
    final_total = 0
    visited = set()
    is_valid = True
    
    for rid, r in routes.items():
        if not r: continue
        
        # Check Load
        load = sum(demands[c] for c in r)
        if load > capacity:
            print(f"[FAIL] Route {rid}: Overload ({load}/{capacity})")
            is_valid = False
            
        # Check Duplicate
        for c in r:
            if c in visited:
                print(f"[FAIL] Customer {c} visited multiple times!")
                is_valid = False
            visited.add(c)
        
        # Recalculate cost independently
        c = calculate_route_cost(r, distances)
        final_total += c
        
    status = "VALID" if is_valid else "INVALID"
    print(f"Status: {status}")
    print(f"Recalculated Total Cost: {final_total}")
    return final_total

def solve(filepath):
    cvrp = Instance(filepath); cvrp.load()
    n_veh = int(re.search(r'-k(\d+)', filepath).group(1))
    dists, dems, cap = cvrp.distances, np.array([0] + cvrp.demands), cvrp.capacity
    
    print("Step 1: C&W...")
    _, cw = ClarkeWright.run(cvrp, n_veh)
    routes = {i: list(r.value) for i, (_, r) in enumerate(cw.items())}
    
    print("Step 2: ILS...")
    best_cost, best_routes = float('inf'), None
    start = time.time()
    for i in range(10):
        if time.time() - start > 120: break
        curr = routes if i==0 else lns_perturbation(best_routes, dists, dems, cap, 0.4)
        curr, cost = inter_route_local_search(curr, dists, dems, cap)
        if cost < best_cost:
            best_cost = cost; best_routes = {k:list(v) for k,v in curr.items()}
            print(f"  ILS Best: {best_cost}")
            
    best_routes, best_cost = guided_local_search(best_routes, dists, dems, cap, time_limit=20)
    best_routes, _ = optimize_all_routes(best_routes, dists, dems, cap)
    best_routes = pairwise_optimize(best_routes, dists, dems, cap)
    
    # Final check
    final_cost = validate_and_print_result(best_routes, dists, dems, cap)
    return best_routes, final_cost

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python route_optimizer_v2.py <instance_path>")
    else:
        solve(sys.argv[1])