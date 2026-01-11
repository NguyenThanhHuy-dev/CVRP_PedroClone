import time
import random
import math
import numpy as np
from typing import Dict, List, Tuple

from src.common.utils import calculate_route_cost
# --- SỬA LỖI Ở ĐÂY: Chỉ import những hàm thực sự có trong local_search.py ---
from src.algorithms.local_search import (
    simple_local_search, 
    comprehensive_local_search
)

# Tạo Alias: Gán tên cũ (inter_route_local_search) bằng hàm mới (simple_local_search)
# để các hàm bên dưới (như guided_local_search) vẫn chạy đúng mà không cần sửa logic.
inter_route_local_search = simple_local_search 

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

def guided_local_search(routes, distances, demands, capacity, time_limit=30):
    print("\n=== Guided Local Search (GLS) ===")
    start = time.time()
    # Dùng simple_local_search (qua alias inter_route_local_search)
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

def perturbation(routes: Dict[int, List[int]], distances: np.ndarray,
                 demands: np.ndarray, capacity: int, strength: int = 5) -> Dict[int, List[int]]:
    routes = {k: list(v) for k, v in routes.items()}
    route_ids = [k for k in routes if routes[k]]
    
    moves_done = 0
    attempts = 0
    max_attempts = strength * 10
    
    while moves_done < strength and attempts < max_attempts:
        attempts += 1
        if len(route_ids) < 2: break
        r1 = random.choice(route_ids)
        if not routes[r1]: continue
        
        pos1 = random.randint(0, len(routes[r1]) - 1)
        customer = routes[r1][pos1]
        
        other_routes = [r for r in route_ids if r != r1 and routes[r]]
        if not other_routes: continue
        r2 = random.choice(other_routes)
            
        demand_r2 = sum(demands[c] for c in routes[r2])
        if demand_r2 + demands[customer] <= capacity:
            routes[r1] = routes[r1][:pos1] + routes[r1][pos1+1:]
            pos2 = random.randint(0, len(routes[r2]))
            routes[r2] = routes[r2][:pos2] + [customer] + routes[r2][pos2:]
            moves_done += 1
            route_ids = [k for k in routes if routes[k]]
    return routes

def iterated_local_search(routes: Dict[int, List[int]], distances: np.ndarray,
                          demands: np.ndarray, capacity: int,
                          max_no_improve: int = 20, time_limit: float = 60.0,
                          use_sa: bool = False) -> Tuple[Dict[int, List[int]], int]:
    """Hàm ILS cho V1: Dùng comprehensive_local_search"""
    start_time = time.time()
    
    # Khởi tạo Local Optimum
    best_routes, best_cost = comprehensive_local_search(routes, distances, demands, capacity)
    current_routes, current_cost = best_routes, best_cost
    
    no_improve = 0
    iteration = 0
    temp = best_cost * 0.05
    cooling_rate = 0.95
    
    while no_improve < max_no_improve:
        if time.time() - start_time > time_limit: break
        iteration += 1
        
        # Perturb
        strength = 3 + (no_improve // 5)
        perturbed = perturbation(current_routes, distances, demands, capacity, strength=strength)
        
        # Local Search (V1 dùng bản comprehensive)
        new_routes, new_cost = comprehensive_local_search(perturbed, distances, demands, capacity)
        
        # Acceptance
        accept = False
        if new_cost < current_cost:
            accept = True
        elif use_sa and temp > 0.1:
            delta = new_cost - current_cost
            prob = math.exp(-delta / temp)
            if random.random() < prob: accept = True
        
        if accept:
            current_routes = new_routes
            current_cost = new_cost
            if new_cost < best_cost:
                best_routes = new_routes
                best_cost = new_cost
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        if use_sa: temp *= cooling_rate
            
    return best_routes, best_cost

def multi_start_ils(routes: Dict[int, List[int]], distances: np.ndarray,
                    demands: np.ndarray, capacity: int,
                    n_restarts: int = 5, time_limit: float = 60.0) -> Tuple[Dict[int, List[int]], int]:
    print("\n=== Multi-Start ILS ===")
    start_time = time.time()
    best_routes = routes
    best_cost = sum(calculate_route_cost(r, distances) for r in routes.values())
    
    time_per_restart = time_limit / n_restarts
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit: break
        
        random.seed(restart * 42 + 7)
        if restart > 0:
            perturbed = perturbation(routes, distances, demands, capacity, strength=10)
        else:
            perturbed = {k: list(v) for k, v in routes.items()}
        
        ils_routes, ils_cost = iterated_local_search(
            perturbed, distances, demands, capacity,
            max_no_improve=30, time_limit=time_per_restart, use_sa=False
        )
        
        if ils_cost < best_cost:
            print(f"  Restart {restart + 1}: NEW BEST {ils_cost}")
            best_routes = ils_routes
            best_cost = ils_cost
        else:
            print(f"  Restart {restart + 1}: {ils_cost}")
            
    print(f"  Best across restarts: {best_cost}")
    return best_routes, best_cost