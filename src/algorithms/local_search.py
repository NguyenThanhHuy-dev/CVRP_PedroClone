import numpy as np
from typing import Dict, List, Tuple
from itertools import combinations
from src.common.utils import calculate_route_cost

def relocate_search(routes: Dict[int, List[int]], distances: np.ndarray,
                    demands: np.ndarray, capacity: int) -> Tuple[Dict[int, List[int]], int, bool]:
    best_saving = 0
    best_move = None
    route_ids = list(routes.keys())
    
    for r1 in route_ids:
        for pos1, customer in enumerate(routes[r1]):
            route1 = routes[r1]
            if len(route1) == 1:
                remove_cost = distances[0, customer] + distances[customer, 0]
            else:
                prev1 = route1[pos1 - 1] if pos1 > 0 else 0
                next1 = route1[pos1 + 1] if pos1 < len(route1) - 1 else 0
                old_links = distances[prev1, customer] + distances[customer, next1]
                new_link = distances[prev1, next1]
                remove_cost = old_links - new_link
            
            for r2 in route_ids:
                if r1 == r2: continue
                route2 = routes[r2]
                if sum(demands[c] for c in route2) + demands[customer] > capacity: continue
                
                for pos2 in range(len(route2) + 1):
                    prev2 = route2[pos2 - 1] if pos2 > 0 else 0
                    next2 = route2[pos2] if pos2 < len(route2) else 0
                    insert_cost = distances[prev2, customer] + distances[customer, next2] - distances[prev2, next2]
                    saving = remove_cost - insert_cost
                    
                    if saving > best_saving:
                        best_saving = saving
                        best_move = (r1, pos1, r2, pos2, customer)
    
    if best_move:
        r1, pos1, r2, pos2, customer = best_move
        new_routes = {k: list(v) for k, v in routes.items()}
        new_routes[r1] = routes[r1][:pos1] + routes[r1][pos1+1:]
        new_routes[r2] = routes[r2][:pos2] + [customer] + routes[r2][pos2:]
        return new_routes, best_saving, True
    return routes, 0, False

def exchange_search(routes: Dict[int, List[int]], distances: np.ndarray,
                    demands: np.ndarray, capacity: int) -> Tuple[Dict[int, List[int]], int, bool]:
    best_saving = 0
    best_move = None
    route_ids = list(routes.keys())
    
    for r1, r2 in combinations(route_ids, 2):
        for pos1, c1 in enumerate(routes[r1]):
            for pos2, c2 in enumerate(routes[r2]):
                demand_r1 = sum(demands[c] for c in routes[r1]) - demands[c1] + demands[c2]
                demand_r2 = sum(demands[c] for c in routes[r2]) - demands[c2] + demands[c1]
                if demand_r1 > capacity or demand_r2 > capacity: continue
                
                route1, route2 = routes[r1], routes[r2]
                prev1 = route1[pos1 - 1] if pos1 > 0 else 0
                next1 = route1[pos1 + 1] if pos1 < len(route1) - 1 else 0
                old_cost1 = distances[prev1, c1] + distances[c1, next1]
                new_cost1 = distances[prev1, c2] + distances[c2, next1]
                
                prev2 = route2[pos2 - 1] if pos2 > 0 else 0
                next2 = route2[pos2 + 1] if pos2 < len(route2) - 1 else 0
                old_cost2 = distances[prev2, c2] + distances[c2, next2]
                new_cost2 = distances[prev2, c1] + distances[c1, next2]
                
                saving = (old_cost1 + old_cost2) - (new_cost1 + new_cost2)
                if saving > best_saving:
                    best_saving = saving
                    best_move = (r1, pos1, c1, r2, pos2, c2)
    
    if best_move:
        r1, pos1, c1, r2, pos2, c2 = best_move
        new_routes = {k: list(v) for k, v in routes.items()}
        new_routes[r1][pos1] = c2
        new_routes[r2][pos2] = c1
        return new_routes, best_saving, True
    return routes, 0, False

def intra_route_2opt(routes: Dict[int, List[int]], distances: np.ndarray) -> Tuple[Dict[int, List[int]], int, bool]:
    best_saving = 0
    best_move = None
    
    for r_id, route in routes.items():
        if len(route) < 3: continue
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                prev_i = route[i - 1] if i > 0 else 0
                next_j = route[j + 1] if j < len(route) - 1 else 0
                old_cost = distances[prev_i, route[i]] + distances[route[j], next_j]
                new_cost = distances[prev_i, route[j]] + distances[route[i], next_j]
                
                saving = old_cost - new_cost
                if saving > best_saving:
                    best_saving = saving
                    best_move = (r_id, i, j)
    
    if best_move:
        r_id, i, j = best_move
        new_routes = {k: list(v) for k, v in routes.items()}
        new_routes[r_id] = routes[r_id][:i] + routes[r_id][i:j+1][::-1] + routes[r_id][j+1:]
        return new_routes, best_saving, True
    return routes, 0, False

def intra_route_3opt(routes: Dict[int, List[int]], distances: np.ndarray) -> Tuple[Dict[int, List[int]], int, bool]:
    best_saving = 0
    best_move = None
    
    for r_id, route in routes.items():
        if len(route) < 5: continue
        n = len(route)
        for i in range(n - 4):
            for j in range(i + 2, n - 2):
                for k in range(j + 2, n):
                    prev_i = route[i - 1] if i > 0 else 0
                    node_i = route[i]
                    node_j1 = route[j + 1] if j + 1 < n else 0
                    node_k = route[k]
                    next_k = route[k + 1] if k + 1 < n else 0
                    
                    new_routes_attempts = []
                    # Option 1: reverse [i, j]
                    seg1, seg2 = route[i:j+1][::-1], route[j+1:k+1]
                    new_routes_attempts.append((1, route[:i] + seg1 + seg2 + route[k+1:]))
                    # Option 2: reverse [j+1, k]
                    seg1, seg2 = route[i:j+1], route[j+1:k+1][::-1]
                    new_routes_attempts.append((2, route[:i] + seg1 + seg2 + route[k+1:]))
                    # Option 3: swap segments
                    seg1, seg2 = route[j+1:k+1], route[i:j+1]
                    new_routes_attempts.append((3, route[:i] + seg1 + seg2 + route[k+1:]))
                    
                    original_cost = calculate_route_cost(route, distances)
                    for opt, new_r in new_routes_attempts:
                        new_c = calculate_route_cost(new_r, distances)
                        saving = original_cost - new_c
                        if saving > best_saving:
                            best_saving = saving
                            best_move = (r_id, i, j, k, opt)

    if best_move:
        r_id, i, j, k, opt = best_move
        route = routes[r_id]
        new_routes = {key: list(val) for key, val in routes.items()}
        if opt == 1:
            seg1, seg2 = route[i:j+1][::-1], route[j+1:k+1]
        elif opt == 2:
            seg1, seg2 = route[i:j+1], route[j+1:k+1][::-1]
        else:
            seg1, seg2 = route[j+1:k+1], route[i:j+1]
        new_routes[r_id] = route[:i] + seg1 + seg2 + route[k+1:]
        return new_routes, best_saving, True
    return routes, 0, False

def or_opt_search(routes: Dict[int, List[int]], distances: np.ndarray,
                  demands: np.ndarray, capacity: int, seq_len: int = 2) -> Tuple[Dict[int, List[int]], int, bool]:
    best_saving = 0
    best_move = None
    route_ids = list(routes.keys())
    
    for r1 in route_ids:
        route1 = routes[r1]
        if len(route1) < seq_len: continue
        for pos1 in range(len(route1) - seq_len + 1):
            seq = route1[pos1:pos1 + seq_len]
            seq_demand = sum(demands[c] for c in seq)
            
            prev1 = route1[pos1 - 1] if pos1 > 0 else 0
            next1 = route1[pos1 + seq_len] if pos1 + seq_len < len(route1) else 0
            remove_saving = distances[prev1, seq[0]] + distances[seq[-1], next1] - distances[prev1, next1]
            
            for r2 in route_ids:
                if r1 == r2: continue
                route2 = routes[r2]
                if sum(demands[c] for c in route2) + seq_demand > capacity: continue
                
                for pos2 in range(len(route2) + 1):
                    prev2 = route2[pos2 - 1] if pos2 > 0 else 0
                    next2 = route2[pos2] if pos2 < len(route2) else 0
                    insert_cost = distances[prev2, seq[0]] + distances[seq[-1], next2] - distances[prev2, next2]
                    saving = remove_saving - insert_cost
                    if saving > best_saving:
                        best_saving = saving
                        best_move = (r1, pos1, seq_len, r2, pos2, seq)

    if best_move:
        r1, pos1, seq_len, r2, pos2, seq = best_move
        new_routes = {k: list(v) for k, v in routes.items()}
        new_routes[r1] = routes[r1][:pos1] + routes[r1][pos1 + seq_len:]
        new_routes[r2] = routes[r2][:pos2] + list(seq) + routes[r2][pos2:]
        return new_routes, best_saving, True
    return routes, 0, False

def cross_exchange(routes: Dict[int, List[int]], distances: np.ndarray,
                   demands: np.ndarray, capacity: int, max_seg_len: int = 3) -> Tuple[Dict[int, List[int]], int, bool]:
    best_saving = 0
    best_move = None
    route_ids = list(routes.keys())
    
    for r1, r2 in combinations(route_ids, 2):
        route1, route2 = routes[r1], routes[r2]
        if not route1 or not route2: continue
        
        for len1 in range(1, min(len(route1) + 1, max_seg_len + 1)):
            for len2 in range(1, min(len(route2) + 1, max_seg_len + 1)):
                for i in range(len(route1) - len1 + 1):
                    for j in range(len(route2) - len2 + 1):
                        seg1 = route1[i:i + len1]
                        seg2 = route2[j:j + len2]
                        
                        d1 = sum(demands[c] for c in route1) - sum(demands[c] for c in seg1) + sum(demands[c] for c in seg2)
                        d2 = sum(demands[c] for c in route2) - sum(demands[c] for c in seg2) + sum(demands[c] for c in seg1)
                        if d1 > capacity or d2 > capacity: continue
                        
                        prev1 = route1[i - 1] if i > 0 else 0
                        next1 = route1[i + len1] if i + len1 < len(route1) else 0
                        prev2 = route2[j - 1] if j > 0 else 0
                        next2 = route2[j + len2] if j + len2 < len(route2) else 0
                        
                        old_cost = (distances[prev1, seg1[0]] + distances[seg1[-1], next1] +
                                   distances[prev2, seg2[0]] + distances[seg2[-1], next2])
                        new_cost = (distances[prev1, seg2[0]] + distances[seg2[-1], next1] +
                                   distances[prev2, seg1[0]] + distances[seg1[-1], next2])
                        
                        saving = old_cost - new_cost
                        if saving > best_saving:
                            best_saving = saving
                            best_move = (r1, i, len1, r2, j, len2)
    
    if best_move:
        r1, i, len1, r2, j, len2 = best_move
        new_routes = {k: list(v) for k, v in routes.items()}
        seg1 = routes[r1][i:i + len1]
        seg2 = routes[r2][j:j + len2]
        new_routes[r1] = routes[r1][:i] + seg2 + routes[r1][i + len1:]
        new_routes[r2] = routes[r2][:j] + seg1 + routes[r2][j + len2:]
        return new_routes, best_saving, True
    return routes, 0, False

def simple_local_search(routes, distances, demands, capacity):
    """Local search rút gọn của V2 (chỉ Relocate đơn giản)"""
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

def comprehensive_local_search(routes, distances, demands, capacity):
    """Local Search toàn diện của V1: Chạy liên hoàn các move"""
    improved = True
    iteration = 0
    max_iterations = 100
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # 1. Relocate
        routes, _, did_improve = relocate_search(routes, distances, demands, capacity)
        if did_improve: improved = True; continue
        
        # 2. Exchange
        routes, _, did_improve = exchange_search(routes, distances, demands, capacity)
        if did_improve: improved = True; continue
        
        # 3. Or-Opt (len=2)
        routes, _, did_improve = or_opt_search(routes, distances, demands, capacity, seq_len=2)
        if did_improve: improved = True; continue
        
        # 4. Or-Opt (len=3)
        routes, _, did_improve = or_opt_search(routes, distances, demands, capacity, seq_len=3)
        if did_improve: improved = True; continue
        
        # 5. Cross-Exchange
        routes, _, did_improve = cross_exchange(routes, distances, demands, capacity)
        if did_improve: improved = True; continue
        
        # 6. Intra-Route 2-Opt
        routes, _, did_improve = intra_route_2opt(routes, distances)
        if did_improve: improved = True; continue
        
        # 7. Intra-Route 3-Opt (Khá chậm, nhưng mạnh)
        routes, _, did_improve = intra_route_3opt(routes, distances)
        if did_improve: improved = True; continue

    total_cost = sum(calculate_route_cost(r, distances) for r in routes.values())
    return routes, total_cost