import numpy as np
from typing import List, Dict, Tuple
from itertools import combinations
from core.cost_utils import calculate_route_cost

def relocate_search(routes: Dict[int, List[int]], distances: np.ndarray,
                    demands: np.ndarray, capacity: int) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    Relocate: Move one customer from one route to another.
    Returns the best improvement found.
    """
    best_saving = 0
    best_move = None
    
    route_ids = list(routes.keys())
    
    for r1 in route_ids:
        for pos1, customer in enumerate(routes[r1]):
            # Calculate removal cost from r1
            route1 = routes[r1]
            if len(route1) == 1:
                remove_cost = distances[0, customer] + distances[customer, 0]
                new_cost_r1 = 0
            else:
                if pos1 == 0:
                    prev1 = 0
                    next1 = route1[1]
                elif pos1 == len(route1) - 1:
                    prev1 = route1[pos1 - 1]
                    next1 = 0
                else:
                    prev1 = route1[pos1 - 1]
                    next1 = route1[pos1 + 1]
                
                old_links = distances[prev1, customer] + distances[customer, next1]
                new_link = distances[prev1, next1]
                remove_cost = old_links - new_link
                
                new_route1 = route1[:pos1] + route1[pos1+1:]
                new_cost_r1 = calculate_route_cost(new_route1, distances)
            
            # Try inserting into other routes
            for r2 in route_ids:
                if r1 == r2:
                    continue
                
                route2 = routes[r2]
                demand_r2 = sum(demands[c] for c in route2)
                
                if demand_r2 + demands[customer] > capacity:
                    continue
                
                # Find best insertion position
                for pos2 in range(len(route2) + 1):
                    if pos2 == 0:
                        prev2 = 0
                        next2 = route2[0] if route2 else 0
                    elif pos2 == len(route2):
                        prev2 = route2[-1]
                        next2 = 0
                    else:
                        prev2 = route2[pos2 - 1]
                        next2 = route2[pos2]
                    
                    insert_cost = distances[prev2, customer] + distances[customer, next2] - distances[prev2, next2]
                    
                    saving = remove_cost - insert_cost
                    
                    if saving > best_saving:
                        best_saving = saving
                        best_move = (r1, pos1, r2, pos2, customer)
    
    if best_move:
        r1, pos1, r2, pos2, customer = best_move
        # Apply move
        new_routes = {k: list(v) for k, v in routes.items()}
        new_routes[r1] = routes[r1][:pos1] + routes[r1][pos1+1:]
        new_routes[r2] = routes[r2][:pos2] + [customer] + routes[r2][pos2:]
        
        return new_routes, best_saving, True
    
    return routes, 0, False


def exchange_search(routes: Dict[int, List[int]], distances: np.ndarray,
                    demands: np.ndarray, capacity: int) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    Exchange: Swap one customer from route1 with one customer from route2.
    Returns the best improvement found.
    """
    best_saving = 0
    best_move = None
    
    route_ids = list(routes.keys())
    
    for r1, r2 in combinations(route_ids, 2):
        for pos1, c1 in enumerate(routes[r1]):
            for pos2, c2 in enumerate(routes[r2]):
                # Check capacity
                demand_r1 = sum(demands[c] for c in routes[r1]) - demands[c1] + demands[c2]
                demand_r2 = sum(demands[c] for c in routes[r2]) - demands[c2] + demands[c1]
                
                if demand_r1 > capacity or demand_r2 > capacity:
                    continue
                
                # Calculate cost change for r1
                route1 = routes[r1]
                prev1 = route1[pos1 - 1] if pos1 > 0 else 0
                next1 = route1[pos1 + 1] if pos1 < len(route1) - 1 else 0
                old_cost1 = distances[prev1, c1] + distances[c1, next1]
                new_cost1 = distances[prev1, c2] + distances[c2, next1]
                
                # Calculate cost change for r2
                route2 = routes[r2]
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
        # Apply move
        new_routes = {k: list(v) for k, v in routes.items()}
        new_routes[r1][pos1] = c2
        new_routes[r2][pos2] = c1
        
        return new_routes, best_saving, True
    
    return routes, 0, False


def intra_route_2opt(routes: Dict[int, List[int]], distances: np.ndarray) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    2-opt within each route: reverse a segment to improve.
    """
    best_saving = 0
    best_move = None
    
    for r_id, route in routes.items():
        if len(route) < 3:
            continue
        
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                # Current edges: (prev_i, route[i]), (route[j], next_j)
                prev_i = route[i - 1] if i > 0 else 0
                next_j = route[j + 1] if j < len(route) - 1 else 0
                
                # After reversing [i, j]: (prev_i, route[j]), (route[i], next_j)
                old_cost = distances[prev_i, route[i]] + distances[route[j], next_j]
                new_cost = distances[prev_i, route[j]] + distances[route[i], next_j]
                
                saving = old_cost - new_cost
                if saving > best_saving:
                    best_saving = saving
                    best_move = (r_id, i, j)
    
    if best_move:
        r_id, i, j = best_move
        new_routes = {k: list(v) for k, v in routes.items()}
        # Reverse segment [i, j]
        new_routes[r_id] = routes[r_id][:i] + routes[r_id][i:j+1][::-1] + routes[r_id][j+1:]
        return new_routes, best_saving, True
    
    return routes, 0, False


def intra_route_3opt(routes: Dict[int, List[int]], distances: np.ndarray) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    3-opt within each route: reconnect 3 edges in the best way.
    Much stronger than 2-opt but slower.
    """
    best_saving = 0
    best_move = None
    
    for r_id, route in routes.items():
        if len(route) < 5:
            continue
        
        n = len(route)
        for i in range(n - 4):
            for j in range(i + 2, n - 2):
                for k in range(j + 2, n):
                    # Get nodes
                    prev_i = route[i - 1] if i > 0 else 0
                    node_i = route[i]
                    node_i1 = route[i + 1] if i + 1 < n else 0
                    
                    node_j = route[j]
                    node_j1 = route[j + 1] if j + 1 < n else 0
                    
                    node_k = route[k]
                    next_k = route[k + 1] if k + 1 < n else 0
                    if k == n - 1:
                        next_k = 0  # Return to depot
                    
                    # Original cost of 3 edges
                    old_cost = (distances[prev_i, node_i] + 
                               distances[route[j], node_j1] +
                               distances[node_k, next_k])
                    
                    # Try different reconnection options
                    # Option 1: reverse segment [i, j]
                    seg1 = route[i:j+1][::-1]
                    seg2 = route[j+1:k+1]
                    new_route = route[:i] + seg1 + seg2 + route[k+1:]
                    new_cost1 = calculate_route_cost(new_route, distances)
                    
                    # Option 2: reverse segment [j+1, k]
                    seg1 = route[i:j+1]
                    seg2 = route[j+1:k+1][::-1]
                    new_route = route[:i] + seg1 + seg2 + route[k+1:]
                    new_cost2 = calculate_route_cost(new_route, distances)
                    
                    # Option 3: swap segments
                    seg1 = route[j+1:k+1]
                    seg2 = route[i:j+1]
                    new_route = route[:i] + seg1 + seg2 + route[k+1:]
                    new_cost3 = calculate_route_cost(new_route, distances)
                    
                    original_cost = calculate_route_cost(route, distances)
                    
                    for opt, new_cost in [(1, new_cost1), (2, new_cost2), (3, new_cost3)]:
                        saving = original_cost - new_cost
                        if saving > best_saving:
                            best_saving = saving
                            best_move = (r_id, i, j, k, opt)
    
    if best_move:
        r_id, i, j, k, opt = best_move
        route = routes[r_id]
        new_routes = {key: list(val) for key, val in routes.items()}
        
        if opt == 1:
            seg1 = route[i:j+1][::-1]
            seg2 = route[j+1:k+1]
            new_routes[r_id] = route[:i] + seg1 + seg2 + route[k+1:]
        elif opt == 2:
            seg1 = route[i:j+1]
            seg2 = route[j+1:k+1][::-1]
            new_routes[r_id] = route[:i] + seg1 + seg2 + route[k+1:]
        else:
            seg1 = route[j+1:k+1]
            seg2 = route[i:j+1]
            new_routes[r_id] = route[:i] + seg1 + seg2 + route[k+1:]
        
        return new_routes, best_saving, True
    
    return routes, 0, False


def cross_exchange(routes: Dict[int, List[int]], distances: np.ndarray,
                   demands: np.ndarray, capacity: int,
                   max_seg_len: int = 3) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    Cross-Exchange: Exchange segments between two routes.
    """
    best_saving = 0
    best_move = None
    
    route_ids = list(routes.keys())
    
    for r1, r2 in combinations(route_ids, 2):
        route1, route2 = routes[r1], routes[r2]
        if not route1 or not route2:
            continue
        
        for len1 in range(1, min(len(route1) + 1, max_seg_len + 1)):
            for len2 in range(1, min(len(route2) + 1, max_seg_len + 1)):
                for i in range(len(route1) - len1 + 1):
                    for j in range(len(route2) - len2 + 1):
                        seg1 = route1[i:i + len1]
                        seg2 = route2[j:j + len2]
                        
                        # Check capacity
                        demand1 = sum(demands[c] for c in route1) - sum(demands[c] for c in seg1) + sum(demands[c] for c in seg2)
                        demand2 = sum(demands[c] for c in route2) - sum(demands[c] for c in seg2) + sum(demands[c] for c in seg1)
                        
                        if demand1 > capacity or demand2 > capacity:
                            continue
                        
                        # Calculate cost change
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


def or_opt_search(routes: Dict[int, List[int]], distances: np.ndarray,
                  demands: np.ndarray, capacity: int, 
                  seq_len: int = 2) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    Or-Opt: Move a sequence of consecutive customers to another route.
    """
    best_saving = 0
    best_move = None
    
    route_ids = list(routes.keys())
    
    for r1 in route_ids:
        route1 = routes[r1]
        if len(route1) < seq_len:
            continue
            
        for pos1 in range(len(route1) - seq_len + 1):
            # Sequence to move
            seq = route1[pos1:pos1 + seq_len]
            seq_demand = sum(demands[c] for c in seq)
            
            # Calculate removal cost
            prev1 = route1[pos1 - 1] if pos1 > 0 else 0
            next1 = route1[pos1 + seq_len] if pos1 + seq_len < len(route1) else 0
            
            old_links = distances[prev1, seq[0]] + distances[seq[-1], next1]
            new_link = distances[prev1, next1]
            remove_saving = old_links - new_link
            
            # Try inserting into other routes
            for r2 in route_ids:
                if r1 == r2:
                    continue
                
                route2 = routes[r2]
                demand_r2 = sum(demands[c] for c in route2)
                
                if demand_r2 + seq_demand > capacity:
                    continue
                
                # Find best insertion position
                for pos2 in range(len(route2) + 1):
                    prev2 = route2[pos2 - 1] if pos2 > 0 else 0
                    next2 = route2[pos2] if pos2 < len(route2) else 0
                    
                    insert_cost = (distances[prev2, seq[0]] + distances[seq[-1], next2] 
                                   - distances[prev2, next2])
                    
                    saving = remove_saving - insert_cost
                    
                    if saving > best_saving:
                        best_saving = saving
                        best_move = (r1, pos1, seq_len, r2, pos2, seq)
    
    if best_move:
        r1, pos1, seq_len, r2, pos2, seq = best_move
        # Apply move
        new_routes = {k: list(v) for k, v in routes.items()}
        new_routes[r1] = routes[r1][:pos1] + routes[r1][pos1 + seq_len:]
        new_routes[r2] = routes[r2][:pos2] + list(seq) + routes[r2][pos2:]
        
        return new_routes, best_saving, True
    
    return routes, 0, False


def inter_route_local_search(routes: Dict[int, List[int]], distances: np.ndarray,
                             demands: np.ndarray, capacity: int,
                             max_iterations: int = 100, verbose: bool = True) -> Tuple[Dict[int, List[int]], int]:
    """
    Apply Relocate and Exchange moves until no improvement.
    """
    if verbose:
        print("\n=== Inter-route Local Search (Relocate + Exchange) ===")
    
    improved = True
    iteration = 0
    total_saving = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try relocate
        routes, saving, did_improve = relocate_search(routes, distances, demands, capacity)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  Relocate: -{saving}")
            continue
        
        # Try exchange
        routes, saving, did_improve = exchange_search(routes, distances, demands, capacity)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  Exchange: -{saving}")
            continue
        
        # Try Or-Opt (move pairs)
        routes, saving, did_improve = or_opt_search(routes, distances, demands, capacity, seq_len=2)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  Or-Opt(2): -{saving}")
            continue
        
        # Try Or-Opt (move triples)
        routes, saving, did_improve = or_opt_search(routes, distances, demands, capacity, seq_len=3)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  Or-Opt(3): -{saving}")
            continue
        
        # Try Cross-Exchange
        routes, saving, did_improve = cross_exchange(routes, distances, demands, capacity)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  Cross-Exchange: -{saving}")
            continue
        
        # Try intra-route 2-opt
        routes, saving, did_improve = intra_route_2opt(routes, distances)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  2-opt: -{saving}")
            continue
        
        # Try intra-route 3-opt (stronger)
        routes, saving, did_improve = intra_route_3opt(routes, distances)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  3-opt: -{saving}")
    
    total_cost = sum(calculate_route_cost(r, distances) for r in routes.values())
    if verbose:
        print(f"  Total improvement: -{total_saving}")
    
    return routes, total_cost
