import time
import numpy as np
from typing import List, Dict, Tuple
from core.cost_utils import get_route_edges, calculate_route_cost
from search.local_search import inter_route_local_search

def gls_local_search(routes: Dict[int, List[int]], distances: np.ndarray,
                     demands: np.ndarray, capacity: int,
                     penalties: np.ndarray, lambda_param: float,
                     max_iterations: int = 100) -> Tuple[Dict[int, List[int]], float]:
    """
    Local search using augmented cost function (with penalties).
    Includes relocate, exchange, and 2-opt with augmented cost.
    """
    from itertools import combinations
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try relocate with augmented cost
        best_saving = 0
        best_move = None
        route_ids = list(routes.keys())
        
        for r1 in route_ids:
            for pos1, customer in enumerate(routes[r1]):
                route1 = routes[r1]
                if len(route1) == 1:
                    prev1, next1 = 0, 0
                else:
                    prev1 = route1[pos1 - 1] if pos1 > 0 else 0
                    next1 = route1[pos1 + 1] if pos1 < len(route1) - 1 else 0
                
                # Augmented cost of removing
                old_aug = (distances[prev1, customer] + lambda_param * penalties[prev1, customer] +
                          distances[customer, next1] + lambda_param * penalties[customer, next1])
                new_aug = distances[prev1, next1] + lambda_param * penalties[prev1, next1]
                remove_saving = old_aug - new_aug
                
                for r2 in route_ids:
                    if r1 == r2:
                        continue
                    
                    route2 = routes[r2]
                    demand_r2 = sum(demands[c] for c in route2)
                    if demand_r2 + demands[customer] > capacity:
                        continue
                    
                    for pos2 in range(len(route2) + 1):
                        prev2 = route2[pos2 - 1] if pos2 > 0 else 0
                        next2 = route2[pos2] if pos2 < len(route2) else 0
                        
                        # Augmented cost of inserting
                        old_aug2 = distances[prev2, next2] + lambda_param * penalties[prev2, next2]
                        new_aug2 = (distances[prev2, customer] + lambda_param * penalties[prev2, customer] +
                                   distances[customer, next2] + lambda_param * penalties[customer, next2])
                        insert_cost = new_aug2 - old_aug2
                        
                        saving = remove_saving - insert_cost
                        if saving > best_saving + 1e-6:
                            best_saving = saving
                            best_move = ('relocate', r1, pos1, r2, pos2, customer)
        
        for r1, r2 in combinations(route_ids, 2):
            for pos1, c1 in enumerate(routes[r1]):
                for pos2, c2 in enumerate(routes[r2]):
                    demand_r1 = sum(demands[c] for c in routes[r1]) - demands[c1] + demands[c2]
                    demand_r2 = sum(demands[c] for c in routes[r2]) - demands[c2] + demands[c1]
                    
                    if demand_r1 > capacity or demand_r2 > capacity:
                        continue
                    
                    route1, route2 = routes[r1], routes[r2]
                    prev1 = route1[pos1 - 1] if pos1 > 0 else 0
                    next1 = route1[pos1 + 1] if pos1 < len(route1) - 1 else 0
                    prev2 = route2[pos2 - 1] if pos2 > 0 else 0
                    next2 = route2[pos2 + 1] if pos2 < len(route2) - 1 else 0
                    
                    old_aug = (distances[prev1, c1] + lambda_param * penalties[prev1, c1] +
                              distances[c1, next1] + lambda_param * penalties[c1, next1] +
                              distances[prev2, c2] + lambda_param * penalties[prev2, c2] +
                              distances[c2, next2] + lambda_param * penalties[c2, next2])
                    new_aug = (distances[prev1, c2] + lambda_param * penalties[prev1, c2] +
                              distances[c2, next1] + lambda_param * penalties[c2, next1] +
                              distances[prev2, c1] + lambda_param * penalties[prev2, c1] +
                              distances[c1, next2] + lambda_param * penalties[c1, next2])
                    
                    saving = old_aug - new_aug
                    if saving > best_saving + 1e-6:
                        best_saving = saving
                        best_move = ('exchange', r1, pos1, c1, r2, pos2, c2)
        
        for r_id, route in routes.items():
            if len(route) < 3:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    prev_i = route[i - 1] if i > 0 else 0
                    next_j = route[j + 1] if j < len(route) - 1 else 0
                    
                    old_aug = (distances[prev_i, route[i]] + lambda_param * penalties[prev_i, route[i]] +
                              distances[route[j], next_j] + lambda_param * penalties[route[j], next_j])
                    new_aug = (distances[prev_i, route[j]] + lambda_param * penalties[prev_i, route[j]] +
                              distances[route[i], next_j] + lambda_param * penalties[route[i], next_j])
                    
                    saving = old_aug - new_aug
                    if saving > best_saving + 1e-6:
                        best_saving = saving
                        best_move = ('2opt', r_id, i, j)
        
        if best_move:
            improved = True
            if best_move[0] == 'relocate':
                _, r1, pos1, r2, pos2, customer = best_move
                new_routes = {k: list(v) for k, v in routes.items()}
                new_routes[r1] = routes[r1][:pos1] + routes[r1][pos1+1:]
                new_routes[r2] = routes[r2][:pos2] + [customer] + routes[r2][pos2:]
                routes = new_routes
            elif best_move[0] == 'exchange':
                _, r1, pos1, c1, r2, pos2, c2 = best_move
                new_routes = {k: list(v) for k, v in routes.items()}
                new_routes[r1][pos1] = c2
                new_routes[r2][pos2] = c1
                routes = new_routes
            else:  # 2opt
                _, r_id, i, j = best_move
                new_routes = {k: list(v) for k, v in routes.items()}
                new_routes[r_id] = routes[r_id][:i] + routes[r_id][i:j+1][::-1] + routes[r_id][j+1:]
                routes = new_routes
    
    total_cost = sum(calculate_route_cost(r, distances) for r in routes.values())
    return routes, total_cost


def guided_local_search(routes: Dict[int, List[int]], distances: np.ndarray,
                        demands: np.ndarray, capacity: int,
                        max_iterations: int = 100,
                        time_limit: float = 60.0) -> Tuple[Dict[int, List[int]], int]:
    print("\n=== Guided Local Search ===")
    
    n = distances.shape[0]
    penalties = np.zeros((n, n), dtype=float)
    
    avg_dist = np.mean(distances[distances > 0])
    lambda_param = 0.1 * avg_dist
    
    best_routes, best_cost = inter_route_local_search(
        routes, distances, demands, capacity, verbose=False
    )
    current_routes = best_routes
    print(f"  Initial: {best_cost}")
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            print(f"  Time limit reached at iteration {iteration}")
            break
        
        edges = get_route_edges(current_routes)
        
        max_utility = -1
        edges_to_penalize = []
        
        for i, j in edges:
            utility = distances[i, j] / (1 + penalties[i, j])
            if utility > max_utility + 1e-6:
                max_utility = utility
                edges_to_penalize = [(i, j)]
            elif abs(utility - max_utility) < 1e-6:
                edges_to_penalize.append((i, j))
        
        for i, j in edges_to_penalize:
            penalties[i, j] += 1
            penalties[j, i] += 1  # Symmetric
        
        new_routes, new_cost = gls_local_search(
            current_routes, distances, demands, capacity,
            penalties, lambda_param
        )
        current_routes = new_routes
        
        if new_cost < best_cost:
            print(f"  Iteration {iteration + 1}: {best_cost} -> {new_cost} (IMPROVED -{best_cost - new_cost})")
            best_routes = new_routes
            best_cost = new_cost
    
    print(f"  Final cost: {best_cost}")
    return best_routes, best_cost
