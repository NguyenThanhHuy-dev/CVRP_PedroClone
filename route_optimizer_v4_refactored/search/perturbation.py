import random
import numpy as np
from typing import List, Dict

def perturbation(routes: Dict[int, List[int]], distances: np.ndarray,
                 demands: np.ndarray, capacity: int, 
                 strength: int = 5) -> Dict[int, List[int]]:
    """
    Perturb solution by performing random moves (accept even if worse).
    """
    routes = {k: list(v) for k, v in routes.items()}
    route_ids = [k for k in routes if routes[k]]
    
    moves_done = 0
    attempts = 0
    max_attempts = strength * 10
    
    while moves_done < strength and attempts < max_attempts:
        attempts += 1
        
        if len(route_ids) < 2:
            break
            
        # Pick random customer from random route
        r1 = random.choice(route_ids)
        if not routes[r1]:
            continue
        
        pos1 = random.randint(0, len(routes[r1]) - 1)
        customer = routes[r1][pos1]
        
        # Try to move to random position in another route
        other_routes = [r for r in route_ids if r != r1 and routes[r]]
        if not other_routes:
            continue
        r2 = random.choice(other_routes)
            
        demand_r2 = sum(demands[c] for c in routes[r2])
        if demand_r2 + demands[customer] <= capacity:
            routes[r1] = routes[r1][:pos1] + routes[r1][pos1+1:]
            pos2 = random.randint(0, len(routes[r2]))
            routes[r2] = routes[r2][:pos2] + [customer] + routes[r2][pos2:]
            moves_done += 1
            
            # Update route_ids
            route_ids = [k for k in routes if routes[k]]
    
    return routes


def double_bridge_perturbation(routes: Dict[int, List[int]], distances: np.ndarray,
                               demands: np.ndarray, capacity: int) -> Dict[int, List[int]]:
    """
    Double Bridge: Exchange segments between two routes.
    This is a stronger perturbation that can escape deep local optima.
    """
    routes = {k: list(v) for k, v in routes.items()}
    route_ids = [k for k in routes if len(routes[k]) >= 2]
    
    if len(route_ids) < 2:
        return routes
    
    # Pick two random routes
    r1, r2 = random.sample(route_ids, 2)
    route1, route2 = routes[r1], routes[r2]
    
    if len(route1) < 2 or len(route2) < 2:
        return routes
    
    # Pick random cut points
    cut1 = random.randint(1, len(route1) - 1)
    cut2 = random.randint(1, len(route2) - 1)
    
    # Exchange segments
    new_route1 = route1[:cut1] + route2[cut2:]
    new_route2 = route2[:cut2] + route1[cut1:]
    
    # Check capacity
    demand1 = sum(demands[c] for c in new_route1)
    demand2 = sum(demands[c] for c in new_route2)
    
    if demand1 <= capacity and demand2 <= capacity:
        routes[r1] = new_route1
        routes[r2] = new_route2
    
    return routes


def lns_perturbation(routes: Dict[int, List[int]], distances: np.ndarray,
                     demands: np.ndarray, capacity: int,
                     destroy_pct: float = 0.3,
                     destroy_method: str = 'random') -> Dict[int, List[int]]:
    """
    LNS-style perturbation with multiple destroy methods:
    - 'random': Random removal
    - 'worst': Remove customers with highest insertion cost
    - 'shaw': Remove similar customers (close to each other)
    
    Repair using Regret-2 heuristic.
    """
    routes = {k: list(v) for k, v in routes.items()}
    
    # Collect all customers with their positions
    all_customers = []
    customer_info = {}  # customer -> (route_id, position)
    for r_id, route in routes.items():
        for pos, c in enumerate(route):
            all_customers.append(c)
            customer_info[c] = (r_id, pos)
    
    if len(all_customers) < 3:
        return routes
    
    n_remove = max(2, int(len(all_customers) * destroy_pct))
    
    # Choose destroy method
    if destroy_method == 'worst':
        # Calculate removal cost for each customer
        removal_costs = []
        for c in all_customers:
            r_id, pos = customer_info[c]
            route = routes[r_id]
            prev_node = route[pos - 1] if pos > 0 else 0
            next_node = route[pos + 1] if pos < len(route) - 1 else 0
            
            # Cost saved by removing this customer
            cost_saved = (distances[prev_node, c] + distances[c, next_node] - 
                         distances[prev_node, next_node])
            removal_costs.append((cost_saved, c))
        
        # Sort by cost (descending) - remove worst customers first
        removal_costs.sort(reverse=True)
        removed = [c for _, c in removal_costs[:n_remove]]
        
    elif destroy_method == 'shaw':
        # Shaw removal: remove customers similar to a seed customer
        seed = random.choice(all_customers)
        
        # Calculate "relatedness" to seed (based on distance)
        relatedness = []
        for c in all_customers:
            if c != seed:
                rel = distances[seed, c]
                relatedness.append((rel, c))
        
        # Sort by relatedness (ascending) - most similar first
        relatedness.sort()
        removed = [seed] + [c for _, c in relatedness[:n_remove - 1]]
        
    else:  # random
        removed = random.sample(all_customers, n_remove)
    
    # Remove customers from routes
    for r_id in routes:
        routes[r_id] = [c for c in routes[r_id] if c not in removed]
    
    # Reinsert using Regret-2 heuristic
    while removed:
        best_regret = -float('inf')
        best_customer = None
        best_route = None
        best_pos = None
        
        for customer in removed:
            # Find best and second-best insertion for this customer
            insertions = []
            
            for r_id, route in routes.items():
                route_demand = sum(demands[c] for c in route)
                if route_demand + demands[customer] > capacity:
                    continue
                
                for pos in range(len(route) + 1):
                    prev_node = route[pos - 1] if pos > 0 else 0
                    next_node = route[pos] if pos < len(route) else 0
                    
                    cost_increase = (distances[prev_node, customer] + 
                                    distances[customer, next_node] -
                                    distances[prev_node, next_node])
                    
                    insertions.append((cost_increase, r_id, pos))
            
            if not insertions:
                continue
            
            insertions.sort(key=lambda x: x[0])
            best_cost = insertions[0][0]
            second_cost = insertions[1][0] if len(insertions) > 1 else best_cost + 100
            
            regret = second_cost - best_cost
            
            if regret > best_regret:
                best_regret = regret
                best_customer = customer
                best_route = insertions[0][1]
                best_pos = insertions[0][2]
        
        if best_customer is not None:
            routes[best_route].insert(best_pos, best_customer)
            removed.remove(best_customer)
        else:
            # No feasible insertion, create new route
            customer = removed.pop(0)
            new_id = max(routes.keys()) + 1
            routes[new_id] = [customer]
    
    return routes
