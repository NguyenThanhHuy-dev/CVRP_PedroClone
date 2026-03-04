import numpy as np
from typing import List, Dict, Tuple

def calculate_route_cost(route: List[int], distances: np.ndarray) -> int:
    """Calculate cost of a route."""
    if not route:
        return 0
    cost = distances[0, route[0]]
    for i in range(len(route) - 1):
        cost += distances[route[i], route[i + 1]]
    cost += distances[route[-1], 0]
    return int(cost)

def calculate_augmented_cost(route: List[int], distances: np.ndarray, 
                             penalties: np.ndarray, lambda_param: float) -> float:
    """Calculate augmented cost = actual cost + lambda * penalties."""
    if not route:
        return 0
    
    cost = distances[0, route[0]] + lambda_param * penalties[0, route[0]]
    for i in range(len(route) - 1):
        cost += distances[route[i], route[i + 1]] + lambda_param * penalties[route[i], route[i + 1]]
    cost += distances[route[-1], 0] + lambda_param * penalties[route[-1], 0]
    return cost

def get_route_edges(routes: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    """Get all edges in the current solution."""
    edges = []
    for route in routes.values():
        if not route:
            continue
        edges.append((0, route[0]))
        for i in range(len(route) - 1):
            edges.append((route[i], route[i + 1]))
        edges.append((route[-1], 0))
    return edges
