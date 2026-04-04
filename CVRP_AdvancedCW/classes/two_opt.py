# classes/two_opt.py
from typing import Dict, List, Tuple
import numpy as np

class TwoOpt:
    ''' Class for the 2-opt heuristic (List-based version) '''

    @staticmethod
    def _calc_cost(route: List[int], distances: np.ndarray) -> int:
        if not route: return 0
        cost = distances[0, route[0]]
        for i in range(len(route) - 1):
            cost += distances[route[i], route[i+1]]
        cost += distances[route[-1], 0]
        return int(cost)

    @staticmethod
    def run(routes: Dict[int, List[int]], distances: np.ndarray) -> Tuple[Dict[int, List[int]], float, bool]:
        best_routes = {k: list(v) for k, v in routes.items()}
        improved = False
        
        for r_id, route in best_routes.items():
            n = len(route)
            if n < 2: continue
            route_improved = True
            while route_improved:
                route_improved = False
                for i in range(n - 1):
                    for j in range(i + 2, n + 1):
                        new_r = route[:i] + route[i:j][::-1] + route[j:]
                        if TwoOpt._calc_cost(new_r, distances) < TwoOpt._calc_cost(route, distances):
                            route = new_r
                            route_improved = True
                            improved = True
                            break
                    if route_improved: break
            best_routes[r_id] = route
            
        total_cost = sum(TwoOpt._calc_cost(r, distances) for r in best_routes.values())
        return best_routes, float(total_cost), improved