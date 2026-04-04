# classes/cross_exchange.py
from typing import Dict, List, Tuple
from itertools import combinations
import numpy as np

class CrossExchange:
    ''' Class for Cross-Exchange heuristic (List-based version) '''

    @staticmethod
    def _calc_cost(route: List[int], distances: np.ndarray) -> int:
        if not route: return 0
        cost = distances[0, route[0]]
        for i in range(len(route) - 1):
            cost += distances[route[i], route[i+1]]
        cost += distances[route[-1], 0]
        return int(cost)

    @staticmethod
    def run(routes: Dict[int, List[int]], distances: np.ndarray, demands: np.ndarray, capacity: int) -> Tuple[Dict[int, List[int]], float, bool]:
        best_routes = {k: list(v) for k, v in routes.items()}
        improved = False
        
        for id1, id2 in combinations(list(best_routes.keys()), 2):
            r1, r2 = best_routes[id1], best_routes[id2]
            pair_improved = True
            while pair_improved:
                pair_improved = False
                for k1 in [1, 2, 3]:
                    if len(r1) < k1: continue
                    for k2 in [1, 2, 3]:
                        if len(r2) < k2: continue
                        for i in range(len(r1) - k1 + 1):
                            for j in range(len(r2) - k2 + 1):
                                block1, block2 = r1[i:i+k1], r2[j:j+k2]
                                
                                # Tính toán tải trọng mới
                                dem1 = sum(demands[c] for c in r1) - sum(demands[c] for c in block1) + sum(demands[c] for c in block2)
                                dem2 = sum(demands[c] for c in r2) - sum(demands[c] for c in block2) + sum(demands[c] for c in block1)
                                
                                if dem1 > capacity or dem2 > capacity: continue
                                    
                                new_r1 = r1[:i] + block2 + r1[i+k1:]
                                new_r2 = r2[:j] + block1 + r2[j+k2:]
                                
                                old_c = CrossExchange._calc_cost(r1, distances) + CrossExchange._calc_cost(r2, distances)
                                new_c = CrossExchange._calc_cost(new_r1, distances) + CrossExchange._calc_cost(new_r2, distances)
                                
                                if new_c < old_c:
                                    r1, r2 = new_r1, new_r2
                                    best_routes[id1], best_routes[id2] = r1, r2
                                    pair_improved = True
                                    improved = True
                                    break
                            if pair_improved: break
                        if pair_improved: break
                    if pair_improved: break
                    
        total_cost = sum(CrossExchange._calc_cost(r, distances) for r in best_routes.values())
        return best_routes, float(total_cost), improved