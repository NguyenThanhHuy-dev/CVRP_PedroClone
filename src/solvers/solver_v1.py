import time
import numpy as np
import re
from src.common.instance import Instance
from src.common.utils import calculate_route_cost
from src.algorithms.heuristics import ClarkeWright
from src.algorithms.local_search import comprehensive_local_search, intra_route_2opt
from src.algorithms.metaheuristics import multi_start_ils, guided_local_search
from src.algorithms.exact import SingleRouteOptimizer, PairwiseRouteOptimizer

class SolverV1:
    def __init__(self, config):
        self.config = config

    def solve(self, filepath):
        print(f"\n>>> SOLVER V1: OLD PIPELINE (Classic Heuristics)")
        
        # 1. Load Data
        cvrp = Instance(filepath); cvrp.load()
        n_veh = int(re.search(r'-k(\d+)', filepath).group(1))
        dists, dems, cap = cvrp.distances, np.array([0] + cvrp.demands), cvrp.capacity

        start_total = time.time()

        # 2. Clarke-Wright
        print("Step 1: Clarke-Wright...")
        _, cw_routes = ClarkeWright.run(cvrp, n_veh)
        routes = {i: list(r.value) for i, (_, r) in enumerate(cw_routes.items())}
        
        # 3. Two-Opt
        print("Step 2: Intra-route 2-Opt...")
        routes, _, _ = intra_route_2opt(routes, dists)

        # 4. Multi-Start ILS
        n_restarts = self.config.get('v1_restarts', 5)
        ils_time = self.config.get('v1_ils_time', 60)
        routes, _ = multi_start_ils(routes, dists, dems, cap, 
                                    n_restarts=n_restarts, time_limit=ils_time)

        # 5. GLS (Chạy ngắn để refine)
        routes, _ = guided_local_search(routes, dists, dems, cap, time_limit=20)

        # 6. MaxSAT
        if self.config.get('use_maxsat', True):
            print("Step 6: MaxSAT Optimization...")
            # Single
            opt_routes = {}
            for v, r in routes.items():
                if len(r) > 12: 
                    opt_routes[v] = r; continue
                opt = SingleRouteOptimizer(r, dists)
                new_r, _ = opt.optimize()
                opt_routes[v] = new_r
            routes = opt_routes
            
            # Pairwise (Tái sử dụng logic pairwise từ Exact module)
            improved = True
            st_pair = time.time()
            while improved and (time.time() - st_pair < 30):
                improved = False
                from itertools import combinations
                ids = list(routes.keys())
                for i, j in combinations(ids, 2):
                    if not routes[i] or not routes[j]: continue
                    if len(routes[i]) + len(routes[j]) > 8: continue
                    opt = PairwiseRouteOptimizer(routes[i], routes[j], dists, dems, cap)
                    r1, r2, c = opt.optimize()
                    cur_c = calculate_route_cost(routes[i], dists) + calculate_route_cost(routes[j], dists)
                    if c < cur_c:
                        routes[i], routes[j] = r1, r2
                        improved = True

        final_cost = sum(calculate_route_cost(r, dists) for r in routes.values())
        return routes, final_cost