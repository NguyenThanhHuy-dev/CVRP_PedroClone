import time
import numpy as np
import re
from src.common.instance import Instance
from src.common.utils import calculate_route_cost
from src.algorithms.heuristics import ClarkeWright
from src.algorithms.local_search import simple_local_search
from src.algorithms.metaheuristics import guided_local_search, lns_perturbation
from src.algorithms.exact import SingleRouteOptimizer, PairwiseRouteOptimizer

class SolverV2:
    def __init__(self, config):
        self.config = config

    def solve(self, filepath):
        print(f"\n>>> SOLVER V2: NEW PIPELINE (Hybrid LNS + MaxSAT)")
        
        cvrp = Instance(filepath); cvrp.load()
        n_veh = int(re.search(r'-k(\d+)', filepath).group(1))
        dists, dems, cap = cvrp.distances, np.array([0] + cvrp.demands), cvrp.capacity

        # 1. Clarke-Wright
        print("Step 1: Clarke-Wright...")
        _, cw = ClarkeWright.run(cvrp, n_veh)
        routes = {i: list(r.value) for i, (_, r) in enumerate(cw.items())}

        # 2. Hybrid ILS/LNS Loop
        print("Step 2: Hybrid ILS/LNS...")
        ils_iter = self.config.get('ils_iterations', 10)
        best_cost = float('inf')
        best_routes = None
        
        start_ils = time.time()
        curr_routes = routes
        for i in range(ils_iter):
            if time.time() - start_ils > 120: break
            
            # Nếu không phải vòng đầu, dùng LNS để phá cấu trúc
            if i > 0 and best_routes:
                curr_routes = lns_perturbation(best_routes, dists, dems, cap, destroy_pct=0.4)
            
            # Local Search (Simple)
            curr_routes, cost = simple_local_search(curr_routes, dists, dems, cap)
            
            if cost < best_cost:
                best_cost = cost
                best_routes = {k: list(v) for k, v in curr_routes.items()}
                print(f"  > ILS Iter {i+1}: New Best {best_cost}")

        # 3. GLS
        gls_time = self.config.get('gls_time_limit', 20)
        best_routes, _ = guided_local_search(best_routes, dists, dems, cap, time_limit=gls_time)

        # 4. MaxSAT
        if self.config.get('use_maxsat', True):
            print("Step 4: MaxSAT Optimization...")
            # Single
            for v, r in best_routes.items():
                if len(r) > 11: continue
                opt = SingleRouteOptimizer(r, dists)
                best_routes[v], _ = opt.optimize()
            
            # Pairwise
            improved = True
            st_pair = time.time()
            while improved and (time.time() - st_pair < 60):
                improved = False
                ids = list(best_routes.keys())
                from itertools import combinations
                for i, j in combinations(ids, 2):
                    if not best_routes[i] or not best_routes[j]: continue
                    if len(best_routes[i]) + len(best_routes[j]) > 9: continue
                    
                    cur_c = calculate_route_cost(best_routes[i], dists) + calculate_route_cost(best_routes[j], dists)
                    opt = PairwiseRouteOptimizer(best_routes[i], best_routes[j], dists, dems, cap)
                    r1, r2, new_c = opt.optimize()
                    
                    if new_c < cur_c:
                        best_routes[i], best_routes[j] = r1, r2
                        improved = True

        final_cost = sum(calculate_route_cost(r, dists) for r in best_routes.values())
        return best_routes, final_cost