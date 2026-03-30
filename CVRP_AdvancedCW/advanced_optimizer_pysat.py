#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Advanced CVRP Optimizer using MaxSAT + Multiprocessing Timeout
==============================================================
"""

import sys
import os
import time
import logging
import multiprocessing
import numpy as np
from typing import List, Tuple, Dict, Any
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- CẤU HÌNH LOGGING ---
log_filename = f"solver_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# =====================================================================
# MAX-SAT OPTIMIZERS (SINGLE & PAIRWISE)
# =====================================================================

class SingleRouteOptimizer:
    """Tối ưu 1 route bằng MaxSAT."""
    def __init__(self, customers: List[int], distances: np.ndarray):
        self.customers = customers
        self.n = len(customers) + 1
        self.distances = distances
        self.local_to_global = {0: 0}
        for i, c in enumerate(customers):
            self.local_to_global[i + 1] = c

        self.local_dist = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                gi, gj = self.local_to_global[i], self.local_to_global[j]
                self.local_dist[i, j] = distances[gi, gj]

        self.var_id = 0
        self.conNet = [[0] * self.n for _ in range(self.n)]
        self.rchNet = [[0] * self.n for _ in range(self.n)]
        self.wcnf = WCNF()

    def new_var_id(self) -> int:
        self.var_id += 1
        return self.var_id

    def optimize(self) -> Tuple[List[int], int]:
        for i in range(self.n):
            for j in range(self.n):
                if i != j: self.conNet[i][j] = self.new_var_id()
        for i in range(1, self.n):
            for j in range(i + 1, self.n):
                var = self.new_var_id()
                self.rchNet[i][j] = var
                self.rchNet[j][i] = -var

        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.local_dist[i, j] > 0:
                    self.wcnf.append([-self.conNet[i][j]], weight=self.local_dist[i, j])

        for i in range(1, self.n):
            for j in range(1, self.n):
                if i != j and self.rchNet[i][j] != 0:
                    self.wcnf.append([-self.conNet[i][j], self.rchNet[i][j]])

        for a in range(1, self.n):
            for b in range(1, self.n):
                if a == b: continue
                for c in range(1, self.n):
                    if c == a or c == b: continue
                    r_ab = self.rchNet[a][b]
                    r_bc = self.rchNet[b][c]
                    r_ac = self.rchNet[a][c]
                    if r_ab != 0 and r_bc != 0 and r_ac != 0:
                        self.wcnf.append([-r_ab, -r_bc, r_ac])

        for a in range(1, self.n):
            for b in range(1, self.n):
                if a == b: continue
                for c in range(1, self.n):
                    if c == a or c == b: continue
                    r_ab = self.rchNet[a][b]
                    r_bc = self.rchNet[b][c]
                    l_ac = self.conNet[a][c]
                    if r_ab != 0 and r_bc != 0 and l_ac != 0:
                        self.wcnf.append([-r_ab, -r_bc, -l_ac])

        out_vars = [self.conNet[0][j] for j in range(1, self.n)]
        self.wcnf.append(out_vars)
        for i in range(len(out_vars)):
            for j in range(i + 1, len(out_vars)):
                self.wcnf.append([-out_vars[i], -out_vars[j]])

        in_vars = [self.conNet[i][0] for i in range(1, self.n)]
        self.wcnf.append(in_vars)
        for i in range(len(in_vars)):
            for j in range(i + 1, len(in_vars)):
                self.wcnf.append([-in_vars[i], -in_vars[j]])

        for c in range(1, self.n):
            in_v = [self.conNet[i][c] for i in range(self.n) if i != c]
            self.wcnf.append(in_v)
            for i in range(len(in_v)):
                for j in range(i + 1, len(in_v)):
                    self.wcnf.append([-in_v[i], -in_v[j]])

            out_v = [self.conNet[c][j] for j in range(self.n) if j != c]
            self.wcnf.append(out_v)
            for i in range(len(out_v)):
                for j in range(i + 1, len(out_v)):
                    self.wcnf.append([-out_v[i], -out_v[j]])

        with RC2(self.wcnf, verbose=0) as solver:
            model = solver.compute()
            if model:
                positive = set(v for v in model if v > 0)
                route = []
                current = 0
                visited = {0}
                for _ in range(self.n - 1):
                    found = False
                    for j in range(self.n):
                        if j not in visited and self.conNet[current][j] in positive:
                            gid = self.local_to_global[j]
                            if gid != 0: route.append(gid)
                            visited.add(j)
                            current = j
                            found = True
                            break
                    if not found: break
                return route, solver.cost
            else:
                orig_cost = self.local_dist[0, 1]
                for i in range(1, self.n - 1): orig_cost += self.local_dist[i, i + 1]
                orig_cost += self.local_dist[self.n - 1, 0]
                return self.customers, orig_cost


class PairwiseRouteOptimizer:
    """Tối ưu 2 routes bằng MaxSAT."""
    def __init__(self, c1: List[int], c2: List[int], dist: np.ndarray, dem: np.ndarray, cap: int):
        self.c1, self.c2 = c1, c2
        self.all_customers = c1 + c2
        self.n = len(self.all_customers) + 1
        self.dist, self.dem, self.cap = dist, dem, cap
        self.l2g = {0: 0}
        for i, c in enumerate(self.all_customers): self.l2g[i + 1] = c
        self.ldist = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                self.ldist[i, j] = dist[self.l2g[i], self.l2g[j]]
        self.var_id = 0
        self.con = [[[0]*self.n for _ in range(self.n)] for _ in range(2)]
        self.rch = [[[0]*self.n for _ in range(self.n)] for _ in range(2)]
        self.asn = [0]*self.n
        self.wcnf = WCNF()

    def nid(self): self.var_id += 1; return self.var_id

    def optimize(self) -> Tuple[List[int], List[int], int, bool]:
        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i != j: self.con[v][i][j] = self.nid()
        for v in range(2):
            for i in range(1, self.n):
                for j in range(i+1, self.n):
                    v_r = self.nid()
                    self.rch[v][i][j] = v_r
                    self.rch[v][j][i] = -v_r
        for i in range(1, self.n): self.asn[i] = self.nid()

        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i != j and self.ldist[i,j] > 0:
                        self.wcnf.append([-self.con[v][i][j]], weight=self.ldist[i,j])

        for i in range(1, self.n):
            for v in range(2):
                sign = 1 if v==1 else -1
                in_e = [self.con[v][j][i] for j in range(self.n) if j!=i]
                self.wcnf.append([-sign*self.asn[i]] + in_e)
                for j in range(self.n):
                    if j!=i:
                        self.wcnf.append([sign*self.asn[i], -self.con[v][j][i]])
                        self.wcnf.append([sign*self.asn[i], -self.con[v][i][j]])

        for v in range(2):
            for i in range(1, self.n):
                for j in range(1, self.n):
                    if i!=j and self.rch[v][i][j]!=0:
                        self.wcnf.append([-self.con[v][i][j], self.rch[v][i][j]])
            for a in range(1, self.n):
                for b in range(1, self.n):
                    if a==b: continue
                    for c in range(1, self.n):
                        if c in (a,b): continue
                        rab, rbc, rac = self.rch[v][a][b], self.rch[v][b][c], self.rch[v][a][c]
                        if rab!=0 and rbc!=0 and rac!=0:
                            self.wcnf.append([-rab, -rbc, rac])
            for a in range(1, self.n):
                for b in range(1, self.n):
                    if a==b: continue
                    for c in range(1, self.n):
                        if c in (a,b): continue
                        rab, rbc, lac = self.rch[v][a][b], self.rch[v][b][c], self.con[v][a][c]
                        if rab!=0 and rbc!=0 and lac!=0:
                            self.wcnf.append([-rab, -rbc, -lac])

            out_v = [self.con[v][0][j] for j in range(1, self.n)]
            for i in range(len(out_v)):
                for j in range(i+1, len(out_v)): self.wcnf.append([-out_v[i], -out_v[j]])
            in_v = [self.con[v][i][0] for i in range(1, self.n)]
            for i in range(len(in_v)):
                for j in range(i+1, len(in_v)): self.wcnf.append([-in_v[i], -in_v[j]])

        for i in range(1, self.n):
            for v in range(2):
                in_e = [self.con[v][j][i] for j in range(self.n) if j!=i]
                out_e = [self.con[v][i][j] for j in range(self.n) if j!=i]
                for x in range(len(in_e)):
                    for y in range(x+1, len(in_e)): self.wcnf.append([-in_e[x], -in_e[y]])
                for x in range(len(out_e)):
                    for y in range(x+1, len(out_e)): self.wcnf.append([-out_e[x], -out_e[y]])
                for ie in in_e: self.wcnf.append([-ie] + out_e)
            alle = []
            for v in range(2): alle.extend([self.con[v][j][i] for j in range(self.n) if j!=i])
            self.wcnf.append(alle)

        with RC2(self.wcnf, verbose=0) as solver:
            model = solver.compute()
            if model:
                pos = set(v for v in model if v>0)
                rts = [[], []]
                for v in range(2):
                    cur = 0; vis = {0}
                    for _ in range(self.n-1):
                        f = False
                        for j in range(self.n):
                            if j not in vis and self.con[v][cur][j] in pos:
                                gid = self.l2g[j]
                                if gid!=0: rts[v].append(gid)
                                vis.add(j); cur = j; f = True; break
                        if not f: break
                if sum(self.dem[c] for c in rts[0]) > self.cap or sum(self.dem[c] for c in rts[1]) > self.cap:
                    return [], [], float('inf'), False
                return rts[0], rts[1], solver.cost, True
            return [], [], float('inf'), False

# =====================================================================
# MULTIPROCESSING WRAPPERS
# =====================================================================
def _solve_single_worker(customers, distances):
    opt = SingleRouteOptimizer(customers, distances)
    return opt.optimize()

def _solve_pairwise_worker(c1, c2, distances, demands, capacity):
    opt = PairwiseRouteOptimizer(c1, c2, distances, demands, capacity)
    return opt.optimize()

# =====================================================================
# ADVANCED CVRP OPTIMIZER (Parameterized & Tracked)
# =====================================================================
class AdvancedCVRPOptimizer:
    def __init__(self, distances: np.ndarray, demands: np.ndarray, capacity: int, n_vehicles: int, config: Dict[str, Any] = None):
        self.distances = distances
        self.demands = demands
        self.capacity = capacity
        self.n_vehicles = n_vehicles
        # Mặc định (Default Hyperparameters)
        if config is None:
            config = {}
        self.max_single_size = config.get("max_single_size", 11)
        self.single_timeout = config.get("single_timeout", 5.0)
        self.max_pairwise_size = config.get("max_pairwise_size", 10)
        self.pairwise_timeout = config.get("pairwise_timeout", 8.0)
        self.n_closest_pairs = config.get("n_closest_pairs", 3)
        self.patience = config.get("patience", 5)
        # BỘ ĐẾM THỐNG KÊ (Statistics Counters)
        self.stat_single_improvements = 0
        self.stat_pairwise_improvements = 0

    def compute_route_cost(self, route: List[int]) -> int:
        if not route: return 0
        cost = self.distances[0, route[0]]
        for i in range(len(route) - 1):
            cost += self.distances[route[i], route[i + 1]]
        cost += self.distances[route[-1], 0]
        return cost

    def compute_total_cost(self, routes: Dict[int, List[int]]) -> float:
        return sum(self.compute_route_cost(r) for r in routes.values())

    def find_closest_route_pairs(self, routes: Dict[int, List[int]]) -> List[Tuple[int, int]]:
        route_ids = list(routes.keys())
        if len(route_ids) < 2: return []
        pair_scores = []
        for i, j in combinations(route_ids, 2):
            route_i, route_j = routes[i], routes[j]
            if not route_i or not route_j: continue
            min_dist = float('inf')
            for c1 in route_i:
                for c2 in route_j:
                    min_dist = min(min_dist, self.distances[c1, c2])
            pair_scores.append((min_dist, i, j))
        pair_scores.sort()
        return [(i, j) for _, i, j in pair_scores[:self.n_closest_pairs]]

    def optimize_single_route_safe(self, route: List[int]) -> Tuple[List[int], int]:
        if len(route) <= 1: return route, self.compute_route_cost(route)
        if len(route) > self.max_single_size: return route, self.compute_route_cost(route)

        pool = multiprocessing.Pool(processes=1)
        res = pool.apply_async(_solve_single_worker, (route, self.distances))
        try:
            opt_r, opt_c = res.get(timeout=self.single_timeout)
            return opt_r, opt_c
        except multiprocessing.TimeoutError:
            logging.warning(" [Timeout] Single MaxSAT bị kẹt. Đã skip.")
            return route, self.compute_route_cost(route)
        except Exception as e:
            return route, self.compute_route_cost(route)
        finally:
            pool.terminate()
            pool.join()

    def optimize_route_pair_safe(self, route1: List[int], route2: List[int]) -> Tuple[List[int], List[int], int, bool]:
        if not route1 and not route2: return [], [], 0, True
        if len(route1) + len(route2) > self.max_pairwise_size: return [], [], float('inf'), False

        pool = multiprocessing.Pool(processes=1)
        res = pool.apply_async(_solve_pairwise_worker, (route1, route2, self.distances, self.demands, self.capacity))
        try:
            r1, r2, cost, success = res.get(timeout=self.pairwise_timeout)
            return r1, r2, cost, success
        except multiprocessing.TimeoutError:
            logging.warning(" [Timeout] Pairwise MaxSAT bị kẹt. Đã skip.")
            return [], [], float('inf'), False
        except Exception as e:
            return [], [], float('inf'), False
        finally:
            pool.terminate()
            pool.join()

    def try_relocate(self, routes: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], float, bool]:
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost = self.compute_total_cost(routes)
        improved = False
        route_ids = list(routes.keys())

        for src_id in route_ids:
            for dst_id in route_ids:
                if src_id == dst_id: continue
                src_route = routes[src_id]
                dst_route = routes[dst_id]
                for i, customer in enumerate(src_route):
                    dst_demand = sum(self.demands[c] for c in dst_route)
                    if dst_demand + self.demands[customer] > self.capacity:
                        continue
                    for j in range(len(dst_route) + 1):
                        new_src = src_route[:i] + src_route[i+1:]
                        new_dst = dst_route[:j] + [customer] + dst_route[j:]
                        new_cost = self.compute_route_cost(new_src) + self.compute_route_cost(new_dst)
                        old_cost = self.compute_route_cost(src_route) + self.compute_route_cost(dst_route)
                        if new_cost < old_cost:
                            test_routes = {k: list(v) for k, v in routes.items()}
                            test_routes[src_id] = new_src
                            test_routes[dst_id] = new_dst
                            total_cost = self.compute_total_cost(test_routes)
                            if total_cost < best_cost:
                                best_cost = total_cost
                                best_routes = test_routes
                                improved = True
        return best_routes, best_cost, improved

    def try_exchange(self, routes: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], float, bool]:
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost = self.compute_total_cost(routes)
        improved = False
        route_ids = list(routes.keys())

        for id1, id2 in combinations(route_ids, 2):
            route1, route2 = routes[id1], routes[id2]
            for i, c1 in enumerate(route1):
                for j, c2 in enumerate(route2):
                    new_demand1 = sum(self.demands[c] for c in route1) - self.demands[c1] + self.demands[c2]
                    new_demand2 = sum(self.demands[c] for c in route2) - self.demands[c2] + self.demands[c1]
                    if new_demand1 > self.capacity or new_demand2 > self.capacity:
                        continue
                    new_route1 = route1[:i] + [c2] + route1[i+1:]
                    new_route2 = route2[:j] + [c1] + route2[j+1:]
                    new_cost = self.compute_route_cost(new_route1) + self.compute_route_cost(new_route2)
                    old_cost = self.compute_route_cost(route1) + self.compute_route_cost(route2)
                    if new_cost < old_cost:
                        test_routes = {k: list(v) for k, v in routes.items()}
                        test_routes[id1] = new_route1
                        test_routes[id2] = new_route2
                        total_cost = self.compute_total_cost(test_routes)
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_routes = test_routes
                            improved = True
        return best_routes, best_cost, improved

    def optimize(self, initial_routes: Dict[int, List[int]], max_iterations: int = 100) -> Tuple[Dict[int, List[int]], float]:
        routes = {k: list(v) for k, v in initial_routes.items()}
        best_cost = self.compute_total_cost(routes)
        logging.info("=========================================")
        logging.info("BẮT ĐẦU VÒNG LẶP OPTIMIZE (TUNING MODE)")
        logging.info(f" -> Max Single Size : {self.max_single_size}")
        logging.info(f" -> Single Timeout : {self.single_timeout}s")
        logging.info(f" -> Max Pairwise Size: {self.max_pairwise_size}")
        logging.info(f" -> Pairwise Timeout : {self.pairwise_timeout}s")
        logging.info(f" -> Closest Pairs : {self.n_closest_pairs}")
        logging.info(f" -> Patience (No Imp): {self.patience}")
        logging.info(f" -> Max Iterations : {max_iterations}")
        logging.info(f"Chi phí khởi điểm: {best_cost:.2f}")
        logging.info("=========================================")
        iteration = 0
        no_improvement_count = 0
        while iteration < max_iterations and no_improvement_count < self.patience:
            iteration += 1
            improved = False
            logging.info(f"--- Đang chạy Iteration {iteration}/{max_iterations} ---")
            # Phase 1: Relocate
            routes, new_r_cost, reloc_imp = self.try_relocate(routes)
            if reloc_imp:
                improved = True
                logging.info(f" [P1] Relocate giảm chi phí xuống còn: {new_r_cost:.2f}")

            # Phase 2: Exchange
            routes, new_e_cost, exch_imp = self.try_exchange(routes)
            if exch_imp:
                improved = True
                logging.info(f" [P2] Exchange giảm chi phí xuống còn: {new_e_cost:.2f}")

            # Phase 3: Single MaxSAT
            single_imp = 0
            for v, route in routes.items():
                old_c = self.compute_route_cost(route)
                opt_r, opt_c = self.optimize_single_route_safe(route)
                if opt_c < old_c:
                    routes[v] = opt_r
                    improved = True
                    single_imp += (old_c - opt_c)
                    self.stat_single_improvements += 1 # Tăng bộ đếm
            if single_imp > 0:
                logging.info(f" [P3] MaxSAT Single Route cải thiện được: {single_imp}")

            # Phase 4: Pairwise MaxSAT
            pairs = self.find_closest_route_pairs(routes)
            for i, j in pairs:
                r1, r2 = routes[i], routes[j]
                old_c = self.compute_route_cost(r1) + self.compute_route_cost(r2)
                opt1, opt2, new_c, success = self.optimize_route_pair_safe(r1, r2)
                if success and new_c < old_c:
                    routes[i], routes[j] = opt1, opt2
                    improved = True
                    self.stat_pairwise_improvements += 1 # Tăng bộ đếm
                    logging.info(f" [P4] MaxSAT Pairwise ({i},{j}) giảm: {old_c - new_c}")

            # End Iteration Evaluation
            current_cost = self.compute_total_cost(routes)
            if current_cost < best_cost - 0.001:
                best_cost = current_cost
                no_improvement_count = 0
                logging.info(f"=> ITER {iteration}: BEST COST MỚI = {best_cost:.2f}")
            else:
                no_improvement_count += 1
                logging.info(f"=> ITER {iteration}: Không cải thiện (no_improve={no_improvement_count}/{self.patience})")

            if not improved: no_improvement_count += 1
            
        logging.info(f"Hoàn thành tối ưu. Final cost: {best_cost:.2f}")
        return routes, best_cost

# =====================================================================
# MAIN SOLVER ROUTINE
# =====================================================================
def solve_advanced(filepath: str, config: Dict[str, Any] = None, max_iterations: int = 50):
    import re
    from classes.instance import Instance
    from classes.clarke_wright import ClarkeWright
    from classes.two_opt import TwoOpt
    logging.info("=" * 70)
    logging.info(f"KHỞI CHẠY ADVANCED CVRP SOLVER CHO FILE: {os.path.basename(filepath)}")
    logging.info("=" * 70)
    cvrp = Instance(filepath)
    cvrp.load()
    cvrp.distances = np.floor(cvrp.distances + 0.5).astype(int)
    n_vehicles = int(re.search(r'-k(\d+)', filepath).group(1))
    logging.info(f"Thông tin dữ liệu: Dimension={cvrp.dimension}, K={n_vehicles}, Capacity={cvrp.capacity}")
    
    # 1. Clarke-Wright
    logging.info("--- Step 1: Clarke-Wright Heuristic ---")
    try:
        cw_time, cw_routes = ClarkeWright.run(cvrp, n_vehicles)
    except Exception:
        logging.warning("⚠️ Clarke-Wright kẹt với K giới hạn. Chạy fallback K=999...")
        cw_time, cw_routes = ClarkeWright.run(cvrp, 999)
    cw_cost = sum(route.cost for route in cw_routes.values())
    logging.info(f" Kết quả CW: Cost = {cw_cost}, Time = {cw_time:.3f}s")
    
    # 2. Two-Opt
    logging.info("--- Step 2: Two-Opt Local Search ---")
    two_opt_time, two_opt_routes = TwoOpt.run(cw_routes)
    two_opt_cost = sum(route.cost for route in two_opt_routes.values())
    logging.info(f" Kết quả Two-Opt: Cost = {two_opt_cost}, Time = {two_opt_time:.3f}s")
    
    routes = {i: list(route.value) for i, (_, route) in enumerate(two_opt_routes.items())}
    routes = {i: r for i, r in routes.items() if len(r) > 0}
    
    # 3. Advanced Optimizer
    logging.info("--- Step 3: Advanced MaxSAT Optimization ---")
    start_time = time.time()
    optimizer = AdvancedCVRPOptimizer(
        distances=cvrp.distances,
        demands=np.array(cvrp.demands),
        capacity=cvrp.capacity,
        n_vehicles=n_vehicles,
        config=config
    )
    opt_routes, opt_cost = optimizer.optimize(routes, max_iterations=max_iterations)
    maxsat_time = time.time() - start_time
    
    # Lấy thống kê
    stats = {
        "single_imp_count": optimizer.stat_single_improvements,
        "pairwise_imp_count": optimizer.stat_pairwise_improvements
    }
    
    # Báo cáo
    logging.info("=" * 70)
    logging.info("BÁO CÁO TỔNG KẾT (SUMMARY)")
    logging.info("=" * 70)
    logging.info(f" 1. Chi phí Clarke-Wright: {cw_cost}")
    logging.info(f" 2. Chi phí sau Two-Opt: {two_opt_cost} (giảm được {cw_cost - two_opt_cost})")
    final_int_cost = int(opt_cost)
    logging.info(f" 3. Chi phí sau MaxSAT: {final_int_cost}")
    logging.info(f" => TỔNG CẢI THIỆN: {cw_cost - final_int_cost}")
    logging.info(f" => Số lần MaxSAT Single giúp giảm cost : {stats['single_imp_count']} lần")
    logging.info(f" => Số lần MaxSAT Pairwise giúp giảm cost: {stats['pairwise_imp_count']} lần")
    logging.info(f" => TỔNG THỜI GIAN CHẠY: {cw_time + two_opt_time + maxsat_time:.3f}s")
    
    logging.info("\nChi tiết các tuyến cuối cùng (Final Routes):")
    total_verify = 0
    for v, route in sorted(opt_routes.items()):
        if route:
            demand = sum(cvrp.demands[c] for c in route)
            cost = optimizer.compute_route_cost(route)
            total_verify += cost
            logging.info(f" Tuyến xe {v}: {route} (Tải={demand}/{cvrp.capacity}, Chi phí={cost})")
    
    logging.info(f"Chi phí xác minh lại (Verified distance): {total_verify}")
    # Trả về thêm tham số stats
    return opt_routes, total_verify, stats

if __name__ == "__main__":
    # Test thử truyền config
    test_config = {
        "max_single_size": 12,
        "single_timeout": 5.0,
        "max_pairwise_size": 10,
        "pairwise_timeout": 10.0,
        "n_closest_pairs": 5,
        "patience": 5
    }
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "instances/E-n31-k7.vrp"
        
    solve_advanced(filepath, config=test_config, max_iterations=50)