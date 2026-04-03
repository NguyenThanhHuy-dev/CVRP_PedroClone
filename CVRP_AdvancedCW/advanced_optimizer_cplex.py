#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Advanced CVRP Optimizer using 100% CPLEX (Single & Pairwise)
=============================================================
"""

import sys
import os
import time
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from itertools import combinations

from cplex_optimizer import CplexPairwiseRouteOptimizer, CplexSingleRouteOptimizer

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
# ADVANCED CVRP OPTIMIZER (100% CPLEX)
# =====================================================================

# =====================================================================
# ADVANCED CVRP OPTIMIZER (100% CPLEX)
# =====================================================================
class AdvancedCVRPOptimizer:
    def __init__(self, distances: np.ndarray, demands: np.ndarray, capacity: int, n_vehicles: int, config: Dict[str, Any] = None):
        self.distances = distances
        self.demands = demands
        self.capacity = capacity
        self.n_vehicles = n_vehicles
        self.penalty_rate = 10000.0  # Hệ số phạt vượt tải (Soft Capacity)
        
        if config is None:
            config = {}
            
        self.max_single_size = config.get("max_single_size", 40)
        self.single_timeout = config.get("single_timeout", 5.0)
        self.max_pairwise_size = config.get("max_pairwise_size", 25)
        self.pairwise_timeout = config.get("pairwise_timeout", 10.0)
        self.n_closest_pairs = config.get("n_closest_pairs", 5)
        self.patience = config.get("patience", 5)
        self.global_timeout = config.get("global_timeout", None) # ĐÃ THÊM GLOBAL TIMEOUT
        
        self.stat_single_improvements = 0
        self.stat_pairwise_improvements = 0
        self.stat_single_timeouts = 0
        self.stat_pairwise_timeouts = 0
        self.stat_single_timeouts = 0
        self.stat_pairwise_timeouts = 0
        self.stat_global_timeout = False
        
        self._global_start: Optional[float] = None

    # --- HÀM TÍNH THỜI GIAN ĐỘNG ---
    def _remaining(self) -> Optional[float]:
        if self.global_timeout is None or self._global_start is None:
            return None
        return max(0.0, self.global_timeout - (time.time() - self._global_start))

    def _effective_timeout(self, config_timeout: float) -> float:
        rem = self._remaining()
        if rem is None:
            return config_timeout
        return min(config_timeout, max(0.0, rem))

    # --- HÀM TÍNH TOÁN COST ---
    def compute_route_cost(self, route: List[int]) -> int:
        if not route: return 0
        cost = self.distances[0, route[0]]
        for i in range(len(route) - 1):
            cost += self.distances[route[i], route[i + 1]]
        cost += self.distances[route[-1], 0]
        return cost

    def compute_pure_cost(self, routes: Dict[int, List[int]]) -> float:
        return sum(self.compute_route_cost(r) for r in routes.values())

    def compute_total_cost_with_penalty(self, routes: Dict[int, List[int]]) -> float:
        dist = self.compute_pure_cost(routes)
        penalty = sum(max(0, sum(self.demands[c] for c in r) - self.capacity) for r in routes.values())
        return dist + (self.penalty_rate * penalty)

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

    # --- HÀM TỐI ƯU GỌI BỘ GIẢI CPLEX ---
    def optimize_single_route_safe(self, route: List[int]) -> Tuple[List[int], int]:
        if len(route) <= 1: return route, self.compute_route_cost(route)
        if len(route) > self.max_single_size: return route, self.compute_route_cost(route)

        eff = self._effective_timeout(self.single_timeout)
        if eff <= 0: 
            self.stat_single_timeouts += 1
            return route, self.compute_route_cost(route)

        try:
            # GỌI LỚP CPLEX SINGLE TỪ FILE HIỆN TẠI
            # (Đảm bảo lớp CplexSingleRouteOptimizer có tồn tại ở phần trên của file)
            opt = CplexSingleRouteOptimizer(route, self.distances, timeout=eff)
            opt_r, opt_c = opt.optimize()
            return opt_r, opt_c
        except Exception as e:
            logging.error(f"Lỗi Solver Single (CPLEX): {e}")
            self.stat_single_timeouts += 1
            return route, self.compute_route_cost(route)

    def optimize_route_pair_safe(self, route1: List[int], route2: List[int]) -> Tuple[List[int], List[int], int, bool]:
        if not route1 and not route2: return [], [], 0, True
        if len(route1) + len(route2) > self.max_pairwise_size: return [], [], float('inf'), False 

        eff = self._effective_timeout(self.pairwise_timeout)
        if eff <= 0: 
            self.stat_pairwise_timeouts += 1
            return [], [], float('inf'), False

        try:
            # GỌI LỚP CPLEX PAIRWISE TỪ FILE HIỆN TẠI
            # (Đảm bảo lớp CplexPairwiseRouteOptimizer có tồn tại ở phần trên của file)
            opt = CplexPairwiseRouteOptimizer(route1, route2, self.distances, self.demands, self.capacity, timeout=eff)
            r1, r2, cost, success = opt.optimize()
            return r1, r2, cost, success
        except Exception as e:
            logging.error(f"Lỗi Solver Pairwise (CPLEX): {e}")
            self.stat_pairwise_timeouts += 1
            return [], [], float('inf'), False

    # --- HÀM RELOCATE VÀ EXCHANGE ---
    def try_relocate(self, routes: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], float, bool]:
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost = self.compute_total_cost_with_penalty(routes)
        improved = False
        route_ids = list(routes.keys())
        for src_id in route_ids:
            for dst_id in route_ids:
                if src_id == dst_id: continue
                src_route = routes[src_id]
                dst_route = routes[dst_id]
                for i, customer in enumerate(src_route):
                    for j in range(len(dst_route) + 1):
                        new_src = src_route[:i] + src_route[i+1:]
                        new_dst = dst_route[:j] + [customer] + dst_route[j:]
                        test_routes = {k: list(v) for k, v in routes.items()}
                        test_routes[src_id] = new_src
                        test_routes[dst_id] = new_dst
                        total_cost = self.compute_total_cost_with_penalty(test_routes)
                        if total_cost < best_cost - 0.001:
                            best_cost, best_routes, improved = total_cost, test_routes, True
        return best_routes, best_cost, improved

    def try_exchange(self, routes: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], float, bool]:
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost = self.compute_total_cost_with_penalty(routes)
        improved = False
        route_ids = list(routes.keys())
        for id1, id2 in combinations(route_ids, 2):
            route1, route2 = routes[id1], routes[id2]
            for i, c1 in enumerate(route1):
                for j, c2 in enumerate(route2):
                    new_route1 = route1[:i] + [c2] + route1[i+1:]
                    new_route2 = route2[:j] + [c1] + route2[j+1:]
                    test_routes = {k: list(v) for k, v in routes.items()}
                    test_routes[id1] = new_route1
                    test_routes[id2] = new_route2
                    total_cost = self.compute_total_cost_with_penalty(test_routes)
                    if total_cost < best_cost - 0.001:
                        best_cost, best_routes, improved = total_cost, test_routes, True
        return best_routes, best_cost, improved

    # --- KIẾN TRÚC VND LOOP ĐÃ CẬP NHẬT ---
    def optimize(self, initial_routes: Dict[int, List[int]], max_iterations: int = 100) -> Tuple[Dict[int, List[int]], float]:
        routes = {k: list(v) for k, v in initial_routes.items()}
        best_cost = self.compute_total_cost_with_penalty(routes)
        
        self._global_start = time.time()
        
        logging.info("=========================================")
        logging.info("BẮT ĐẦU VÒNG LẶP OPTIMIZE CPLEX (HYBRID LNS / VND)")
        logging.info(f" -> Max Single Size  : {self.max_single_size}")
        logging.info(f" -> Single Timeout   : {self.single_timeout}s")
        logging.info(f" -> Max Pairwise Size: {self.max_pairwise_size}")
        logging.info(f" -> Pairwise Timeout : {self.pairwise_timeout}s")
        logging.info(f" -> Global Timeout   : {'không giới hạn' if self.global_timeout is None else str(self.global_timeout) + 's'}")
        logging.info(f" -> Patience         : {self.patience}")
        logging.info(f" -> Max Iterations   : {max_iterations}")
        logging.info(f"Chi phí khởi điểm (kể cả phạt): {best_cost:.2f}")
        logging.info("=========================================")
        
        iteration = 0
        no_improvement_count = 0
        
        while iteration < max_iterations and no_improvement_count < self.patience:
            rem = self._remaining()
            if rem is not None and rem <= 0:
                self.stat_global_timeout = True
                logging.info(f"[Global Timeout] Dừng trước iteration {iteration+1}.")
                break

            iteration += 1
            rem_str = "∞" if rem is None else f"{rem:.0f}s"
            logging.info(f"--- Iteration {iteration}/{max_iterations} (còn {rem_str}) ---")
            
            # BƯỚC 1: VÒNG LẶP TRONG (Vắt kiệt Toán tử Nhẹ)
            local_improved = True
            inner_loop_count = 0
            while local_improved:
                inner_loop_count += 1
                local_improved = False
                if self._remaining() is not None and self._remaining() <= 0:
                    self.stat_global_timeout = True
                    break

                routes, new_r_cost, reloc_imp = self.try_relocate(routes)
                if reloc_imp:
                    local_improved = True
                    logging.info(f"  [Inner {inner_loop_count}] Relocate → {new_r_cost:.2f}")

                routes, new_e_cost, exch_imp = self.try_exchange(routes)
                if exch_imp:
                    local_improved = True
                    logging.info(f"  [Inner {inner_loop_count}] Exchange → {new_e_cost:.2f}")

                single_imp = 0
                for v, route in routes.items():
                    old_c = self.compute_route_cost(route)
                    opt_r, opt_c = self.optimize_single_route_safe(route)
                    if opt_c < old_c:
                        routes[v] = opt_r
                        local_improved = True
                        single_imp += (old_c - opt_c)
                        self.stat_single_improvements += 1
                if single_imp > 0:
                    logging.info(f"  [Inner {inner_loop_count}] Single Route giảm: {single_imp}")

            # BƯỚC 2: TOÁN TỬ NẶNG (Pairwise - Phá vỡ cấu trúc bằng CPLEX)
            if self._remaining() is not None and self._remaining() <= 0: 
                self.stat_global_timeout = True
                break
            
            pairs = self.find_closest_route_pairs(routes)
            pairwise_success = False
            for i, j in pairs:
                r1, r2 = routes[i], routes[j]
                old_c = self.compute_route_cost(r1) + self.compute_route_cost(r2)
                opt1, opt2, new_c, success = self.optimize_route_pair_safe(r1, r2)
                if success and new_c < old_c:
                    routes[i], routes[j] = opt1, opt2
                    pairwise_success = True
                    self.stat_pairwise_improvements += 1
                    logging.info(f"  [P4] Pairwise ({i},{j}) thành công! Giảm: {old_c - new_c}")
                    break # Chỉ cần sửa 1 cặp là cấu trúc xô lệch, quay lại Relocate ăn tiếp

            # BƯỚC 3: ĐÁNH GIÁ
            current_cost = self.compute_total_cost_with_penalty(routes)
            if current_cost < best_cost - 0.001:
                best_cost = current_cost
                no_improvement_count = 0
                logging.info(f"=> ITER {iteration}: BEST COST MỚI = {best_cost:.2f}")
            else:
                no_improvement_count += 1
                logging.info(f"=> ITER {iteration}: Không cải thiện (no_improve={no_improvement_count}/{self.patience})")

        final_pure_cost = self.compute_pure_cost(routes)
        total_elapsed = time.time() - self._global_start
        logging.info(f"Hoàn thành tối ưu trong {total_elapsed:.1f}s. Final Pure Cost: {final_pure_cost}")
        return routes, final_pure_cost
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
    
    logging.info("--- Step 1: Clarke-Wright Heuristic ---")
    try:
        cw_time, cw_routes = ClarkeWright.run(cvrp, n_vehicles)
    except Exception:
        logging.warning("⚠️ Clarke-Wright kẹt với K giới hạn. Chạy fallback K=999...")
        cw_time, cw_routes = ClarkeWright.run(cvrp, 999)
    cw_cost = sum(route.cost for route in cw_routes.values())
    logging.info(f"  Kết quả CW: Cost = {cw_cost}, Time = {cw_time:.3f}s")
    
    logging.info("--- Step 2: Two-Opt Local Search ---")
    two_opt_time, two_opt_routes = TwoOpt.run(cw_routes)
    two_opt_cost = sum(route.cost for route in two_opt_routes.values())
    logging.info(f"  Kết quả Two-Opt: Cost = {two_opt_cost}, Time = {two_opt_time:.3f}s")
    
    routes = {i: list(route.value) for i, (_, route) in enumerate(two_opt_routes.items())}
    routes = {i: r for i, r in routes.items() if len(r) > 0}
    
    logging.info("--- Step 3: Advanced Optimization (100% CPLEX) ---")
    start_time = time.time()
    optimizer = AdvancedCVRPOptimizer(
        distances=cvrp.distances,
        demands=np.array(cvrp.demands),
        capacity=cvrp.capacity,
        n_vehicles=n_vehicles,
        config=config
    )
    opt_routes, opt_cost = optimizer.optimize(routes, max_iterations=max_iterations)
    opt_time = time.time() - start_time
    
    stats = {
        "single_imp_count": optimizer.stat_single_improvements,
        "pairwise_imp_count": optimizer.stat_pairwise_improvements,
        "single_timeouts": optimizer.stat_single_timeouts,     # <-- THÊM DÒNG NÀY
        "pairwise_timeouts": optimizer.stat_pairwise_timeouts, # <-- THÊM DÒNG NÀY
        "global_timeout": optimizer.stat_global_timeout
    }
    
    logging.info("=" * 70)
    logging.info("BÁO CÁO TỔNG KẾT (SUMMARY)")
    logging.info("=" * 70)
    logging.info(f"  1. Chi phí Clarke-Wright:     {cw_cost}")
    logging.info(f"  2. Chi phí sau Two-Opt:       {two_opt_cost}")
    final_int_cost = int(opt_cost)
    logging.info(f"  3. Chi phí sau CPLEX:        {final_int_cost}")
    logging.info(f"  => TỔNG CẢI THIỆN:            {cw_cost - final_int_cost}")
    logging.info(f"  => Số lần CPLEX Single cải thiện  : {stats['single_imp_count']} lần")
    logging.info(f"  => Số lần CPLEX Pairwise cải thiện: {stats['pairwise_imp_count']} lần")
    logging.info(f"  => TỔNG THỜI GIAN CHẠY:       {cw_time + two_opt_time + opt_time:.3f}s")
    
    total_verify = 0
    for v, route in sorted(opt_routes.items()):
        if route:
            demand = sum(cvrp.demands[c] for c in route)
            cost = optimizer.compute_route_cost(route)
            total_verify += cost
    
    return opt_routes, total_verify, stats