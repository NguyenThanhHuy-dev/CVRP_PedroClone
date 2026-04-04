#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Advanced CVRP Optimizer – CPLEX MIP
=====================================
Nghiệm khởi tạo: Clarke-Wright + Two-Opt.
Tối ưu hóa: Single-Route MIP (TSP) và Pairwise MIP (2-vehicle VRP) bằng CPLEX.

Global timeout bắt buộc: 1200 giây.
"""

import sys
import os
import time
import logging
import re
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from itertools import combinations

from cplex_optimizer import CplexSingleRouteOptimizer, CplexPairwiseRouteOptimizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- LOGGING ---
log_filename = f"solver_cplex_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

GLOBAL_TIMEOUT_DEFAULT = 1200.0



# ──────────────────────────────────────────────────────────────────────────────
# OPTIMIZER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class AdvancedCVRPOptimizer:
    def __init__(
        self,
        distances: np.ndarray,
        demands: np.ndarray,
        capacity: int,
        n_vehicles: int,
        config: Dict[str, Any] = None,
    ):
        self.distances  = distances
        self.demands    = demands
        self.capacity   = capacity
        self.n_vehicles = n_vehicles

        if config is None:
            config = {}

        self.max_single_size   = config.get("max_single_size",   30)
        self.single_timeout    = config.get("single_timeout",     5.0)
        self.max_pairwise_size = config.get("max_pairwise_size", 20)
        self.pairwise_timeout  = config.get("pairwise_timeout",  15.0)
        self.n_closest_pairs   = config.get("n_closest_pairs",  999)
        self.patience          = config.get("patience",          20)
        self.global_timeout    = config.get("global_timeout",   GLOBAL_TIMEOUT_DEFAULT)

        self.stat_single_improvements   = 0
        self.stat_pairwise_improvements = 0
        self.stat_single_timeouts       = 0
        self.stat_pairwise_timeouts     = 0
        self.stat_global_timeout        = False

        self._global_start: Optional[float] = None

    # --- Thời gian ---
    def _remaining(self) -> Optional[float]:
        if self.global_timeout is None or self._global_start is None:
            return None
        return max(0.0, self.global_timeout - (time.time() - self._global_start))

    def _effective_timeout(self, config_timeout: float) -> float:
        rem = self._remaining()
        if rem is None:
            return config_timeout
        return min(config_timeout, max(0.0, rem))

    def _is_timed_out(self) -> bool:
        rem = self._remaining()
        return rem is not None and rem <= 0

    # --- Chi phí ---
    def compute_route_cost(self, route: List[int]) -> int:
        if not route:
            return 0
        cost = self.distances[0, route[0]]
        for i in range(len(route) - 1):
            cost += self.distances[route[i], route[i + 1]]
        cost += self.distances[route[-1], 0]
        return int(cost)

    def compute_total_cost(self, routes: Dict[int, List[int]]) -> int:
        return sum(self.compute_route_cost(r) for r in routes.values())

    # --- Cặp gần nhất ---
    def find_closest_route_pairs(self, routes: Dict[int, List[int]]) -> List[Tuple[int, int]]:
        route_ids = list(routes.keys())
        if len(route_ids) < 2:
            return []
        pair_scores = []
        for i, j in combinations(route_ids, 2):
            ri, rj = routes[i], routes[j]
            if not ri or not rj:
                continue
            min_d = min(self.distances[c1, c2] for c1 in ri for c2 in rj)
            pair_scores.append((min_d, i, j))
        pair_scores.sort()
        return [(i, j) for _, i, j in pair_scores[: self.n_closest_pairs]]

    # --- Single-Route MIP ---
    def optimize_single_route_safe(self, route: List[int]) -> Tuple[List[int], int]:
        if len(route) <= 1:
            return route, self.compute_route_cost(route)
        if len(route) > self.max_single_size:
            return route, self.compute_route_cost(route)

        eff = self._effective_timeout(self.single_timeout)
        if eff <= 0:
            self.stat_single_timeouts += 1
            return route, self.compute_route_cost(route)

        try:
            opt = CplexSingleRouteOptimizer(route, self.distances, timeout=eff)
            opt_r, opt_c = opt.optimize()
            return opt_r, opt_c
        except Exception as e:
            logging.error(f"Lỗi CPLEX Single: {e}")
            self.stat_single_timeouts += 1
            return route, self.compute_route_cost(route)

    # --- Pairwise MIP ---
    def optimize_route_pair_safe(
        self, route1: List[int], route2: List[int]
    ) -> Tuple[List[int], List[int], int, bool]:
        if not route1 and not route2:
            return [], [], 0, True
        if len(route1) + len(route2) > self.max_pairwise_size:
            return [], [], float("inf"), False

        eff = self._effective_timeout(self.pairwise_timeout)
        if eff <= 0:
            self.stat_pairwise_timeouts += 1
            return [], [], float("inf"), False

        try:
            opt = CplexPairwiseRouteOptimizer(
                route1, route2, self.distances, self.demands, self.capacity, timeout=eff
            )
            r1, r2, cost, success = opt.optimize()
            return r1, r2, cost, success
        except Exception as e:
            logging.error(f"Lỗi CPLEX Pairwise: {e}")
            self.stat_pairwise_timeouts += 1
            return [], [], float("inf"), False

    # --- VÒNG LẶP CHÍNH ---
    def optimize(
        self, initial_routes: Dict[int, List[int]], max_iterations: int = 100
    ) -> Tuple[Dict[int, List[int]], int]:
        routes      = {k: list(v) for k, v in initial_routes.items()}
        best_cost   = self.compute_total_cost(routes)
        best_routes = {k: list(v) for k, v in routes.items()}

        self._global_start = time.time()

        logging.info("=========================================")
        logging.info("BẮT ĐẦU OPTIMIZE  (100% CPLEX MIP)")
        logging.info(f" -> Max Single Size  : {self.max_single_size}")
        logging.info(f" -> Single Timeout   : {self.single_timeout}s")
        logging.info(f" -> Max Pairwise Size: {self.max_pairwise_size}")
        logging.info(f" -> Pairwise Timeout : {self.pairwise_timeout}s")
        logging.info(f" -> Global Timeout   : {self.global_timeout}s")
        logging.info(f" -> Closest Pairs    : {self.n_closest_pairs}")
        logging.info(f" -> Patience         : {self.patience}")
        logging.info(f" -> Max Iterations   : {max_iterations}")
        logging.info(f" -> Chi phí khởi điểm: {best_cost}")
        logging.info("=========================================")

        no_improve = 0
        iteration  = 0

        while iteration < max_iterations and no_improve < self.patience:
            if self._is_timed_out():
                self.stat_global_timeout = True
                logging.info(f"[GlobalTimeout] Dừng trước iteration {iteration+1}.")
                break

            iteration += 1
            rem_str = f"{self._remaining():.0f}s" if self._remaining() is not None else "∞"
            logging.info(f"--- Iter {iteration}/{max_iterations}  (còn {rem_str}) ---")

            iter_improved = False

            # BƯỚC 1: Tối ưu từng tuyến đơn bằng CPLEX TSP
            for v, route in list(routes.items()):
                if self._is_timed_out():
                    self.stat_global_timeout = True
                    break
                old_c = self.compute_route_cost(route)
                opt_r, opt_c = self.optimize_single_route_safe(route)
                if opt_c < old_c:
                    routes[v] = opt_r
                    self.stat_single_improvements += 1
                    iter_improved = True
                    logging.info(f"  [Single] Route {v}: {old_c} → {opt_c}  (−{old_c - opt_c})")

            if self._is_timed_out():
                self.stat_global_timeout = True
                break

            # BƯỚC 2: Tối ưu cặp tuyến bằng CPLEX 2-vehicle VRP
            pairs = self.find_closest_route_pairs(routes)
            for i, j in pairs:
                if self._is_timed_out():
                    self.stat_global_timeout = True
                    break
                r1, r2 = routes[i], routes[j]
                old_c  = self.compute_route_cost(r1) + self.compute_route_cost(r2)
                o1, o2, new_c, success = self.optimize_route_pair_safe(r1, r2)
                if success and new_c < old_c:
                    routes[i], routes[j] = o1, o2
                    self.stat_pairwise_improvements += 1
                    iter_improved = True
                    logging.info(f"  [Pair] ({i},{j}): {old_c} → {new_c}  (−{old_c - new_c})")
                    break  # cấu trúc thay đổi → cần tính lại danh sách cặp

            # BƯỚC 3: Đánh giá
            cur = self.compute_total_cost(routes)
            if cur < best_cost - 0.5:
                best_cost   = cur
                best_routes = {k: list(v) for k, v in routes.items()}
                no_improve  = 0
                logging.info(f"=> ITER {iteration}: BEST MỚI = {best_cost}")
            else:
                no_improve += 1
                logging.info(
                    f"=> ITER {iteration}: Không cải thiện ({no_improve}/{self.patience})"
                )

        elapsed    = time.time() - self._global_start
        final_cost = self.compute_total_cost(best_routes)
        logging.info(f"Hoàn thành trong {elapsed:.1f}s. Final Cost: {final_cost}")
        return best_routes, final_cost


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def solve_advanced(
    filepath: str,
    config: Dict[str, Any] = None,
    max_iterations: int = 150,
) -> Tuple[Dict[int, List[int]], int, Dict[str, Any]]:
    import re
    from classes.instance import Instance
    from classes.clarke_wright import ClarkeWright
    from classes.two_opt import TwoOpt

    if config is None:
        config = {}
    # Bắt buộc global_timeout = 1200s
    config.setdefault("global_timeout", GLOBAL_TIMEOUT_DEFAULT)

    logging.info("=" * 70)
    logging.info(f"CPLEX MIP SOLVER  |  FILE: {os.path.basename(filepath)}")
    logging.info("=" * 70)

    cvrp = Instance(filepath)
    cvrp.load()
    cvrp.distances = np.floor(cvrp.distances + 0.5).astype(int)
    n_vehicles = int(re.search(r"-k(\d+)", filepath).group(1))

    logging.info(
        f"Thông tin dữ liệu: Dimension={cvrp.dimension}, K={n_vehicles}, Capacity={cvrp.capacity}"
    )

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
    logging.info(
        f"  Kết quả Two-Opt: Cost = {two_opt_cost}, Time = {two_opt_time:.3f}s"
    )

    routes = {
        i: list(route.value) for i, (_, route) in enumerate(two_opt_routes.items())
    }
    routes = {i: r for i, r in routes.items() if len(r) > 0}

    logging.info("--- Step 3: Advanced Optimization (CPLEX MIP) ---")
    start_time = time.time()
    optimizer = AdvancedCVRPOptimizer(
        distances=cvrp.distances,
        demands=np.array(cvrp.demands),
        capacity=cvrp.capacity,
        n_vehicles=n_vehicles,
        config=config,
    )
    opt_routes, opt_cost = optimizer.optimize(routes, max_iterations=max_iterations)
    opt_time = time.time() - start_time

    stats = {
        "single_imp_count":    optimizer.stat_single_improvements,
        "pairwise_imp_count":  optimizer.stat_pairwise_improvements,
        "single_timeouts":     optimizer.stat_single_timeouts,
        "pairwise_timeouts":   optimizer.stat_pairwise_timeouts,
        "global_timeout":      optimizer.stat_global_timeout,
    }

    logging.info("=" * 70)
    logging.info("BÁO CÁO TỔNG KẾT (SUMMARY)")
    logging.info("=" * 70)
    logging.info(f"  1. Chi phí Clarke-Wright:     {cw_cost}")
    logging.info(f"  2. Chi phí sau Two-Opt:       {two_opt_cost}")
    final_int_cost = int(opt_cost)
    logging.info(f"  3. Chi phí sau CPLEX MIP:     {final_int_cost}")
    logging.info(f"  => TỔNG CẢI THIỆN:            {cw_cost - final_int_cost}")
    logging.info(
        f"  => Số lần CPLEX Single cải thiện  : {stats['single_imp_count']} lần"
    )
    logging.info(
        f"  => Số lần CPLEX Pairwise cải thiện: {stats['pairwise_imp_count']} lần"
    )
    logging.info(
        f"  => TỔNG THỜI GIAN CHẠY:       {cw_time + two_opt_time + opt_time:.3f}s"
    )

    total_verify = sum(
        optimizer.compute_route_cost(r) for r in opt_routes.values() if r
    )

    return opt_routes, total_verify, stats