#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Advanced CVRP Optimizer – CPLEX MIP
=====================================
Nghiệm khởi tạo: Clarke-Wright.
Tối ưu hóa: Single-Route MIP (TSP) và Pairwise MIP (2-vehicle VRP) bằng CPLEX.

Cấu trúc chính xác theo Phase 4 của bản MaxSAT:
  - Bước 4.1: Single route – duyệt toàn bộ, bão hòa rồi mới sang bước tiếp.
  - Bước 4.2: Pairwise – vòng lặp cho đến khi không cải thiện hoặc hết timeout.
              Mỗi khi tìm được cải thiện → break, tính lại danh sách cặp, lặp lại.

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
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "cplex")
os.makedirs(_LOG_DIR, exist_ok=True)
log_filename = os.path.join(_LOG_DIR, f"solver_cplex_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

GLOBAL_TIMEOUT_DEFAULT = 1200.0
SOLVER_NAME = "CPLEX-MIP"

# ──────────────────────────────────────────────────────────────────────────────
# OPTIMIZER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class AdvancedCVRPOptimizer:
    _POOL_CLEANUP_BUFFER = 2.0  # giây dành cho cleanup sau mỗi lần gọi CPLEX

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
        usable = rem - self._POOL_CLEANUP_BUFFER
        return min(config_timeout, max(0.0, usable))

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

    # --- Sắp xếp cặp tuyến theo khoảng cách gần nhất (y hệt pysat) ---
    def get_sorted_route_pairs(self, routes: Dict[int, List[int]]) -> List[Tuple[int, int]]:
        """
        Sinh ra TẤT CẢ các tổ hợp cặp xe,
        sắp xếp theo khoảng cách ngắn nhất giữa hai tuyến (gần nhất → xa nhất).
        """
        ids = list(routes.keys())
        if len(ids) < 2:
            return []
        scores = []
        for i, j in combinations(ids, 2):
            ri, rj = routes[i], routes[j]
            if not ri or not rj:
                continue
            min_dist = min(self.distances[a, b] for a in ri for b in rj)
            scores.append((min_dist, i, j))
        scores.sort(key=lambda x: x[0])
        return [(i, j) for _, i, j in scores]

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

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4: CPLEX MIP (cấu trúc y hệt run_phase_4_maxsat trong pysat)
    # ──────────────────────────────────────────────────────────────────────────
    def run_phase_4_cplex(
        self, routes: Dict[int, List[int]]
    ) -> Tuple[Dict[int, List[int]], int]:
        """
        CHẶNG 4: TỐI ƯU CHÍNH XÁC CPLEX MIP
        1. Tối ưu từng tuyến đơn lẻ (bão hòa toàn bộ rồi mới sang bước 2).
        2. Tối ưu từng cặp tuyến (vét cạn cho đến khi không giảm được nữa hoặc hết Timeout).
        """
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost = self.compute_total_cost(best_routes)

        logging.info(
            f"=== BẮT ĐẦU CHẶNG 4: CPLEX MIP CHÍNH XÁC | Cost: {best_cost} ==="
        )

        # ---------------------------------------------------------
        # BƯỚC 4.1: TỐI ƯU TUYẾN ĐƠN LẺ (SINGLE ROUTE)
        # ---------------------------------------------------------
        logging.info("  [GĐ 4.1] Giải bài toán TSP trên từng tuyến đơn...")
        single_imp_total = 0
        for v, route in list(best_routes.items()):
            if self._is_timed_out():
                self.stat_global_timeout = True
                break

            old_c = self.compute_route_cost(route)
            opt_r, opt_c = self.optimize_single_route_safe(route)

            if opt_c < old_c:
                best_routes[v] = opt_r
                single_imp_total += old_c - opt_c
                self.stat_single_improvements += 1

        if single_imp_total > 0:
            best_cost = self.compute_total_cost(best_routes)
            logging.info(
                f"    -> Single CPLEX cải thiện: {single_imp_total}. Cost mới: {best_cost}"
            )

        if self._is_timed_out():
            return best_routes, best_cost

        # ---------------------------------------------------------
        # BƯỚC 4.2: TỐI ƯU CẶP TUYẾN ĐƯỜNG (PAIRWISE)
        # ---------------------------------------------------------
        logging.info("  [GĐ 4.2] Xét từng cặp tuyến để phân bổ lại khách hàng...")

        pair_iter = 0

        while not self._is_timed_out():
            pair_iter += 1
            rem = self._remaining()
            rem_str = "∞" if rem is None else f"{rem:.0f}s"
            logging.info(f"  --- Pairwise Iter {pair_iter} (còn {rem_str}) ---")

            pairs = self.get_sorted_route_pairs(best_routes)

            pair_improved_in_this_iter = False
            for i, j in pairs:
                if self._is_timed_out():
                    self.stat_global_timeout = True
                    break

                r1, r2 = best_routes[i], best_routes[j]
                old_c = self.compute_route_cost(r1) + self.compute_route_cost(r2)

                o1, o2, nc, ok = self.optimize_route_pair_safe(r1, r2)

                if ok and nc < old_c:
                    best_routes[i], best_routes[j] = o1, o2
                    pair_improved_in_this_iter = True
                    self.stat_pairwise_improvements += 1

                    best_cost = self.compute_total_cost(best_routes)
                    logging.info(
                        f"    [P4.2] Pair ({i},{j}) giảm {old_c - nc} -> BEST = {best_cost}"
                    )

                    # Chạy lại single-opt trên 2 tuyến vừa thay đổi
                    for v in (i, j):
                        if self._is_timed_out():
                            break
                        old_sc = self.compute_route_cost(best_routes[v])
                        opt_r, opt_c = self.optimize_single_route_safe(best_routes[v])
                        if opt_c < old_sc:
                            best_routes[v] = opt_r
                            self.stat_single_improvements += 1
                            best_cost = self.compute_total_cost(best_routes)
                            logging.info(
                                f"      [S-re] Route {v}: {old_sc} → {opt_c}  (−{old_sc - opt_c}) | BEST = {best_cost}"
                            )

                    break  # Thay đổi cấu trúc → tính lại danh sách cặp gần nhất

            if not pair_improved_in_this_iter:
                logging.info(
                    "    -> Đã đạt cực trị toàn cục cho mọi cặp dưới ngưỡng. KẾT THÚC CHẶNG 4."
                )
                break

        return best_routes, best_cost

    # ──────────────────────────────────────────────────────────────────────────
    # ENTRY: chỉ chạy phase 4
    # ──────────────────────────────────────────────────────────────────────────
    def optimize(
        self, initial_routes: Dict[int, List[int]]
    ) -> Tuple[Dict[int, List[int]], int]:
        routes = {k: list(v) for k, v in initial_routes.items()}

        self._global_start = time.time()

        logging.info("=========================================")
        logging.info("BẮT ĐẦU OPTIMIZE  (100% CPLEX MIP)")
        logging.info(f" -> Max Single Size  : {self.max_single_size}")
        logging.info(f" -> Single Timeout   : {self.single_timeout}s")
        logging.info(f" -> Max Pairwise Size: {self.max_pairwise_size}")
        logging.info(f" -> Pairwise Timeout : {self.pairwise_timeout}s")
        logging.info(f" -> Global Timeout   : {self.global_timeout}s")
        logging.info(f" -> Chi phí khởi điểm: {self.compute_total_cost(routes)}")
        logging.info("=========================================")

        routes, best_cost = self.run_phase_4_cplex(routes)

        elapsed = time.time() - self._global_start
        logging.info(f"Hoàn thành trong {elapsed:.1f}s. Final Cost: {best_cost}")
        return routes, best_cost


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def solve_advanced(
    filepath: str,
    config: Dict[str, Any] = None,
    max_iterations: int = 150,
    target_cost: float = 0.0,
) -> Tuple[Dict[int, List[int]], int, Dict[str, Any]]:
    from classes.instance import Instance
    from classes.clarke_wright import ClarkeWright

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
        # Thực hiện khởi tạo với đúng số lượng xe n_vehicles (Hard Constraint)
        cw_time, cw_routes = ClarkeWright.run(cvrp, n_vehicles)
    except Exception as e:
        # TRƯỜNG HỢP UNSAT: Ngắt pipeline ngay lập tức nếu không tìm được nghiệm khả thi
        logging.error(f"❌ CW THẤT BẠI (UNSAT): Không thể đóng gói vào đúng K={n_vehicles} xe.")
        logging.error(f"Chi tiết: {str(e)}")
        
        # Trả về kết quả rỗng và trạng thái UNSAT để đồng bộ với Benchmark
        stats = {
            "solver_name": SOLVER_NAME if 'SOLVER_NAME' in locals() else "MIP-Solver",
            "single_imp_count": 0,
            "pairwise_imp_count": 0,
            "single_timeouts": 0,
            "pairwise_timeouts": 0,
            "global_timeout": False,
            "status": "UNSAT"
        }
        return {}, float('inf'), stats

    # THỐNG NHẤT ĐỊNH DẠNG LOG: Sử dụng .3f để đồng bộ việc parse dữ liệu tự động
    cw_cost = sum(r.cost for r in cw_routes.values())
    logging.info(f"  Kết quả CW: Cost = {cw_cost}, Time = {cw_time:.3f}s")
    routes = {
        i: list(route.value) for i, (_, route) in enumerate(cw_routes.items())
    }
    routes = {i: r for i, r in routes.items() if len(r) > 0}

    logging.info("--- Step 2: Advanced Optimization (CPLEX MIP) ---")
    start_time = time.time()
    optimizer = AdvancedCVRPOptimizer(
        distances=cvrp.distances,
        demands=np.array(cvrp.demands),
        capacity=cvrp.capacity,
        n_vehicles=n_vehicles,
        config=config,
    )
    opt_routes, opt_cost = optimizer.optimize(routes)
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
    final_int_cost = int(opt_cost)
    logging.info(f"  2. Chi phí sau CPLEX MIP:     {final_int_cost}")
    logging.info(f"  => TỔNG CẢI THIỆN:            {cw_cost - final_int_cost}")
    logging.info(
        f"  => Số lần CPLEX Single cải thiện  : {stats['single_imp_count']} lần"
    )
    logging.info(
        f"  => Số lần CPLEX Pairwise cải thiện: {stats['pairwise_imp_count']} lần"
    )
    logging.info(
        f"  => TỔNG THỜI GIAN CHẠY:       {cw_time + opt_time:.3f}s"
    )

    total_verify = sum(
        optimizer.compute_route_cost(r) for r in opt_routes.values() if r
    )

    return opt_routes, total_verify, stats