#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import os
import time
import logging
import multiprocessing
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from itertools import combinations
from pysat.pb import PBEnc
from classes.two_opt import TwoOpt
from classes.or_opt import OrOpt
from classes.cross_exchange import CrossExchange
from classes.alns import ALNS
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "pysat")
os.makedirs(_LOG_DIR, exist_ok=True)
log_filename = os.path.join(_LOG_DIR, f"solver_run_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

SOLVER_NAME = "MaxSAT-RC2"


class SingleRouteOptimizer:
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

    def _nid(self) -> int:
        self.var_id += 1
        return self.var_id

    def _exactly_one(self, lits: List[int]):
        self.wcnf.append(lits)
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                self.wcnf.append([-lits[i], -lits[j]])

    def optimize(self) -> Tuple[List[int], int]:
        # Biến kết nối
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.conNet[i][j] = self._nid()

        # Biến thứ tự (customer-customer only)
        for i in range(1, self.n):
            for j in range(i + 1, self.n):
                v = self._nid()
                self.rchNet[i][j] = v
                self.rchNet[j][i] = -v

        # Mệnh đề mềm: chi phí cạnh
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.local_dist[i, j] > 0:
                    self.wcnf.append(
                        [-self.conNet[i][j]], weight=int(self.local_dist[i, j])
                    )

        # Ràng buộc 1: l(i,j) → r(i,j)
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i != j and self.rchNet[i][j] != 0:
                    self.wcnf.append([-self.conNet[i][j], self.rchNet[i][j]])

        # Ràng buộc 2: Transitivity
        for a in range(1, self.n):
            for b in range(1, self.n):
                if a == b:
                    continue
                for c in range(1, self.n):
                    if c == a or c == b:
                        continue
                    r_ab = self.rchNet[a][b]
                    r_bc = self.rchNet[b][c]
                    r_ac = self.rchNet[a][c]
                    if r_ab != 0 and r_bc != 0 and r_ac != 0:
                        self.wcnf.append([-r_ab, -r_bc, r_ac])

        # Ràng buộc 3: Chain law
        for a in range(1, self.n):
            for b in range(1, self.n):
                if a == b:
                    continue
                for c in range(1, self.n):
                    if c == a or c == b:
                        continue
                    r_ab = self.rchNet[a][b]
                    r_bc = self.rchNet[b][c]
                    l_ac = self.conNet[a][c]
                    if r_ab != 0 and r_bc != 0 and l_ac != 0:
                        self.wcnf.append([-r_ab, -r_bc, -l_ac])

        # Ràng buộc 4: Depot out exactly-one
        self._exactly_one([self.conNet[0][j] for j in range(1, self.n)])

        # Ràng buộc 5: Depot in exactly-one
        self._exactly_one([self.conNet[i][0] for i in range(1, self.n)])

        # Ràng buộc 6: Customer degree exactly-one in + out
        for c in range(1, self.n):
            self._exactly_one([self.conNet[i][c] for i in range(self.n) if i != c])
            self._exactly_one([self.conNet[c][j] for j in range(self.n) if j != c])

        # Ràng buộc 7: Depot-first
        for j in range(1, self.n):
            depot_to_j = self.conNet[0][j]
            for k in range(1, self.n):
                if k == j:
                    continue
                self.wcnf.append([-depot_to_j, -self.conNet[k][j]])

        # Giải
        with RC2(self.wcnf, verbose=0) as solver:
            model = solver.compute()
            if model:
                positive = set(v for v in model if v > 0)
                route, current, visited = [], 0, {0}
                for _ in range(self.n - 1):
                    for j in range(self.n):
                        if j not in visited and self.conNet[current][j] in positive:
                            gid = self.local_to_global[j]
                            if gid != 0:
                                route.append(gid)
                            visited.add(j)
                            current = j
                            break
                return route, solver.cost

        orig = int(self.local_dist[0, 1])
        for i in range(1, self.n - 1):
            orig += int(self.local_dist[i, i + 1])
        orig += int(self.local_dist[self.n - 1, 0])
        return self.customers, orig


class PairwiseRouteOptimizer:

    def __init__(
        self, c1: List[int], c2: List[int], dist: np.ndarray, dem: np.ndarray, cap: int
    ):
        self.c1, self.c2 = c1, c2
        self.all_customers = c1 + c2
        self.n = len(self.all_customers) + 1
        self.dist, self.dem, self.cap = dist, dem, cap
        self.l2g = {0: 0}
        for i, c in enumerate(self.all_customers):
            self.l2g[i + 1] = c
        self.ldist = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                self.ldist[i, j] = dist[self.l2g[i], self.l2g[j]]
        self.var_id = 0
        self.con = [[[0] * self.n for _ in range(self.n)] for _ in range(2)]
        self.rch = [[[0] * self.n for _ in range(self.n)] for _ in range(2)]
        self.asn = [0] * self.n
        self.wcnf = WCNF()

    def _nid(self):
        self.var_id += 1
        return self.var_id

    def optimize(self) -> Tuple[List[int], List[int], int, bool]:
        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        self.con[v][i][j] = self._nid()
        for v in range(2):
            for i in range(1, self.n):
                for j in range(i + 1, self.n):
                    vr = self._nid()
                    self.rch[v][i][j] = vr
                    self.rch[v][j][i] = -vr
        for i in range(1, self.n):
            self.asn[i] = self._nid()

        # Mệnh đề mềm
        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i != j and self.ldist[i, j] > 0:
                        self.wcnf.append(
                            [-self.con[v][i][j]], weight=int(self.ldist[i, j])
                        )

        # Assignment consistency
        for i in range(1, self.n):
            for v in range(2):
                sign = 1 if v == 1 else -1
                in_e = [self.con[v][j][i] for j in range(self.n) if j != i]
                self.wcnf.append([-sign * self.asn[i]] + in_e)
                for j in range(self.n):
                    if j != i:
                        self.wcnf.append([sign * self.asn[i], -self.con[v][j][i]])
                        self.wcnf.append([sign * self.asn[i], -self.con[v][i][j]])

        # Implication + Transitivity + Chain law (per vehicle)
        for v in range(2):
            for i in range(1, self.n):
                for j in range(1, self.n):
                    if i != j and self.rch[v][i][j] != 0:
                        self.wcnf.append([-self.con[v][i][j], self.rch[v][i][j]])
            for a in range(1, self.n):
                for b in range(1, self.n):
                    if a == b:
                        continue
                    for c in range(1, self.n):
                        if c in (a, b):
                            continue
                        rab, rbc, rac = (
                            self.rch[v][a][b],
                            self.rch[v][b][c],
                            self.rch[v][a][c],
                        )
                        if rab != 0 and rbc != 0 and rac != 0:
                            self.wcnf.append([-rab, -rbc, rac])
            for a in range(1, self.n):
                for b in range(1, self.n):
                    if a == b:
                        continue
                    for c in range(1, self.n):
                        if c in (a, b):
                            continue
                        rab, rbc, lac = (
                            self.rch[v][a][b],
                            self.rch[v][b][c],
                            self.con[v][a][c],
                        )
                        if rab != 0 and rbc != 0 and lac != 0:
                            self.wcnf.append([-rab, -rbc, -lac])

        # Depot degree per vehicle (at-most-one; at-least-one không cần vì
        # mỗi xe phải phục vụ ít nhất 1 khách — được đảm bảo bởi flow)
        for v in range(2):
            out_d = [self.con[v][0][j] for j in range(1, self.n)]
            in_d = [self.con[v][i][0] for i in range(1, self.n)]
            self.wcnf.append(out_d)
            for x in range(len(out_d)):
                for y in range(x + 1, len(out_d)):
                    self.wcnf.append([-out_d[x], -out_d[y]])
            self.wcnf.append(in_d)
            for x in range(len(in_d)):
                for y in range(x + 1, len(in_d)):
                    self.wcnf.append([-in_d[x], -in_d[y]])

        # Flow conservation per customer per vehicle
        for i in range(1, self.n):
            for v in range(2):
                in_e = [self.con[v][j][i] for j in range(self.n) if j != i]
                out_e = [self.con[v][i][j] for j in range(self.n) if j != i]
                for x in range(len(in_e)):
                    for y in range(x + 1, len(in_e)):
                        self.wcnf.append([-in_e[x], -in_e[y]])
                for x in range(len(out_e)):
                    for y in range(x + 1, len(out_e)):
                        self.wcnf.append([-out_e[x], -out_e[y]])
                for ie in in_e:
                    self.wcnf.append([-ie] + out_e)
            # Mỗi customer phải được thăm bởi đúng 1 xe (at-least-one toàn cục)
            alle = []
            for v in range(2):
                alle.extend([self.con[v][j][i] for j in range(self.n) if j != i])
            self.wcnf.append(alle)

        with RC2(self.wcnf, verbose=0) as solver:
            model = solver.compute()
            if model:
                pos = set(v for v in model if v > 0)
                rts = [[], []]
                for v in range(2):
                    cur, vis = 0, {0}
                    for _ in range(self.n - 1):
                        for j in range(self.n):
                            if j not in vis and self.con[v][cur][j] in pos:
                                gid = self.l2g[j]
                                if gid != 0:
                                    rts[v].append(gid)
                                vis.add(j)
                                cur = j
                                break
                        else:
                            break
                if (
                    sum(self.dem[c] for c in rts[0]) > self.cap
                    or sum(self.dem[c] for c in rts[1]) > self.cap
                ):
                    return [], [], float("inf"), False
                return rts[0], rts[1], solver.cost, True
        return [], [], float("inf"), False


def _solve_single_worker(customers, distances):
    return SingleRouteOptimizer(customers, distances).optimize()


def _solve_pairwise_worker(c1, c2, distances, demands, capacity):
    return PairwiseRouteOptimizer(c1, c2, distances, demands, capacity).optimize()


class AdvancedCVRPOptimizer:
    _POOL_CLEANUP_BUFFER = 2.0  # giây dành cho pool.terminate()/join()

    def __init__(
        self,
        distances: np.ndarray,
        demands: np.ndarray,
        capacity: int,
        n_vehicles: int,
        config: Dict[str, Any] = None,
    ):
        self.distances = distances
        self.demands = demands
        self.capacity = capacity
        self.n_vehicles = n_vehicles

        if config is None:
            config = {}

        self.max_single_size = config.get("max_single_size", 11)
        self.single_timeout = config.get("single_timeout", 10.0)
        self.max_pairwise_size = config.get("max_pairwise_size", 9)
        self.pairwise_timeout = config.get("pairwise_timeout", 20.0)
        self.global_timeout = config.get("global_timeout", None)
        self.stat_single_improvements = 0
        self.stat_pairwise_improvements = 0
        self.stat_single_timeouts = 0
        self.stat_pairwise_timeouts = 0
        self.stat_global_timeout = False

        self._global_start: Optional[float] = None

    # ------------------------------------------------------------------
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

    def compute_route_cost(self, route: List[int]) -> int:
        if not route:
            return 0
        cost = self.distances[0, route[0]]
        for i in range(len(route) - 1):
            cost += self.distances[route[i], route[i + 1]]
        cost += self.distances[route[-1], 0]
        return cost

    def compute_total_cost(self, routes: Dict[int, List[int]]) -> float:
        return sum(self.compute_route_cost(r) for r in routes.values())

    def get_sorted_route_pairs(self, routes: Dict[int, List[int]]) -> List[Tuple[int, int]]:
        """ 
        Sinh ra TẤT CẢ các tổ hợp cặp xe, 
        nhưng SẮP XẾP CHÚNG THEO KHOẢNG CÁCH (Gần nhất -> Xa nhất)
        để Max-SAT ưu tiên giải các cặp có xác suất cải thiện cao trước.
        """
        ids = list(routes.keys())
        if len(ids) < 2:
            return []
            
        scores = []
        for i, j in combinations(ids, 2):
            ri, rj = routes[i], routes[j]
            if not ri or not rj:
                continue
            # Tính khoảng cách ngắn nhất giữa 1 điểm của xe I và 1 điểm của xe J
            min_dist = min(self.distances[a, b] for a in ri for b in rj)
            scores.append((min_dist, i, j))
            
        # Sắp xếp tăng dần theo khoảng cách (min_dist)
        scores.sort(key=lambda x: x[0])
        
        # Trả về TOÀN BỘ các cặp đã được sắp xếp (Không dùng [:n] để cắt bớt nữa)
        return [(i, j) for _, i, j in scores]

    # ------------------------------------------------------------------
    def optimize_single_route_safe(self, route):
        if len(route) <= 1:
            return route, self.compute_route_cost(route)
        if len(route) > self.max_single_size:
            return route, self.compute_route_cost(route)

        eff = self._effective_timeout(self.single_timeout)
        if eff <= 0:
            return route, self.compute_route_cost(route)

        pool = multiprocessing.Pool(processes=1)
        res = pool.apply_async(_solve_single_worker, (route, self.distances))
        try:
            return res.get(timeout=eff)
        except multiprocessing.TimeoutError:
            self.stat_single_timeouts += 1
            logging.warning(f"    [S-Timeout {eff:.1f}s]")
            return route, self.compute_route_cost(route)
        except Exception:
            return route, self.compute_route_cost(route)
        finally:
            pool.terminate()
            pool.join()

    def optimize_route_pair_safe(self, route1, route2):
        if not route1 and not route2:
            return [], [], 0, True
        if len(route1) + len(route2) > self.max_pairwise_size:
            return [], [], float("inf"), False

        eff = self._effective_timeout(self.pairwise_timeout)
        if eff <= 0:
            return [], [], float("inf"), False

        pool = multiprocessing.Pool(processes=1)
        res = pool.apply_async(
            _solve_pairwise_worker,
            (route1, route2, self.distances, self.demands, self.capacity),
        )
        try:
            return res.get(timeout=eff)
        except multiprocessing.TimeoutError:
            self.stat_pairwise_timeouts += 1
            logging.warning(f"    [P-Timeout {eff:.1f}s]")
            return [], [], float("inf"), False
        except Exception:
            return [], [], float("inf"), False
        finally:
            pool.terminate()
            pool.join()

    # ------------------------------------------------------------------
    def try_relocate(self, routes):
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost = self.compute_total_cost(routes)
        improved = False
        for src_id in list(routes.keys()):
            for dst_id in list(routes.keys()):
                if src_id == dst_id:
                    continue
                sr, dr = routes[src_id], routes[dst_id]
                for i, cust in enumerate(sr):
                    if (
                        sum(self.demands[c] for c in dr) + self.demands[cust]
                        > self.capacity
                    ):
                        continue
                    for j in range(len(dr) + 1):
                        ns = sr[:i] + sr[i + 1 :]
                        nd = dr[:j] + [cust] + dr[j:]
                        if self.compute_route_cost(ns) + self.compute_route_cost(
                            nd
                        ) < self.compute_route_cost(sr) + self.compute_route_cost(dr):
                            t = {k: list(v) for k, v in routes.items()}
                            t[src_id], t[dst_id] = ns, nd
                            tc = self.compute_total_cost(t)
                            if tc < best_cost:
                                best_cost, best_routes, improved = tc, t, True
        return best_routes, best_cost, improved

    def try_exchange(self, routes):
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost = self.compute_total_cost(routes)
        improved = False
        for id1, id2 in combinations(list(routes.keys()), 2):
            r1, r2 = routes[id1], routes[id2]
            for i, c1 in enumerate(r1):
                for j, c2 in enumerate(r2):
                    d1 = (
                        sum(self.demands[c] for c in r1)
                        - self.demands[c1]
                        + self.demands[c2]
                    )
                    d2 = (
                        sum(self.demands[c] for c in r2)
                        - self.demands[c2]
                        + self.demands[c1]
                    )
                    if d1 > self.capacity or d2 > self.capacity:
                        continue
                    nr1 = r1[:i] + [c2] + r1[i + 1 :]
                    nr2 = r2[:j] + [c1] + r2[j + 1 :]
                    if self.compute_route_cost(nr1) + self.compute_route_cost(
                        nr2
                    ) < self.compute_route_cost(r1) + self.compute_route_cost(r2):
                        t = {k: list(v) for k, v in routes.items()}
                        t[id1], t[id2] = nr1, nr2
                        tc = self.compute_total_cost(t)
                        if tc < best_cost:
                            best_cost, best_routes, improved = tc, t, True
        return best_routes, best_cost, improved

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def run_phase_2_local_search(self, routes: Dict[int, List[int]], silent: bool = False) -> Tuple[Dict[int, List[int]], float]:
        """ CHẶNG 2: TÌM KIẾM CỤC BỘ ĐA TOÁN TỬ """
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost = self.compute_total_cost(best_routes)

        if not silent:
            logging.info(f"=== GIAI ĐOẠN 2: LOCAL SEARCH ĐA TOÁN TỬ | Bắt đầu: {best_cost:.2f} ===")

        global_improved = True
        iteration = 0

        while global_improved:
            if self._is_timed_out():
                self.stat_global_timeout = True
                break

            iteration += 1
            global_improved = False
            start_iter_cost = best_cost

            # 1. Liên tuyến (Inter-route)
            best_routes, best_cost, imp = self.try_relocate(best_routes)
            global_improved = global_improved or imp

            best_routes, best_cost, imp = self.try_exchange(best_routes)
            global_improved = global_improved or imp

            best_routes, best_cost, imp = CrossExchange.run(best_routes, self.distances, self.demands, self.capacity)
            global_improved = global_improved or imp

            # 2. Nội tuyến (Intra-route)
            best_routes, best_cost, imp = TwoOpt.run(best_routes, self.distances)
            global_improved = global_improved or imp

            best_routes, best_cost, imp = OrOpt.run(best_routes, self.distances)
            global_improved = global_improved or imp

            if global_improved and not silent:
                logging.info(f"  [LS Iter {iteration}] Cải thiện cost: {start_iter_cost:.2f} -> {best_cost:.2f}")

        if not silent:
            logging.info(f"=== KẾT THÚC GIAI ĐOẠN 2 | Đạt cực trị tại: {best_cost:.2f} ===")
            
        return best_routes, best_cost
    # ------------------------------------------------------------------

# ------------------------------------------------------------------
    def run_phase_3_alns(self, routes: Dict[int, List[int]], max_iterations: int, target_cost: float = 0.0) -> Tuple[Dict[int, List[int]], float]:
        """
        CHẶNG 3: ALNS + GLS
        Sử dụng cơ chế phá hủy/sửa chữa thích nghi để nhảy ra khỏi cực trị địa phương.
        """
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost = self.compute_total_cost(best_routes)
        current_routes = {k: list(v) for k, v in routes.items()}

        logging.info(f"=== BẮT ĐẦU CHẶNG 3: ALNS + GLS | Vòng lặp: {max_iterations} | Cost: {best_cost:.2f} ===")

        # 1. Khởi tạo tham số Toán học GLS
        n_nodes = len(self.distances)
        penalties = np.zeros((n_nodes, n_nodes), dtype=int)
        
        valid_edges = self.distances[self.distances > 0]
        avg_dist = np.mean(valid_edges) if len(valid_edges) > 0 else 0
        lambda_val = 0.1 * avg_dist

        # 2. Khởi tạo thông số ALNS
        total_customers = sum(len(r) for r in routes.values())
        # Phá hủy từ 10% đến 25% số khách hàng
        q_min = max(1, int(0.10 * total_customers))
        q_max = max(2, int(0.25 * total_customers))

        operators = [ALNS.destroy_random, ALNS.destroy_worst, ALNS.destroy_related]
        op_names = ["Random Destroy", "Worst Destroy", "Related Destroy"]
        op_weights = [1.0, 1.0, 1.0] # Trọng số cho vòng quay Roulette

        no_improve_count = 0

        for it in range(max_iterations):
            if self._is_timed_out():
                self.stat_global_timeout = True
                break
            
            if target_cost > 0 and best_cost <= target_cost + 1e-5:
                logging.info(f"  [ALNS] 🎉 Đã chạm mốc BKS ({target_cost}). Ngắt vòng lặp ALNS sớm!")
                break

            q = random.randint(q_min, q_max)

            # Chọn toán tử bằng Cò quay (Roulette Wheel)
            total_weight = sum(op_weights)
            probs = [w / total_weight for w in op_weights]
            op_idx = np.random.choice(3, p=probs)
            destroy_op = operators[op_idx]

            # Bước 3.1: Phá hủy
            if destroy_op == ALNS.destroy_random:
                partial_routes, removed = destroy_op(current_routes, q)
            else:
                partial_routes, removed = destroy_op(current_routes, self.distances, q)

            # Bước 3.2: Sửa chữa bằng Regret-2 (GLS Penalties được áp dụng tại đây)
            repaired_routes = ALNS.repair_regret_2(
                partial_routes, removed, self.distances, self.demands,
                self.capacity, penalties, lambda_val
            )

            # BƯỚC VÁ LỖ HỔNG (FIX CRASH): Nếu không thể nhét vừa hàng do kẹt Capacity
            if repaired_routes is None:
                no_improve_count += 1
                # Vẫn giữ nguyên current_routes để vòng sau phá hủy lại từ đầu
                if no_improve_count >= 3:
                    ALNS.update_gls_penalties(current_routes, self.distances, penalties)
                    no_improve_count = 0
                    if it % 20 == 0:
                        logging.info(f"  [GLS Iter {it+1}] Cập nhật ma trận phạt do kẹt tải trọng liên tục.")
                
                # BỎ QUA VÒNG LẶP NÀY, ĐI TIẾP VÒNG ALNS TIẾP THEO
                continue

            # Bước 3.3: Dùng Local Search (Chặng 2) để dọn dẹp và trượt xuống hố cực trị mới
            ls_routes, ls_cost = self.run_phase_2_local_search(repaired_routes, silent=True)

            # Bước 3.4: Đánh giá nghiệm và Thích nghi
            if ls_cost < best_cost - 0.001:
                best_cost = ls_cost
                best_routes = {k: list(v) for k, v in ls_routes.items()}
                current_routes = {k: list(v) for k, v in ls_routes.items()}
                
                # Thưởng lớn (x5) cho toán tử làm tốt
                op_weights[op_idx] += 5.0
                no_improve_count = 0
                logging.info(f"  [ALNS Iter {it+1}] {op_names[op_idx]} phá vỡ giới hạn -> BEST MỚI: {best_cost:.2f}")
            else:
                no_improve_count += 1
                # Thưởng nhỏ khuyến khích sự đa dạng không gian
                op_weights[op_idx] += 0.5 
                
                # Bắt buộc di chuyển sang nghiệm mới để liên tục khám phá bản đồ
                current_routes = {k: list(v) for k, v in ls_routes.items()}

                # Bước 3.5: Cập nhật GLS (Phạt) nếu bị kẹt 3 vòng liên tiếp
                if no_improve_count >= 3:
                    ALNS.update_gls_penalties(current_routes, self.distances, penalties)
                    no_improve_count = 0
                    if it % 20 == 0:
                        logging.info(f"  [GLS Iter {it+1}] Cập nhật ma trận phạt để thay đổi hướng đi.")

        logging.info(f"=== KẾT THÚC CHẶNG 3 | Cost xuất sắc nhất đạt được: {best_cost:.2f} ===")
        return best_routes, best_cost
    
    def run_phase_4_maxsat(
        self, routes: Dict[int, List[int]]
    ) -> Tuple[Dict[int, List[int]], float]:
        """
        CHẶNG 4: TỐI ƯU CHÍNH XÁC MAX-SAT
        1. Tối ưu từng tuyến đơn lẻ.
        2. Tối ưu từng cặp tuyến (Vét cạn cho đến khi không giảm được nữa hoặc hết Timeout).
        """
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost = self.compute_total_cost(best_routes)

        logging.info(
            f"=== BẮT ĐẦU CHẶNG 4: MAX-SAT CHÍNH XÁC | Cost: {best_cost:.2f} ==="
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

            # Logic: Mã hóa Max-SAT O(m^2) biến và O(m^3) mệnh đề
            opt_r, opt_c = self.optimize_single_route_safe(route)

            if opt_c < old_c:
                best_routes[v] = opt_r
                single_imp_total += old_c - opt_c
                self.stat_single_improvements += 1

        if single_imp_total > 0:
            best_cost = self.compute_total_cost(best_routes)
            logging.info(
                f"    -> Single MaxSAT cải thiện: {single_imp_total:.2f}. Cost mới: {best_cost:.2f}"
            )

        if self._is_timed_out():
            return best_routes, best_cost

        # ---------------------------------------------------------
        # BƯỚC 4.2: TỐI ƯU CẶP TUYẾN ĐƯỜNG (PAIRWISE)
        # ---------------------------------------------------------
        logging.info("  [GĐ 4.2] Xét từng cặp tuyến để phân bổ lại khách hàng...")
        import random

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

                # Hàm này đã chứa điều kiện: tổng khách hàng <= ngưỡng max_pairwise_size
                o1, o2, nc, ok = self.optimize_route_pair_safe(r1, r2)

                if ok and nc < old_c:
                    best_routes[i], best_routes[j] = o1, o2
                    pair_improved_in_this_iter = True
                    self.stat_pairwise_improvements += 1

                    best_cost = self.compute_total_cost(best_routes)
                    logging.info(
                        f"    [P4.2] Pair ({i},{j}) giảm {old_c - nc:.0f} -> BEST = {best_cost:.2f}"
                    )
                    break  # Thay đổi cấu trúc -> Tính lại danh sách cặp gần nhất

            if not pair_improved_in_this_iter:
                logging.info(
                    "    -> Đã đạt cực trị toàn cục cho mọi cặp dưới ngưỡng. KẾT THÚC CHẶNG 4."
                )
                break

        return best_routes, best_cost

    # ------------------------------------------------------------------
    def optimize(
        self, initial_routes: Dict, max_iterations: int = 100, target_cost: float = 0.0
    ) -> Tuple[Dict, float]:
        routes = {k: list(v) for k, v in initial_routes.items()}
        best_cost = self.compute_total_cost(routes)

        self._global_start = time.time()

        logging.info("=========================================")
        logging.info("BẮT ĐẦU TUYẾN TRÌNH TỐI ƯU (PIPELINE)")
        logging.info("=========================================")

        # ---------------------------------------------------------
        # CHẶNG 2: LOCAL SEARCH
        # ---------------------------------------------------------
        routes, best_cost = self.run_phase_2_local_search(routes)

        if self._is_timed_out():
            return routes, best_cost

        # ---------------------------------------------------------
        # CHẶNG 3: ALNS (PLACEHOLDER)
        # ---------------------------------------------------------
        routes, best_cost = self.run_phase_3_alns(routes, max_iterations, target_cost)
        
        if target_cost > 0 and best_cost <= target_cost + 1e-5:
            logging.info(f"🎉 ĐÃ ĐẠT NGHIỆM TỐI ƯU TOÀN CỤC (BKS = {target_cost}). DỪNG SỚM!")
            elapsed = time.time() - self._global_start
            return routes, best_cost

        if self._is_timed_out():
            return routes, best_cost
        
        # Hiện tại pass qua thẳng Chặng 4

        # CHẶNG 4: MAX-SAT (Tối ưu Toán học)
        routes, best_cost = self.run_phase_4_maxsat(routes)

        elapsed = time.time() - self._global_start
        logging.info(
            f"HOÀN THÀNH PIPELINE. Final cost={best_cost:.2f} | Time={elapsed:.1f}s"
        )
        return routes, best_cost


GLOBAL_TIMEOUT_DEFAULT = 1200.0


def solve_advanced(
    filepath: str,
    config: Dict[str, Any] = None,
    max_iterations: int = 50,
    target_cost: float = 0.0,
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
    logging.info(f"FILE: {os.path.basename(filepath)}")
    logging.info("=" * 70)

    cvrp = Instance(filepath)
    cvrp.load()
    cvrp.distances = np.floor(cvrp.distances + 0.5).astype(int)
    n_vehicles = int(re.search(r"-k(\d+)", filepath).group(1))
    logging.info(f"Dim={cvrp.dimension} K={n_vehicles} Cap={cvrp.capacity}")

    logging.info("--- Step 1: Clarke-Wright Heuristic ---")
    try:
        cw_time, cw_routes = ClarkeWright.run(cvrp, n_vehicles)
    except Exception:
        logging.warning("CW thất bại → fallback K=999")
        cw_time, cw_routes = ClarkeWright.run(cvrp, 999)
    cw_cost = sum(r.cost for r in cw_routes.values())
    logging.info(f"  cost={cw_cost} time={cw_time:.2f}s")

    # 1. Ép kiểu: Convert NGAY LẬP TỨC từ Object Route sang List[int]
    routes_list = {i: list(r.value) for i, r in cw_routes.items() if r.value}

    # 2. Truyền thẳng List vào Pipeline (Chặng 2 giờ đã tự lo Two-Opt)
    logging.info("--- Step 2, 3, 4: Advanced Optimization Pipeline ---")
    t0 = time.time()
    optimizer = AdvancedCVRPOptimizer(
        distances=cvrp.distances,
        demands=np.array(cvrp.demands),
        capacity=cvrp.capacity,
        n_vehicles=n_vehicles,
        config=config,
    )
    opt_routes, opt_cost = optimizer.optimize(
        routes_list, max_iterations=max_iterations,
        target_cost=target_cost
    )
    pipeline_time = time.time() - t0

    stats = {
        "solver_name": SOLVER_NAME,
        "single_imp_count": optimizer.stat_single_improvements,
        "pairwise_imp_count": optimizer.stat_pairwise_improvements,
        "single_timeouts": optimizer.stat_single_timeouts,
        "pairwise_timeouts": optimizer.stat_pairwise_timeouts,
        "global_timeout": optimizer.stat_global_timeout,
    }

    total_verify = sum(
        optimizer.compute_route_cost(r) for r in opt_routes.values() if r
    )

    logging.info("=" * 70)
    logging.info(f"CW Khởi tạo = {cw_cost} → Sau Pipeline = {int(opt_cost)}")
    logging.info(f"Cải thiện so CW: {cw_cost - int(opt_cost)}")
    logging.info(f"Tổng thời gian: {cw_time + pipeline_time:.1f}s")
    logging.info(f"Verified cost: {total_verify}")

    return opt_routes, total_verify, stats