#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Advanced CVRP Optimizer using MaxSAT + Multiprocessing Timeout
==============================================================

Cấu trúc vòng lặp tối ưu (2 tầng):

  OUTER LOOP (patience_outer lần không cải thiện):
    INNER LOOP (patience_inner lần không cải thiện):
      Phase 1: Relocate
      Phase 2: Exchange
      Phase 3: Single MaxSAT  ← nhanh, chạy nhiều lần đến bão hòa
    Phase 4: Pairwise MaxSAT  ← chậm, chỉ chạy khi inner đã bão hòa

  Lý do tách:
    - Single route (TSP con) giải trong vài giây → nên lặp nhiều lần
    - Pairwise (VRP 2-xe) giải trong hàng trăm giây → chỉ chạy khi
      Relocate/Exchange/Single không còn cải thiện được nữa
    - Tránh tình trạng Single phải "đợi" Pairwise xong mới được chạy lại

Timeout hai tầng:
  - Per-call timeout  : min(config_timeout, remaining - BUFFER)
  - Global timeout    : deadline tuyệt đối, kiểm tra ở đầu mỗi iteration
                        và trước mỗi lần spawn process
"""

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log_filename = f"solver_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

SOLVER_NAME = "MaxSAT-RC2"


# =====================================================================
# SINGLE ROUTE OPTIMIZER
# =====================================================================

class SingleRouteOptimizer:
    """
    Tối ưu 1 route bằng MaxSAT (bài toán TSP con).

    Biến:
      conNet[i][j] = l(i,j): cạnh i→j được dùng
      rchNet[i][j] = r(i,j): i thăm trước j  [chỉ customer-customer]
                              r(j,i) = ¬r(i,j) (anti-symmetric)

    Ràng buộc cứng:
      1. Implication   : l(i,j) → r(i,j)           ∀ i,j ∈ customers
      2. Transitivity  : r(a,b) ∧ r(b,c) → r(a,c)  ∀ distinct a,b,c ∈ customers
      3. Chain law     : r(a,b) ∧ r(b,c) → ¬l(a,c) ∀ distinct a,b,c ∈ customers
      4. Depot out     : exactly-one cạnh 0→j
      5. Depot in      : exactly-one cạnh j→0
      6. Customer deg  : exactly-one in + exactly-one out mỗi customer
      7. Depot-first   : l(0,j) → ¬l(k,j) ∀ k ∈ customers, k≠j
                         (nếu depot đến j trực tiếp, không customer nào khác vào j)
    """

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
        self.wcnf   = WCNF()

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
                self.rchNet[i][j] =  v
                self.rchNet[j][i] = -v

        # Mệnh đề mềm: chi phí cạnh
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.local_dist[i, j] > 0:
                    self.wcnf.append(
                        [-self.conNet[i][j]],
                        weight=int(self.local_dist[i, j])
                    )

        # Ràng buộc 1: l(i,j) → r(i,j)
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i != j and self.rchNet[i][j] != 0:
                    self.wcnf.append([-self.conNet[i][j], self.rchNet[i][j]])

        # Ràng buộc 2: Transitivity
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

        # Ràng buộc 3: Chain law
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
                if k == j: continue
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


# =====================================================================
# PAIRWISE ROUTE OPTIMIZER
# =====================================================================

class PairwiseRouteOptimizer:
    """
    Tối ưu 2 routes bằng MaxSAT (phân hoạch + thứ tự đồng thời).
    Dùng PBEnc để mã hóa ràng buộc tải trọng chính xác thành CNF.
    """

    def __init__(self, c1: List[int], c2: List[int], dist: np.ndarray,
                 dem: np.ndarray, cap: int):
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
        self.con = [[[0]*self.n for _ in range(self.n)] for _ in range(2)]
        self.rch = [[[0]*self.n for _ in range(self.n)] for _ in range(2)]
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
                    self.rch[v][i][j] =  vr
                    self.rch[v][j][i] = -vr
        for i in range(1, self.n):
            self.asn[i] = self._nid()

        # Ràng buộc tải trọng cứng (PBEnc)
        lits_v1, lits_v0, weights = [], [], []
        for i in range(1, self.n):
            lits_v1.append( self.asn[i])
            lits_v0.append(-self.asn[i])
            weights.append(int(self.dem[self.l2g[i]]))

        for lits in (lits_v1, lits_v0):
            cnf = PBEnc.leq(lits=lits, weights=weights,
                            bound=self.cap, top_id=self.var_id)
            for clause in cnf.clauses:
                self.wcnf.append(clause)
            if cnf.clauses:
                self.var_id = max(
                    self.var_id,
                    max(max(abs(l) for l in cl) for cl in cnf.clauses)
                )

        # Mệnh đề mềm
        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i != j and self.ldist[i, j] > 0:
                        self.wcnf.append(
                            [-self.con[v][i][j]],
                            weight=int(self.ldist[i, j])
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
                    if a == b: continue
                    for c in range(1, self.n):
                        if c in (a, b): continue
                        rab, rbc, rac = self.rch[v][a][b], self.rch[v][b][c], self.rch[v][a][c]
                        if rab != 0 and rbc != 0 and rac != 0:
                            self.wcnf.append([-rab, -rbc, rac])
            for a in range(1, self.n):
                for b in range(1, self.n):
                    if a == b: continue
                    for c in range(1, self.n):
                        if c in (a, b): continue
                        rab, rbc, lac = self.rch[v][a][b], self.rch[v][b][c], self.con[v][a][c]
                        if rab != 0 and rbc != 0 and lac != 0:
                            self.wcnf.append([-rab, -rbc, -lac])

        # Depot degree per vehicle (at-most-one; at-least-one không cần vì
        # mỗi xe phải phục vụ ít nhất 1 khách — được đảm bảo bởi flow)
        for v in range(2):
            out_d = [self.con[v][0][j] for j in range(1, self.n)]
            in_d  = [self.con[v][i][0] for i in range(1, self.n)]
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
                in_e  = [self.con[v][j][i] for j in range(self.n) if j != i]
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
                if (sum(self.dem[c] for c in rts[0]) > self.cap or
                        sum(self.dem[c] for c in rts[1]) > self.cap):
                    return [], [], float('inf'), False
                return rts[0], rts[1], solver.cost, True
        return [], [], float('inf'), False


# =====================================================================
# MULTIPROCESSING WRAPPERS
# =====================================================================

def _solve_single_worker(customers, distances):
    return SingleRouteOptimizer(customers, distances).optimize()

def _solve_pairwise_worker(c1, c2, distances, demands, capacity):
    return PairwiseRouteOptimizer(c1, c2, distances, demands, capacity).optimize()


# =====================================================================
# ADVANCED CVRP OPTIMIZER
# =====================================================================

class AdvancedCVRPOptimizer:
    """
    Cấu trúc vòng lặp 2 tầng:

      OUTER LOOP:
        INNER LOOP: Relocate + Exchange + Single MaxSAT
          → lặp đến khi bão hòa (patience_inner lần không cải thiện)
        Pairwise MaxSAT (chạy 1 lần sau khi inner bão hòa)
        → nếu pairwise cải thiện: quay lại inner
        → nếu không: tăng outer_no_imp, dừng nếu đủ patience_outer

    Global timeout:
      effective_timeout = min(config_timeout, remaining - BUFFER)
      Kiểm tra trước mỗi iteration inner và trước mỗi lần spawn.
    """

    _POOL_CLEANUP_BUFFER = 2.0  # giây dành cho pool.terminate()/join()

    def __init__(self, distances: np.ndarray, demands: np.ndarray,
                 capacity: int, n_vehicles: int,
                 config: Dict[str, Any] = None):
        self.distances  = distances
        self.demands    = demands
        self.capacity   = capacity
        self.n_vehicles = n_vehicles

        if config is None:
            config = {}

        self.max_single_size   = config.get("max_single_size",   11)
        self.single_timeout    = config.get("single_timeout",    10.0)
        self.max_pairwise_size = config.get("max_pairwise_size",  9)
        self.pairwise_timeout  = config.get("pairwise_timeout",  20.0)
        self.n_closest_pairs   = config.get("n_closest_pairs",    5)
        self.global_timeout    = config.get("global_timeout",    None)

        # Patience riêng cho inner và outer loop
        # inner_patience: số lần liên tiếp không cải thiện (P1+P2+P3) thì dừng inner
        # outer_patience: số lần liên tiếp pairwise không cải thiện thì dừng toàn bộ
        self.inner_patience = config.get("inner_patience", config.get("patience", 5))
        self.outer_patience = config.get("outer_patience", 3)

        self.stat_single_improvements   = 0
        self.stat_pairwise_improvements = 0
        self.stat_single_timeouts       = 0
        self.stat_pairwise_timeouts     = 0

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

    # ------------------------------------------------------------------
    def compute_route_cost(self, route: List[int]) -> int:
        if not route: return 0
        cost = self.distances[0, route[0]]
        for i in range(len(route) - 1):
            cost += self.distances[route[i], route[i + 1]]
        cost += self.distances[route[-1], 0]
        return cost

    def compute_total_cost(self, routes: Dict[int, List[int]]) -> float:
        return sum(self.compute_route_cost(r) for r in routes.values())

    def find_closest_route_pairs(self, routes):
        ids = list(routes.keys())
        if len(ids) < 2: return []
        scores = []
        for i, j in combinations(ids, 2):
            ri, rj = routes[i], routes[j]
            if not ri or not rj: continue
            d = min(self.distances[a, b] for a in ri for b in rj)
            scores.append((d, i, j))
        scores.sort()
        return [(i, j) for _, i, j in scores[:self.n_closest_pairs]]

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
        res  = pool.apply_async(_solve_single_worker, (route, self.distances))
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
            return [], [], float('inf'), False

        eff = self._effective_timeout(self.pairwise_timeout)
        if eff <= 0:
            return [], [], float('inf'), False

        pool = multiprocessing.Pool(processes=1)
        res  = pool.apply_async(
            _solve_pairwise_worker,
            (route1, route2, self.distances, self.demands, self.capacity)
        )
        try:
            return res.get(timeout=eff)
        except multiprocessing.TimeoutError:
            self.stat_pairwise_timeouts += 1
            logging.warning(f"    [P-Timeout {eff:.1f}s]")
            return [], [], float('inf'), False
        except Exception:
            return [], [], float('inf'), False
        finally:
            pool.terminate()
            pool.join()

    # ------------------------------------------------------------------
    def try_relocate(self, routes):
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost   = self.compute_total_cost(routes)
        improved    = False
        for src_id in list(routes.keys()):
            for dst_id in list(routes.keys()):
                if src_id == dst_id: continue
                sr, dr = routes[src_id], routes[dst_id]
                for i, cust in enumerate(sr):
                    if sum(self.demands[c] for c in dr) + self.demands[cust] > self.capacity:
                        continue
                    for j in range(len(dr) + 1):
                        ns = sr[:i] + sr[i+1:]
                        nd = dr[:j] + [cust] + dr[j:]
                        if (self.compute_route_cost(ns) + self.compute_route_cost(nd) <
                                self.compute_route_cost(sr) + self.compute_route_cost(dr)):
                            t = {k: list(v) for k, v in routes.items()}
                            t[src_id], t[dst_id] = ns, nd
                            tc = self.compute_total_cost(t)
                            if tc < best_cost:
                                best_cost, best_routes, improved = tc, t, True
        return best_routes, best_cost, improved

    def try_exchange(self, routes):
        best_routes = {k: list(v) for k, v in routes.items()}
        best_cost   = self.compute_total_cost(routes)
        improved    = False
        for id1, id2 in combinations(list(routes.keys()), 2):
            r1, r2 = routes[id1], routes[id2]
            for i, c1 in enumerate(r1):
                for j, c2 in enumerate(r2):
                    d1 = sum(self.demands[c] for c in r1) - self.demands[c1] + self.demands[c2]
                    d2 = sum(self.demands[c] for c in r2) - self.demands[c2] + self.demands[c1]
                    if d1 > self.capacity or d2 > self.capacity: continue
                    nr1 = r1[:i] + [c2] + r1[i+1:]
                    nr2 = r2[:j] + [c1] + r2[j+1:]
                    if (self.compute_route_cost(nr1) + self.compute_route_cost(nr2) <
                            self.compute_route_cost(r1) + self.compute_route_cost(r2)):
                        t = {k: list(v) for k, v in routes.items()}
                        t[id1], t[id2] = nr1, nr2
                        tc = self.compute_total_cost(t)
                        if tc < best_cost:
                            best_cost, best_routes, improved = tc, t, True
        return best_routes, best_cost, improved

    # ------------------------------------------------------------------
    def _run_inner_loop(self, routes: Dict, max_iterations: int,
                        outer_idx: int) -> Tuple[Dict, float, bool]:
        """
        Inner loop: Relocate + Exchange + Single MaxSAT.
        Lặp đến khi patience_inner lần liên tiếp không cải thiện
        hoặc hết global timeout.

        Trả về (routes, best_cost, any_improved).
        """
        best_cost   = self.compute_total_cost(routes)
        best_routes = {k: list(v) for k, v in routes.items()}
        any_improved = False
        inner_no_imp = 0
        inner_iter   = 0

        while inner_iter < max_iterations and inner_no_imp < self.inner_patience:
            if self._is_timed_out():
                logging.info("  [Inner] GlobalTimeout → dừng inner.")
                break

            inner_iter += 1
            improved    = False
            rem = self._remaining()
            rem_str = "∞" if rem is None else f"{rem:.0f}s"
            logging.info(
                f"  [Inner {outer_idx}.{inner_iter}] "
                f"cost={best_cost:.2f} no_imp={inner_no_imp}/{self.inner_patience} "
                f"(còn {rem_str})"
            )

            # P1: Relocate
            routes, nc, imp = self.try_relocate(routes)
            if imp:
                improved = True
                logging.info(f"    [P1] Relocate → {nc:.2f}")

            # P2: Exchange
            routes, nc, imp = self.try_exchange(routes)
            if imp:
                improved = True
                logging.info(f"    [P2] Exchange → {nc:.2f}")

            # P3: Single MaxSAT (chạy mọi route trong iteration này)
            single_imp = 0
            for v, route in list(routes.items()):
                if self._is_timed_out():
                    break
                old_c = self.compute_route_cost(route)
                opt_r, opt_c = self.optimize_single_route_safe(route)
                if opt_c < old_c:
                    routes[v] = opt_r
                    improved   = True
                    single_imp += (old_c - opt_c)
                    self.stat_single_improvements += 1
            if single_imp > 0:
                logging.info(f"    [P3] Single +{single_imp:.0f}")

            # Cập nhật best
            cur = self.compute_total_cost(routes)
            if cur < best_cost - 0.001:
                best_cost    = cur
                best_routes  = {k: list(v) for k, v in routes.items()}
                inner_no_imp = 0
                any_improved = True
                logging.info(f"    → INNER BEST = {best_cost:.2f}")
            else:
                inner_no_imp += 1

            if not improved:
                inner_no_imp += 1

        return best_routes, best_cost, any_improved

    # ------------------------------------------------------------------
    def optimize(self, initial_routes: Dict, max_iterations: int = 100) -> Tuple[Dict, float]:
        routes      = {k: list(v) for k, v in initial_routes.items()}
        best_cost   = self.compute_total_cost(routes)
        best_routes = {k: list(v) for k, v in routes.items()}

        self._global_start = time.time()

        logging.info("=========================================")
        logging.info("BẮT ĐẦU OPTIMIZE (MaxSAT-RC2, 2-tầng)")
        logging.info(f"  max_single={self.max_single_size}  s_to={self.single_timeout}s")
        logging.info(f"  max_pair={self.max_pairwise_size}  p_to={self.pairwise_timeout}s")
        logging.info(f"  n_pairs={self.n_closest_pairs}")
        logging.info(f"  inner_patience={self.inner_patience}  outer_patience={self.outer_patience}")
        logging.info(f"  global_timeout={'∞' if self.global_timeout is None else f'{self.global_timeout}s'}")
        logging.info(f"  start_cost={best_cost:.2f}")
        logging.info("=========================================")

        outer_no_imp = 0
        outer_iter   = 0
        # inner_max: mỗi outer iteration dành tối đa bao nhiêu inner iteration
        inner_max = max(1, max_iterations // max(1, self.outer_patience + 1))

        while outer_iter < max_iterations and outer_no_imp < self.outer_patience:
            if self._is_timed_out():
                logging.info(f"[GlobalTimeout] Dừng trước outer iter {outer_iter+1}.")
                break

            outer_iter += 1
            rem = self._remaining()
            rem_str = "∞" if rem is None else f"{rem:.0f}s"
            logging.info(
                f"=== OUTER {outer_iter} | cost={best_cost:.2f} "
                f"outer_no_imp={outer_no_imp}/{self.outer_patience} "
                f"(còn {rem_str}) ==="
            )

            # ── INNER LOOP: Relocate + Exchange + Single ──────────────
            routes, inner_best, inner_improved = self._run_inner_loop(
                routes, inner_max, outer_iter
            )
            if inner_best < best_cost - 0.001:
                best_cost   = inner_best
                best_routes = {k: list(v) for k, v in routes.items()}
                logging.info(f"  [Outer] Inner cải thiện → best={best_cost:.2f}")

            if self._is_timed_out():
                logging.info("[GlobalTimeout] Sau inner loop.")
                break

            # ── PAIRWISE MaxSAT: chạy sau khi inner bão hòa ──────────
            logging.info(f"  [Outer {outer_iter}] Chạy Pairwise MaxSAT...")
            pairs = self.find_closest_route_pairs(routes)
            pair_improved = False

            for i, j in pairs:
                if self._is_timed_out():
                    logging.info("    [P4] GlobalTimeout → dừng pairwise.")
                    break
                r1, r2 = routes[i], routes[j]
                old_c  = self.compute_route_cost(r1) + self.compute_route_cost(r2)
                o1, o2, nc, ok = self.optimize_route_pair_safe(r1, r2)
                if ok and nc < old_c:
                    routes[i], routes[j] = o1, o2
                    pair_improved = True
                    self.stat_pairwise_improvements += 1
                    improvement = old_c - nc
                    logging.info(f"    [P4] Pair ({i},{j}) −{improvement:.0f}")
                    # Cập nhật best nếu tổng chi phí giảm
                    cur = self.compute_total_cost(routes)
                    if cur < best_cost - 0.001:
                        best_cost   = cur
                        best_routes = {k: list(v) for k, v in routes.items()}
                        logging.info(f"    [P4] → OUTER BEST = {best_cost:.2f}")

            # Đánh giá outer iteration
            if inner_improved or pair_improved:
                outer_no_imp = 0
            else:
                outer_no_imp += 1
                logging.info(
                    f"  [Outer] Không cải thiện "
                    f"({outer_no_imp}/{self.outer_patience})"
                )

        elapsed = time.time() - self._global_start
        logging.info(
            f"Hoàn thành. cost={best_cost:.2f} | time={elapsed:.1f}s | "
            f"s_imp={self.stat_single_improvements} s_to={self.stat_single_timeouts} | "
            f"p_imp={self.stat_pairwise_improvements} p_to={self.stat_pairwise_timeouts}"
        )
        return best_routes, best_cost


# =====================================================================
# MAIN SOLVER ROUTINE
# =====================================================================

def solve_advanced(
    filepath: str,
    config: Dict[str, Any] = None,
    max_iterations: int = 50,
) -> Tuple[Dict[int, List[int]], int, Dict[str, Any]]:
    import re
    from classes.instance import Instance
    from classes.clarke_wright import ClarkeWright
    from classes.two_opt import TwoOpt

    logging.info("=" * 70)
    logging.info(f"FILE: {os.path.basename(filepath)}")
    logging.info("=" * 70)

    cvrp = Instance(filepath)
    cvrp.load()
    cvrp.distances = np.floor(cvrp.distances + 0.5).astype(int)
    n_vehicles = int(re.search(r'-k(\d+)', filepath).group(1))
    logging.info(f"Dim={cvrp.dimension} K={n_vehicles} Cap={cvrp.capacity}")

    logging.info("--- Clarke-Wright ---")
    try:
        cw_time, cw_routes = ClarkeWright.run(cvrp, n_vehicles)
    except Exception:
        logging.warning("CW thất bại → fallback K=999")
        cw_time, cw_routes = ClarkeWright.run(cvrp, 999)
    cw_cost = sum(r.cost for r in cw_routes.values())
    logging.info(f"  cost={cw_cost} time={cw_time:.2f}s")

    logging.info("--- Two-Opt ---")
    two_opt_time, two_opt_routes = TwoOpt.run(cw_routes)
    two_opt_cost = sum(r.cost for r in two_opt_routes.values())
    logging.info(f"  cost={two_opt_cost} time={two_opt_time:.2f}s")

    routes = {i: list(r.value)
              for i, (_, r) in enumerate(two_opt_routes.items())}
    routes = {i: r for i, r in routes.items() if r}

    logging.info("--- MaxSAT (2-tầng) ---")
    t0        = time.time()
    optimizer = AdvancedCVRPOptimizer(
        distances=cvrp.distances,
        demands=np.array(cvrp.demands),
        capacity=cvrp.capacity,
        n_vehicles=n_vehicles,
        config=config,
    )
    opt_routes, opt_cost = optimizer.optimize(routes, max_iterations=max_iterations)
    maxsat_time = time.time() - t0

    stats = {
        "solver_name":        SOLVER_NAME,
        "single_imp_count":   optimizer.stat_single_improvements,
        "pairwise_imp_count": optimizer.stat_pairwise_improvements,
        "single_timeouts":    optimizer.stat_single_timeouts,
        "pairwise_timeouts":  optimizer.stat_pairwise_timeouts,
    }

    total_verify = sum(optimizer.compute_route_cost(r)
                       for r in opt_routes.values() if r)

    logging.info("=" * 70)
    logging.info(f"CW={cw_cost} → 2opt={two_opt_cost} → MaxSAT={int(opt_cost)}")
    logging.info(f"Cải thiện so CW: {cw_cost - int(opt_cost)}")
    logging.info(f"Total time: {cw_time+two_opt_time+maxsat_time:.1f}s")
    logging.info(f"Verified cost: {total_verify}")

    return opt_routes, total_verify, stats


if __name__ == "__main__":
    cfg = {
        "max_single_size":   11,
        "single_timeout":    60.0,
        "max_pairwise_size":  9,
        "pairwise_timeout":  600.0,
        "n_closest_pairs":   12,
        "inner_patience":    10,
        "outer_patience":     3,
        "global_timeout":   1200.0,
    }
    fp = sys.argv[1] if len(sys.argv) > 1 else "instances/B/B-n31-k5.vrp"
    solve_advanced(fp, config=cfg, max_iterations=150)