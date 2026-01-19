#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Route Optimizer using MaxSAT
============================
Tối ưu CVRP sử dụng MaxSAT với các chiến lược:
1. Single Route Optimization: Tối ưu thứ tự trong mỗi route
2. Pair-wise Route Optimization: Gộp 2 routes, tối ưu phân hoạch + ordering

Author: Adapted from RTSS + CVRP MaxSAT paper
"""

import sys
import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SingleRouteOptimizer:
    """
    Tối ưu thứ tự các điểm trong một route sử dụng MaxSAT.
    Đây là bài toán TSP nhỏ (chỉ với các điểm trong route).
    """
    
    def __init__(self, customers: List[int], distances: np.ndarray):
        """
        Args:
            customers: List of customer indices (not including depot 0)
            distances: Full distance matrix
        """
        self.customers = customers  # List of customers to visit
        self.n = len(customers) + 1  # Including depot
        self.distances = distances
        
        # Mapping: local index -> global customer index
        self.local_to_global = {0: 0}  # 0 -> depot
        for i, c in enumerate(customers):
            self.local_to_global[i + 1] = c
        
        # Build local distance matrix
        self.local_dist = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                gi, gj = self.local_to_global[i], self.local_to_global[j]
                self.local_dist[i, j] = distances[gi, gj]
        
        # Variables
        self.var_id = 0
        self.conNet = [[0] * self.n for _ in range(self.n)]  # l(i,j)
        self.rchNet = [[0] * self.n for _ in range(self.n)]  # r(i,j)
        
        self.wcnf = WCNF()
    
    def new_var_id(self) -> int:
        self.var_id += 1
        return self.var_id
    
    def gen_variables(self):
        """Generate connection and reachability variables."""
        # Connection variables for all edges except self-loops
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.conNet[i][j] = self.new_var_id()
        
        # Reachability variables with symmetric negation
        # For customers (not depot): r(i,j) = -r(j,i)
        for i in range(1, self.n):
            for j in range(i + 1, self.n):
                var = self.new_var_id()
                self.rchNet[i][j] = var
                self.rchNet[j][i] = -var
        
        # r(0, j) = True for all j (depot is always first) - no variable needed
        # r(j, 0) = False for all j - no variable needed
    
    def gen_soft_clauses(self):
        """Objective: minimize total distance."""
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.local_dist[i, j] > 0:
                    self.wcnf.append([-self.conNet[i][j]], weight=self.local_dist[i, j])
    
    def gen_hard_clauses(self):
        """Generate constraints for TSP (Hamiltonian path from depot)."""
        
        # 1. Implication: l(i,j) -> r(i,j)
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i != j and self.rchNet[i][j] != 0:
                    self.wcnf.append([-self.conNet[i][j], self.rchNet[i][j]])
        
        # 2. Transitivity: r(a,b) ∧ r(b,c) → r(a,c)
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
        
        # 3. Chain law: r(a,b) ∧ r(b,c) → ¬l(a,c)
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
        
        # 4. Depot leaves exactly once
        out_vars = [self.conNet[0][j] for j in range(1, self.n)]
        self.wcnf.append(out_vars)  # At least one
        for i in range(len(out_vars)):
            for j in range(i + 1, len(out_vars)):
                self.wcnf.append([-out_vars[i], -out_vars[j]])  # At most one
        
        # 5. Depot entered exactly once
        in_vars = [self.conNet[i][0] for i in range(1, self.n)]
        self.wcnf.append(in_vars)
        for i in range(len(in_vars)):
            for j in range(i + 1, len(in_vars)):
                self.wcnf.append([-in_vars[i], -in_vars[j]])
        
        # 6. Each customer: exactly one incoming, exactly one outgoing
        for c in range(1, self.n):
            # Incoming
            in_vars = [self.conNet[i][c] for i in range(self.n) if i != c]
            self.wcnf.append(in_vars)
            for i in range(len(in_vars)):
                for j in range(i + 1, len(in_vars)):
                    self.wcnf.append([-in_vars[i], -in_vars[j]])
            
            # Outgoing
            out_vars = [self.conNet[c][j] for j in range(self.n) if j != c]
            self.wcnf.append(out_vars)
            for i in range(len(out_vars)):
                for j in range(i + 1, len(out_vars)):
                    self.wcnf.append([-out_vars[i], -out_vars[j]])
    
    def decode_route(self, model: List[int]) -> List[int]:
        """Decode the route from SAT model."""
        positive = set(v for v in model if v > 0)
        
        route = []
        current = 0  # Start at depot
        visited = {0}
        
        for _ in range(self.n - 1):  # Visit all customers
            found = False
            for j in range(self.n):
                if j not in visited and self.conNet[current][j] in positive:
                    # Convert local to global index
                    global_idx = self.local_to_global[j]
                    if global_idx != 0:  # Don't add depot
                        route.append(global_idx)
                    visited.add(j)
                    current = j
                    found = True
                    break
            if not found:
                break
        
        return route
    
    def optimize(self, timeout: int = 30) -> Tuple[List[int], int]:
        """
        Find optimal ordering of customers in the route.
        
        Returns: (optimized_route, cost)
        """
        self.gen_variables()
        self.gen_soft_clauses()
        self.gen_hard_clauses()
        
        with RC2(self.wcnf, verbose=0) as solver:
            model = solver.compute()
            
            if model:
                route = self.decode_route(model)
                cost = solver.cost
                return route, cost
            else:
                # Return original order if can't optimize
                original_cost = self.local_dist[0, 1]
                for i in range(1, self.n - 1):
                    original_cost += self.local_dist[i, i + 1]
                original_cost += self.local_dist[self.n - 1, 0]
                return self.customers, original_cost


class PairwiseRouteOptimizer:
    """
    Tối ưu 2 routes cùng lúc: gộp customers, tìm phân hoạch + ordering tối ưu.
    
    Ý tưởng:
    - Gộp customers từ 2 routes
    - Dùng MaxSAT để quyết định:
      1. Customer thuộc route nào (assignment)
      2. Thứ tự trong mỗi route (ordering)
    - Đảm bảo capacity constraint
    """
    
    def __init__(self, customers1: List[int], customers2: List[int], 
                 distances: np.ndarray, demands: np.ndarray, capacity: int):
        self.customers1 = customers1
        self.customers2 = customers2
        self.all_customers = customers1 + customers2
        self.n = len(self.all_customers) + 1  # +1 for depot
        self.distances = distances
        self.demands = demands
        self.capacity = capacity
        
        # Mapping
        self.local_to_global = {0: 0}
        for i, c in enumerate(self.all_customers):
            self.local_to_global[i + 1] = c
        self.global_to_local = {v: k for k, v in self.local_to_global.items()}
        
        # Local distance matrix
        self.local_dist = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                gi, gj = self.local_to_global[i], self.local_to_global[j]
                self.local_dist[i, j] = distances[gi, gj]
        
        # Local demands
        self.local_demands = np.zeros(self.n, dtype=int)
        for i in range(1, self.n):
            self.local_demands[i] = demands[self.local_to_global[i]]
        
        # Variables
        self.var_id = 0
        self.wcnf = WCNF()
        
        # conNet[v][i][j]: edge i->j on vehicle v
        self.conNet = [[[0] * self.n for _ in range(self.n)] for _ in range(2)]
        # rchNet[v][i][j]: i before j on vehicle v  
        self.rchNet = [[[0] * self.n for _ in range(self.n)] for _ in range(2)]
        # assign[i]: customer i assigned to vehicle 1 (True) or vehicle 0 (False)
        self.assign = [0] * self.n
    
    def new_var_id(self) -> int:
        self.var_id += 1
        return self.var_id
    
    def gen_variables(self):
        """Generate all variables."""
        # Assignment variables: which vehicle for each customer
        for i in range(1, self.n):
            self.assign[i] = self.new_var_id()
        
        # Connection variables for each vehicle
        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        self.conNet[v][i][j] = self.new_var_id()
        
        # Reachability with symmetric negation
        for v in range(2):
            for i in range(1, self.n):
                for j in range(i + 1, self.n):
                    var = self.new_var_id()
                    self.rchNet[v][i][j] = var
                    self.rchNet[v][j][i] = -var
    
    def gen_soft_clauses(self):
        """Objective: minimize total distance."""
        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i != j and self.local_dist[i, j] > 0:
                        self.wcnf.append([-self.conNet[v][i][j]], 
                                        weight=self.local_dist[i, j])
    
    def gen_hard_clauses(self):
        """Generate all constraints."""
        
        # 1. Assignment consistency: if customer on vehicle v, edges only on v
        for i in range(1, self.n):
            for j in range(self.n):
                if i != j:
                    # assign[i] = True (vehicle 1) -> no edges on vehicle 0
                    self.wcnf.append([-self.assign[i], -self.conNet[0][i][j]])
                    self.wcnf.append([-self.assign[i], -self.conNet[0][j][i]])
                    # assign[i] = False (vehicle 0) -> no edges on vehicle 1
                    self.wcnf.append([self.assign[i], -self.conNet[1][i][j]])
                    self.wcnf.append([self.assign[i], -self.conNet[1][j][i]])
        
        # 2. Each customer visited exactly once (across both vehicles)
        for c in range(1, self.n):
            in_vars = []
            out_vars = []
            for v in range(2):
                for i in range(self.n):
                    if i != c:
                        in_vars.append(self.conNet[v][i][c])
                        out_vars.append(self.conNet[v][c][i])
            
            # Exactly one incoming
            self.wcnf.append(in_vars)
            for i in range(len(in_vars)):
                for j in range(i + 1, len(in_vars)):
                    self.wcnf.append([-in_vars[i], -in_vars[j]])
            
            # Exactly one outgoing
            self.wcnf.append(out_vars)
            for i in range(len(out_vars)):
                for j in range(i + 1, len(out_vars)):
                    self.wcnf.append([-out_vars[i], -out_vars[j]])
        
        # 3. Each vehicle leaves and returns to depot exactly once
        for v in range(2):
            out_vars = [self.conNet[v][0][j] for j in range(1, self.n)]
            in_vars = [self.conNet[v][i][0] for i in range(1, self.n)]
            
            self.wcnf.append(out_vars)
            for i in range(len(out_vars)):
                for j in range(i + 1, len(out_vars)):
                    self.wcnf.append([-out_vars[i], -out_vars[j]])
            
            self.wcnf.append(in_vars)
            for i in range(len(in_vars)):
                for j in range(i + 1, len(in_vars)):
                    self.wcnf.append([-in_vars[i], -in_vars[j]])
        
        # 4. Implication: l(i,j) -> r(i,j)
        for v in range(2):
            for i in range(1, self.n):
                for j in range(1, self.n):
                    if i != j and self.rchNet[v][i][j] != 0:
                        self.wcnf.append([-self.conNet[v][i][j], self.rchNet[v][i][j]])
        
        # 5. Transitivity (simplified - only for same vehicle)
        for v in range(2):
            for a in range(1, self.n):
                for b in range(1, self.n):
                    if a == b:
                        continue
                    for c in range(1, self.n):
                        if c == a or c == b:
                            continue
                        r_ab = self.rchNet[v][a][b]
                        r_bc = self.rchNet[v][b][c]
                        r_ac = self.rchNet[v][a][c]
                        if r_ab != 0 and r_bc != 0 and r_ac != 0:
                            self.wcnf.append([-r_ab, -r_bc, r_ac])
        
        # 6. Chain law
        for v in range(2):
            for a in range(1, self.n):
                for b in range(1, self.n):
                    if a == b:
                        continue
                    for c in range(1, self.n):
                        if c == a or c == b:
                            continue
                        r_ab = self.rchNet[v][a][b]
                        r_bc = self.rchNet[v][b][c]
                        if r_ab != 0 and r_bc != 0:
                            self.wcnf.append([-r_ab, -r_bc, -self.conNet[v][a][c]])
    
    def decode_routes(self, model: List[int]) -> Tuple[List[int], List[int]]:
        """Decode two routes from model."""
        positive = set(v for v in model if v > 0)
        
        routes = [[], []]
        for v in range(2):
            current = 0
            visited = {0}
            for _ in range(self.n - 1):
                found = False
                for j in range(self.n):
                    if j not in visited and self.conNet[v][current][j] in positive:
                        global_idx = self.local_to_global[j]
                        if global_idx != 0:
                            routes[v].append(global_idx)
                        visited.add(j)
                        current = j
                        found = True
                        break
                if not found:
                    break
        
        return routes[0], routes[1]
    
    def check_capacity(self, route: List[int]) -> bool:
        """Check if route respects capacity."""
        return sum(self.demands[c] for c in route) <= self.capacity
    
    def optimize(self, timeout: int = 10) -> Tuple[List[int], List[int], int]:
        """
        Find optimal assignment and ordering for 2 routes.
        
        Returns: (route1, route2, total_cost)
        """
        self.gen_variables()
        self.gen_soft_clauses()
        self.gen_hard_clauses()
        
        with RC2(self.wcnf, verbose=0) as solver:
            model = solver.compute()
            
            if model:
                route1, route2 = self.decode_routes(model)
                
                # Verify capacity
                if self.check_capacity(route1) and self.check_capacity(route2):
                    return route1, route2, solver.cost
        
        # Fallback: return original
        return self.customers1, self.customers2, float('inf')


def calculate_route_cost(route: List[int], distances: np.ndarray) -> int:
    """Calculate cost of a route."""
    if not route:
        return 0
    cost = distances[0, route[0]]
    for i in range(len(route) - 1):
        cost += distances[route[i], route[i + 1]]
    cost += distances[route[-1], 0]
    return int(cost)


def relocate_search(routes: Dict[int, List[int]], distances: np.ndarray,
                    demands: np.ndarray, capacity: int) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    Relocate: Move one customer from one route to another.
    Returns the best improvement found.
    """
    best_saving = 0
    best_move = None
    
    route_ids = list(routes.keys())
    
    for r1 in route_ids:
        for pos1, customer in enumerate(routes[r1]):
            # Calculate removal cost from r1
            route1 = routes[r1]
            if len(route1) == 1:
                remove_cost = distances[0, customer] + distances[customer, 0]
                new_cost_r1 = 0
            else:
                if pos1 == 0:
                    prev1 = 0
                    next1 = route1[1]
                elif pos1 == len(route1) - 1:
                    prev1 = route1[pos1 - 1]
                    next1 = 0
                else:
                    prev1 = route1[pos1 - 1]
                    next1 = route1[pos1 + 1]
                
                old_links = distances[prev1, customer] + distances[customer, next1]
                new_link = distances[prev1, next1]
                remove_cost = old_links - new_link
                
                new_route1 = route1[:pos1] + route1[pos1+1:]
                new_cost_r1 = calculate_route_cost(new_route1, distances)
            
            # Try inserting into other routes
            for r2 in route_ids:
                if r1 == r2:
                    continue
                
                route2 = routes[r2]
                demand_r2 = sum(demands[c] for c in route2)
                
                if demand_r2 + demands[customer] > capacity:
                    continue
                
                # Find best insertion position
                for pos2 in range(len(route2) + 1):
                    if pos2 == 0:
                        prev2 = 0
                        next2 = route2[0] if route2 else 0
                    elif pos2 == len(route2):
                        prev2 = route2[-1]
                        next2 = 0
                    else:
                        prev2 = route2[pos2 - 1]
                        next2 = route2[pos2]
                    
                    insert_cost = distances[prev2, customer] + distances[customer, next2] - distances[prev2, next2]
                    
                    saving = remove_cost - insert_cost
                    
                    if saving > best_saving:
                        best_saving = saving
                        best_move = (r1, pos1, r2, pos2, customer)
    
    if best_move:
        r1, pos1, r2, pos2, customer = best_move
        # Apply move
        new_routes = {k: list(v) for k, v in routes.items()}
        new_routes[r1] = routes[r1][:pos1] + routes[r1][pos1+1:]
        new_routes[r2] = routes[r2][:pos2] + [customer] + routes[r2][pos2:]
        
        return new_routes, best_saving, True
    
    return routes, 0, False


def exchange_search(routes: Dict[int, List[int]], distances: np.ndarray,
                    demands: np.ndarray, capacity: int) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    Exchange: Swap one customer from route1 with one customer from route2.
    Returns the best improvement found.
    """
    best_saving = 0
    best_move = None
    
    route_ids = list(routes.keys())
    
    for r1, r2 in combinations(route_ids, 2):
        for pos1, c1 in enumerate(routes[r1]):
            for pos2, c2 in enumerate(routes[r2]):
                # Check capacity
                demand_r1 = sum(demands[c] for c in routes[r1]) - demands[c1] + demands[c2]
                demand_r2 = sum(demands[c] for c in routes[r2]) - demands[c2] + demands[c1]
                
                if demand_r1 > capacity or demand_r2 > capacity:
                    continue
                
                # Calculate cost change for r1
                route1 = routes[r1]
                prev1 = route1[pos1 - 1] if pos1 > 0 else 0
                next1 = route1[pos1 + 1] if pos1 < len(route1) - 1 else 0
                old_cost1 = distances[prev1, c1] + distances[c1, next1]
                new_cost1 = distances[prev1, c2] + distances[c2, next1]
                
                # Calculate cost change for r2
                route2 = routes[r2]
                prev2 = route2[pos2 - 1] if pos2 > 0 else 0
                next2 = route2[pos2 + 1] if pos2 < len(route2) - 1 else 0
                old_cost2 = distances[prev2, c2] + distances[c2, next2]
                new_cost2 = distances[prev2, c1] + distances[c1, next2]
                
                saving = (old_cost1 + old_cost2) - (new_cost1 + new_cost2)
                
                if saving > best_saving:
                    best_saving = saving
                    best_move = (r1, pos1, c1, r2, pos2, c2)
    
    if best_move:
        r1, pos1, c1, r2, pos2, c2 = best_move
        # Apply move
        new_routes = {k: list(v) for k, v in routes.items()}
        new_routes[r1][pos1] = c2
        new_routes[r2][pos2] = c1
        
        return new_routes, best_saving, True
    
    return routes, 0, False


def inter_route_local_search(routes: Dict[int, List[int]], distances: np.ndarray,
                             demands: np.ndarray, capacity: int,
                             max_iterations: int = 100, verbose: bool = True) -> Tuple[Dict[int, List[int]], int]:
    """
    Apply Relocate and Exchange moves until no improvement.
    """
    if verbose:
        print("\n=== Inter-route Local Search (Relocate + Exchange) ===")
    
    improved = True
    iteration = 0
    total_saving = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try relocate
        routes, saving, did_improve = relocate_search(routes, distances, demands, capacity)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  Relocate: -{saving}")
            continue
        
        # Try exchange
        routes, saving, did_improve = exchange_search(routes, distances, demands, capacity)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  Exchange: -{saving}")
            continue
        
        # Try Or-Opt (move pairs)
        routes, saving, did_improve = or_opt_search(routes, distances, demands, capacity, seq_len=2)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  Or-Opt(2): -{saving}")
            continue
        
        # Try Or-Opt (move triples)
        routes, saving, did_improve = or_opt_search(routes, distances, demands, capacity, seq_len=3)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  Or-Opt(3): -{saving}")
            continue
        
        # Try Cross-Exchange
        routes, saving, did_improve = cross_exchange(routes, distances, demands, capacity)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  Cross-Exchange: -{saving}")
            continue
        
        # Try intra-route 2-opt
        routes, saving, did_improve = intra_route_2opt(routes, distances)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  2-opt: -{saving}")
            continue
        
        # Try intra-route 3-opt (stronger)
        routes, saving, did_improve = intra_route_3opt(routes, distances)
        if did_improve:
            total_saving += saving
            improved = True
            if verbose:
                print(f"  3-opt: -{saving}")
    
    total_cost = sum(calculate_route_cost(r, distances) for r in routes.values())
    if verbose:
        print(f"  Total improvement: -{total_saving}")
    
    return routes, total_cost


# ============================================================================
# GUIDED LOCAL SEARCH (GLS)
# ============================================================================

def calculate_augmented_cost(route: List[int], distances: np.ndarray, 
                             penalties: np.ndarray, lambda_param: float) -> float:
    """Calculate augmented cost = actual cost + lambda * penalties."""
    if not route:
        return 0
    
    cost = distances[0, route[0]] + lambda_param * penalties[0, route[0]]
    for i in range(len(route) - 1):
        cost += distances[route[i], route[i + 1]] + lambda_param * penalties[route[i], route[i + 1]]
    cost += distances[route[-1], 0] + lambda_param * penalties[route[-1], 0]
    return cost


def get_route_edges(routes: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    """Get all edges in the current solution."""
    edges = []
    for route in routes.values():
        if not route:
            continue
        edges.append((0, route[0]))
        for i in range(len(route) - 1):
            edges.append((route[i], route[i + 1]))
        edges.append((route[-1], 0))
    return edges


def gls_local_search(routes: Dict[int, List[int]], distances: np.ndarray,
                     demands: np.ndarray, capacity: int,
                     penalties: np.ndarray, lambda_param: float,
                     max_iterations: int = 100) -> Tuple[Dict[int, List[int]], float]:
    """
    Local search using augmented cost function (with penalties).
    Includes relocate, exchange, and 2-opt with augmented cost.
    """
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try relocate with augmented cost
        best_saving = 0
        best_move = None
        route_ids = list(routes.keys())
        
        for r1 in route_ids:
            for pos1, customer in enumerate(routes[r1]):
                route1 = routes[r1]
                if len(route1) == 1:
                    prev1, next1 = 0, 0
                else:
                    prev1 = route1[pos1 - 1] if pos1 > 0 else 0
                    next1 = route1[pos1 + 1] if pos1 < len(route1) - 1 else 0
                
                # Augmented cost of removing
                old_aug = (distances[prev1, customer] + lambda_param * penalties[prev1, customer] +
                          distances[customer, next1] + lambda_param * penalties[customer, next1])
                new_aug = distances[prev1, next1] + lambda_param * penalties[prev1, next1]
                remove_saving = old_aug - new_aug
                
                for r2 in route_ids:
                    if r1 == r2:
                        continue
                    
                    route2 = routes[r2]
                    demand_r2 = sum(demands[c] for c in route2)
                    if demand_r2 + demands[customer] > capacity:
                        continue
                    
                    for pos2 in range(len(route2) + 1):
                        prev2 = route2[pos2 - 1] if pos2 > 0 else 0
                        next2 = route2[pos2] if pos2 < len(route2) else 0
                        
                        # Augmented cost of inserting
                        old_aug2 = distances[prev2, next2] + lambda_param * penalties[prev2, next2]
                        new_aug2 = (distances[prev2, customer] + lambda_param * penalties[prev2, customer] +
                                   distances[customer, next2] + lambda_param * penalties[customer, next2])
                        insert_cost = new_aug2 - old_aug2
                        
                        saving = remove_saving - insert_cost
                        if saving > best_saving + 1e-6:
                            best_saving = saving
                            best_move = ('relocate', r1, pos1, r2, pos2, customer)
        
        # Try exchange with augmented cost
        for r1, r2 in combinations(route_ids, 2):
            for pos1, c1 in enumerate(routes[r1]):
                for pos2, c2 in enumerate(routes[r2]):
                    demand_r1 = sum(demands[c] for c in routes[r1]) - demands[c1] + demands[c2]
                    demand_r2 = sum(demands[c] for c in routes[r2]) - demands[c2] + demands[c1]
                    
                    if demand_r1 > capacity or demand_r2 > capacity:
                        continue
                    
                    route1, route2 = routes[r1], routes[r2]
                    prev1 = route1[pos1 - 1] if pos1 > 0 else 0
                    next1 = route1[pos1 + 1] if pos1 < len(route1) - 1 else 0
                    prev2 = route2[pos2 - 1] if pos2 > 0 else 0
                    next2 = route2[pos2 + 1] if pos2 < len(route2) - 1 else 0
                    
                    old_aug = (distances[prev1, c1] + lambda_param * penalties[prev1, c1] +
                              distances[c1, next1] + lambda_param * penalties[c1, next1] +
                              distances[prev2, c2] + lambda_param * penalties[prev2, c2] +
                              distances[c2, next2] + lambda_param * penalties[c2, next2])
                    new_aug = (distances[prev1, c2] + lambda_param * penalties[prev1, c2] +
                              distances[c2, next1] + lambda_param * penalties[c2, next1] +
                              distances[prev2, c1] + lambda_param * penalties[prev2, c1] +
                              distances[c1, next2] + lambda_param * penalties[c1, next2])
                    
                    saving = old_aug - new_aug
                    if saving > best_saving + 1e-6:
                        best_saving = saving
                        best_move = ('exchange', r1, pos1, c1, r2, pos2, c2)
        
        # Try 2-opt within routes with augmented cost
        for r_id, route in routes.items():
            if len(route) < 3:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    prev_i = route[i - 1] if i > 0 else 0
                    next_j = route[j + 1] if j < len(route) - 1 else 0
                    
                    old_aug = (distances[prev_i, route[i]] + lambda_param * penalties[prev_i, route[i]] +
                              distances[route[j], next_j] + lambda_param * penalties[route[j], next_j])
                    new_aug = (distances[prev_i, route[j]] + lambda_param * penalties[prev_i, route[j]] +
                              distances[route[i], next_j] + lambda_param * penalties[route[i], next_j])
                    
                    saving = old_aug - new_aug
                    if saving > best_saving + 1e-6:
                        best_saving = saving
                        best_move = ('2opt', r_id, i, j)
        
        # Apply best move
        if best_move:
            improved = True
            if best_move[0] == 'relocate':
                _, r1, pos1, r2, pos2, customer = best_move
                new_routes = {k: list(v) for k, v in routes.items()}
                new_routes[r1] = routes[r1][:pos1] + routes[r1][pos1+1:]
                new_routes[r2] = routes[r2][:pos2] + [customer] + routes[r2][pos2:]
                routes = new_routes
            elif best_move[0] == 'exchange':
                _, r1, pos1, c1, r2, pos2, c2 = best_move
                new_routes = {k: list(v) for k, v in routes.items()}
                new_routes[r1][pos1] = c2
                new_routes[r2][pos2] = c1
                routes = new_routes
            else:  # 2opt
                _, r_id, i, j = best_move
                new_routes = {k: list(v) for k, v in routes.items()}
                new_routes[r_id] = routes[r_id][:i] + routes[r_id][i:j+1][::-1] + routes[r_id][j+1:]
                routes = new_routes
    
    # Return actual cost (not augmented)
    total_cost = sum(calculate_route_cost(r, distances) for r in routes.values())
    return routes, total_cost


def guided_local_search(routes: Dict[int, List[int]], distances: np.ndarray,
                        demands: np.ndarray, capacity: int,
                        max_iterations: int = 100,
                        time_limit: float = 60.0) -> Tuple[Dict[int, List[int]], int]:
    """
    Guided Local Search: penalize frequently used "bad" edges to escape local optima.
    
    Key idea:
    - augmented_cost(edge) = distance(edge) + lambda * penalty(edge)
    - After each local optimum, penalize edges with highest utility
    - utility(edge) = distance(edge) / (1 + penalty(edge))
    """
    print("\n=== Guided Local Search ===")
    
    n = distances.shape[0]
    penalties = np.zeros((n, n), dtype=float)
    
    # Lambda parameter (typically ~0.1 * average edge cost)
    avg_dist = np.mean(distances[distances > 0])
    lambda_param = 0.1 * avg_dist
    
    # Initial local search (without penalties)
    best_routes, best_cost = inter_route_local_search(
        routes, distances, demands, capacity, verbose=False
    )
    current_routes = best_routes
    print(f"  Initial: {best_cost}")
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            print(f"  Time limit reached at iteration {iteration}")
            break
        
        # Get edges in current solution
        edges = get_route_edges(current_routes)
        
        # Find edge(s) with maximum utility to penalize
        max_utility = -1
        edges_to_penalize = []
        
        for i, j in edges:
            utility = distances[i, j] / (1 + penalties[i, j])
            if utility > max_utility + 1e-6:
                max_utility = utility
                edges_to_penalize = [(i, j)]
            elif abs(utility - max_utility) < 1e-6:
                edges_to_penalize.append((i, j))
        
        # Penalize edges with max utility
        for i, j in edges_to_penalize:
            penalties[i, j] += 1
            penalties[j, i] += 1  # Symmetric
        
        # Run local search with augmented cost
        new_routes, new_cost = gls_local_search(
            current_routes, distances, demands, capacity,
            penalties, lambda_param
        )
        
        current_routes = new_routes
        
        if new_cost < best_cost:
            print(f"  Iteration {iteration + 1}: {best_cost} -> {new_cost} (IMPROVED -{best_cost - new_cost})")
            best_routes = new_routes
            best_cost = new_cost
    
    print(f"  Final cost: {best_cost}")
    return best_routes, best_cost


import random

def perturbation(routes: Dict[int, List[int]], distances: np.ndarray,
                 demands: np.ndarray, capacity: int, 
                 strength: int = 5) -> Dict[int, List[int]]:
    """
    Perturb solution by performing random moves (accept even if worse).
    """
    routes = {k: list(v) for k, v in routes.items()}
    route_ids = [k for k in routes if routes[k]]
    
    moves_done = 0
    attempts = 0
    max_attempts = strength * 10
    
    while moves_done < strength and attempts < max_attempts:
        attempts += 1
        
        if len(route_ids) < 2:
            break
            
        # Pick random customer from random route
        r1 = random.choice(route_ids)
        if not routes[r1]:
            continue
        
        pos1 = random.randint(0, len(routes[r1]) - 1)
        customer = routes[r1][pos1]
        
        # Try to move to random position in another route
        other_routes = [r for r in route_ids if r != r1 and routes[r]]
        if not other_routes:
            continue
        r2 = random.choice(other_routes)
            
        demand_r2 = sum(demands[c] for c in routes[r2])
        if demand_r2 + demands[customer] <= capacity:
            routes[r1] = routes[r1][:pos1] + routes[r1][pos1+1:]
            pos2 = random.randint(0, len(routes[r2]))
            routes[r2] = routes[r2][:pos2] + [customer] + routes[r2][pos2:]
            moves_done += 1
            
            # Update route_ids
            route_ids = [k for k in routes if routes[k]]
    
    return routes


def double_bridge_perturbation(routes: Dict[int, List[int]], distances: np.ndarray,
                               demands: np.ndarray, capacity: int) -> Dict[int, List[int]]:
    """
    Double Bridge: Exchange segments between two routes.
    This is a stronger perturbation that can escape deep local optima.
    """
    routes = {k: list(v) for k, v in routes.items()}
    route_ids = [k for k in routes if len(routes[k]) >= 2]
    
    if len(route_ids) < 2:
        return routes
    
    # Pick two random routes
    r1, r2 = random.sample(route_ids, 2)
    route1, route2 = routes[r1], routes[r2]
    
    if len(route1) < 2 or len(route2) < 2:
        return routes
    
    # Pick random cut points
    cut1 = random.randint(1, len(route1) - 1)
    cut2 = random.randint(1, len(route2) - 1)
    
    # Exchange segments
    new_route1 = route1[:cut1] + route2[cut2:]
    new_route2 = route2[:cut2] + route1[cut1:]
    
    # Check capacity
    demand1 = sum(demands[c] for c in new_route1)
    demand2 = sum(demands[c] for c in new_route2)
    
    if demand1 <= capacity and demand2 <= capacity:
        routes[r1] = new_route1
        routes[r2] = new_route2
    
    return routes


def lns_perturbation(routes: Dict[int, List[int]], distances: np.ndarray,
                     demands: np.ndarray, capacity: int,
                     destroy_pct: float = 0.3,
                     destroy_method: str = 'random') -> Dict[int, List[int]]:
    """
    LNS-style perturbation with multiple destroy methods:
    - 'random': Random removal
    - 'worst': Remove customers with highest insertion cost
    - 'shaw': Remove similar customers (close to each other)
    
    Repair using Regret-2 heuristic.
    """
    routes = {k: list(v) for k, v in routes.items()}
    
    # Collect all customers with their positions
    all_customers = []
    customer_info = {}  # customer -> (route_id, position)
    for r_id, route in routes.items():
        for pos, c in enumerate(route):
            all_customers.append(c)
            customer_info[c] = (r_id, pos)
    
    if len(all_customers) < 3:
        return routes
    
    n_remove = max(2, int(len(all_customers) * destroy_pct))
    
    # Choose destroy method
    if destroy_method == 'worst':
        # Calculate removal cost for each customer
        removal_costs = []
        for c in all_customers:
            r_id, pos = customer_info[c]
            route = routes[r_id]
            prev_node = route[pos - 1] if pos > 0 else 0
            next_node = route[pos + 1] if pos < len(route) - 1 else 0
            
            # Cost saved by removing this customer
            cost_saved = (distances[prev_node, c] + distances[c, next_node] - 
                         distances[prev_node, next_node])
            removal_costs.append((cost_saved, c))
        
        # Sort by cost (descending) - remove worst customers first
        removal_costs.sort(reverse=True)
        removed = [c for _, c in removal_costs[:n_remove]]
        
    elif destroy_method == 'shaw':
        # Shaw removal: remove customers similar to a seed customer
        seed = random.choice(all_customers)
        
        # Calculate "relatedness" to seed (based on distance)
        relatedness = []
        for c in all_customers:
            if c != seed:
                rel = distances[seed, c]
                relatedness.append((rel, c))
        
        # Sort by relatedness (ascending) - most similar first
        relatedness.sort()
        removed = [seed] + [c for _, c in relatedness[:n_remove - 1]]
        
    else:  # random
        removed = random.sample(all_customers, n_remove)
    
    # Remove customers from routes
    for r_id in routes:
        routes[r_id] = [c for c in routes[r_id] if c not in removed]
    
    # Reinsert using Regret-2 heuristic
    while removed:
        best_regret = -float('inf')
        best_customer = None
        best_route = None
        best_pos = None
        
        for customer in removed:
            # Find best and second-best insertion for this customer
            insertions = []
            
            for r_id, route in routes.items():
                route_demand = sum(demands[c] for c in route)
                if route_demand + demands[customer] > capacity:
                    continue
                
                for pos in range(len(route) + 1):
                    prev_node = route[pos - 1] if pos > 0 else 0
                    next_node = route[pos] if pos < len(route) else 0
                    
                    cost_increase = (distances[prev_node, customer] + 
                                    distances[customer, next_node] -
                                    distances[prev_node, next_node])
                    
                    insertions.append((cost_increase, r_id, pos))
            
            if not insertions:
                continue
            
            insertions.sort(key=lambda x: x[0])
            best_cost = insertions[0][0]
            second_cost = insertions[1][0] if len(insertions) > 1 else best_cost + 100
            
            regret = second_cost - best_cost
            
            if regret > best_regret:
                best_regret = regret
                best_customer = customer
                best_route = insertions[0][1]
                best_pos = insertions[0][2]
        
        if best_customer is not None:
            routes[best_route].insert(best_pos, best_customer)
            removed.remove(best_customer)
        else:
            # No feasible insertion, create new route
            customer = removed.pop(0)
            new_id = max(routes.keys()) + 1
            routes[new_id] = [customer]
    
    return routes


def iterated_local_search(routes: Dict[int, List[int]], distances: np.ndarray,
                          demands: np.ndarray, capacity: int,
                          max_no_improve: int = 20, 
                          time_limit: float = 60.0,
                          use_sa: bool = False) -> Tuple[Dict[int, List[int]], int]:
    """
    Iterated Local Search with perturbation and optional Simulated Annealing acceptance.
    """
    print("\n=== Iterated Local Search ===")
    
    import time
    import math
    start_time = time.time()
    
    # Apply local search to get initial local optimum
    best_routes, best_cost = inter_route_local_search(
        routes, distances, demands, capacity, verbose=False
    )
    current_routes, current_cost = best_routes, best_cost
    print(f"  Initial local optimum: {best_cost}")
    
    no_improve = 0
    iteration = 0
    
    # SA parameters
    if use_sa:
        temp = best_cost * 0.05  # Initial temperature
        cooling_rate = 0.95
    
    while no_improve < max_no_improve:
        if time.time() - start_time > time_limit:
            print(f"  Time limit reached")
            break
            
        iteration += 1
        
        # ALNS: Alternate between different perturbation types for diversification
        perturb_choice = iteration % 8
        destroy_pct = min(0.5, 0.2 + 0.05 * (no_improve // 3))
        
        if perturb_choice == 0:
            perturbed = double_bridge_perturbation(current_routes, distances, demands, capacity)
        elif perturb_choice == 1:
            # LNS with random removal
            perturbed = lns_perturbation(current_routes, distances, demands, capacity, 
                                         destroy_pct, destroy_method='random')
        elif perturb_choice == 2:
            # LNS with worst removal
            perturbed = lns_perturbation(current_routes, distances, demands, capacity, 
                                         destroy_pct, destroy_method='worst')
        elif perturb_choice == 3:
            # LNS with Shaw removal
            perturbed = lns_perturbation(current_routes, distances, demands, capacity, 
                                         destroy_pct, destroy_method='shaw')
        else:
            strength = 3 + (no_improve // 5)
            perturbed = perturbation(current_routes, distances, demands, capacity, strength=strength)
        
        # Apply local search
        new_routes, new_cost = inter_route_local_search(
            perturbed, distances, demands, capacity, verbose=False
        )
        
        # Accept decision
        accept = False
        if new_cost < current_cost:
            accept = True
        elif use_sa and temp > 0.1:
            # Simulated Annealing acceptance
            delta = new_cost - current_cost
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                accept = True
        
        if accept:
            current_routes = new_routes
            current_cost = new_cost
            
            if new_cost < best_cost:
                print(f"  Iteration {iteration}: {best_cost} -> {new_cost} (IMPROVED -{best_cost - new_cost})")
                best_routes = new_routes
                best_cost = new_cost
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Cool down
        if use_sa:
            temp *= cooling_rate
    
    print(f"  Final cost: {best_cost} (after {iteration} iterations)")
    return best_routes, best_cost


def multi_start_ils(routes: Dict[int, List[int]], distances: np.ndarray,
                    demands: np.ndarray, capacity: int,
                    n_restarts: int = 5, time_limit: float = 60.0) -> Tuple[Dict[int, List[int]], int]:
    """
    Multi-start ILS: Run ILS multiple times with different random seeds.
    """
    print("\n=== Multi-Start ILS ===")
    
    import time
    start_time = time.time()
    
    best_routes = None
    best_cost = float('inf')
    
    time_per_restart = time_limit / n_restarts
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(restart * 42 + 7)
        
        # Strong perturbation for restart
        if restart > 0:
            perturbed = perturbation(routes, distances, demands, capacity, strength=10)
        else:
            perturbed = {k: list(v) for k, v in routes.items()}
        
        # Run ILS with more iterations
        ils_routes, ils_cost = iterated_local_search(
            perturbed, distances, demands, capacity,
            max_no_improve=50, time_limit=time_per_restart,  # Increased from 30 to 50
            use_sa=False
        )
        
        if ils_cost < best_cost:
            print(f"  Restart {restart + 1}: NEW BEST {ils_cost}")
            best_routes = ils_routes
            best_cost = ils_cost
        else:
            print(f"  Restart {restart + 1}: {ils_cost}")
    
    print(f"  Best across restarts: {best_cost}")
    return best_routes, best_cost


def intra_route_2opt(routes: Dict[int, List[int]], distances: np.ndarray) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    2-opt within each route: reverse a segment to improve.
    """
    best_saving = 0
    best_move = None
    
    for r_id, route in routes.items():
        if len(route) < 3:
            continue
        
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                # Current edges: (prev_i, route[i]), (route[j], next_j)
                prev_i = route[i - 1] if i > 0 else 0
                next_j = route[j + 1] if j < len(route) - 1 else 0
                
                # After reversing [i, j]: (prev_i, route[j]), (route[i], next_j)
                old_cost = distances[prev_i, route[i]] + distances[route[j], next_j]
                new_cost = distances[prev_i, route[j]] + distances[route[i], next_j]
                
                saving = old_cost - new_cost
                if saving > best_saving:
                    best_saving = saving
                    best_move = (r_id, i, j)
    
    if best_move:
        r_id, i, j = best_move
        new_routes = {k: list(v) for k, v in routes.items()}
        # Reverse segment [i, j]
        new_routes[r_id] = routes[r_id][:i] + routes[r_id][i:j+1][::-1] + routes[r_id][j+1:]
        return new_routes, best_saving, True
    
    return routes, 0, False


def intra_route_3opt(routes: Dict[int, List[int]], distances: np.ndarray) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    3-opt within each route: reconnect 3 edges in the best way.
    Much stronger than 2-opt but slower.
    """
    best_saving = 0
    best_move = None
    
    for r_id, route in routes.items():
        if len(route) < 5:
            continue
        
        n = len(route)
        for i in range(n - 4):
            for j in range(i + 2, n - 2):
                for k in range(j + 2, n):
                    # Get nodes
                    prev_i = route[i - 1] if i > 0 else 0
                    node_i = route[i]
                    node_i1 = route[i + 1] if i + 1 < n else 0
                    
                    node_j = route[j]
                    node_j1 = route[j + 1] if j + 1 < n else 0
                    
                    node_k = route[k]
                    next_k = route[k + 1] if k + 1 < n else 0
                    if k == n - 1:
                        next_k = 0  # Return to depot
                    
                    # Original cost of 3 edges
                    old_cost = (distances[prev_i, node_i] + 
                               distances[route[j], node_j1] +
                               distances[node_k, next_k])
                    
                    # Try different reconnection options
                    # Option 1: reverse segment [i, j]
                    seg1 = route[i:j+1][::-1]
                    seg2 = route[j+1:k+1]
                    new_route = route[:i] + seg1 + seg2 + route[k+1:]
                    new_cost1 = calculate_route_cost(new_route, distances)
                    
                    # Option 2: reverse segment [j+1, k]
                    seg1 = route[i:j+1]
                    seg2 = route[j+1:k+1][::-1]
                    new_route = route[:i] + seg1 + seg2 + route[k+1:]
                    new_cost2 = calculate_route_cost(new_route, distances)
                    
                    # Option 3: swap segments
                    seg1 = route[j+1:k+1]
                    seg2 = route[i:j+1]
                    new_route = route[:i] + seg1 + seg2 + route[k+1:]
                    new_cost3 = calculate_route_cost(new_route, distances)
                    
                    original_cost = calculate_route_cost(route, distances)
                    
                    for opt, new_cost in [(1, new_cost1), (2, new_cost2), (3, new_cost3)]:
                        saving = original_cost - new_cost
                        if saving > best_saving:
                            best_saving = saving
                            best_move = (r_id, i, j, k, opt)
    
    if best_move:
        r_id, i, j, k, opt = best_move
        route = routes[r_id]
        new_routes = {key: list(val) for key, val in routes.items()}
        
        if opt == 1:
            seg1 = route[i:j+1][::-1]
            seg2 = route[j+1:k+1]
            new_routes[r_id] = route[:i] + seg1 + seg2 + route[k+1:]
        elif opt == 2:
            seg1 = route[i:j+1]
            seg2 = route[j+1:k+1][::-1]
            new_routes[r_id] = route[:i] + seg1 + seg2 + route[k+1:]
        else:
            seg1 = route[j+1:k+1]
            seg2 = route[i:j+1]
            new_routes[r_id] = route[:i] + seg1 + seg2 + route[k+1:]
        
        return new_routes, best_saving, True
    
    return routes, 0, False


def cross_exchange(routes: Dict[int, List[int]], distances: np.ndarray,
                   demands: np.ndarray, capacity: int,
                   max_seg_len: int = 3) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    Cross-Exchange: Exchange segments between two routes.
    """
    best_saving = 0
    best_move = None
    
    route_ids = list(routes.keys())
    
    for r1, r2 in combinations(route_ids, 2):
        route1, route2 = routes[r1], routes[r2]
        if not route1 or not route2:
            continue
        
        for len1 in range(1, min(len(route1) + 1, max_seg_len + 1)):
            for len2 in range(1, min(len(route2) + 1, max_seg_len + 1)):
                for i in range(len(route1) - len1 + 1):
                    for j in range(len(route2) - len2 + 1):
                        seg1 = route1[i:i + len1]
                        seg2 = route2[j:j + len2]
                        
                        # Check capacity
                        demand1 = sum(demands[c] for c in route1) - sum(demands[c] for c in seg1) + sum(demands[c] for c in seg2)
                        demand2 = sum(demands[c] for c in route2) - sum(demands[c] for c in seg2) + sum(demands[c] for c in seg1)
                        
                        if demand1 > capacity or demand2 > capacity:
                            continue
                        
                        # Calculate cost change
                        prev1 = route1[i - 1] if i > 0 else 0
                        next1 = route1[i + len1] if i + len1 < len(route1) else 0
                        prev2 = route2[j - 1] if j > 0 else 0
                        next2 = route2[j + len2] if j + len2 < len(route2) else 0
                        
                        old_cost = (distances[prev1, seg1[0]] + distances[seg1[-1], next1] +
                                   distances[prev2, seg2[0]] + distances[seg2[-1], next2])
                        new_cost = (distances[prev1, seg2[0]] + distances[seg2[-1], next1] +
                                   distances[prev2, seg1[0]] + distances[seg1[-1], next2])
                        
                        saving = old_cost - new_cost
                        if saving > best_saving:
                            best_saving = saving
                            best_move = (r1, i, len1, r2, j, len2)
    
    if best_move:
        r1, i, len1, r2, j, len2 = best_move
        new_routes = {k: list(v) for k, v in routes.items()}
        seg1 = routes[r1][i:i + len1]
        seg2 = routes[r2][j:j + len2]
        new_routes[r1] = routes[r1][:i] + seg2 + routes[r1][i + len1:]
        new_routes[r2] = routes[r2][:j] + seg1 + routes[r2][j + len2:]
        return new_routes, best_saving, True
    
    return routes, 0, False


def or_opt_search(routes: Dict[int, List[int]], distances: np.ndarray,
                  demands: np.ndarray, capacity: int, 
                  seq_len: int = 2) -> Tuple[Dict[int, List[int]], int, bool]:
    """
    Or-Opt: Move a sequence of consecutive customers to another route.
    """
    best_saving = 0
    best_move = None
    
    route_ids = list(routes.keys())
    
    for r1 in route_ids:
        route1 = routes[r1]
        if len(route1) < seq_len:
            continue
            
        for pos1 in range(len(route1) - seq_len + 1):
            # Sequence to move
            seq = route1[pos1:pos1 + seq_len]
            seq_demand = sum(demands[c] for c in seq)
            
            # Calculate removal cost
            prev1 = route1[pos1 - 1] if pos1 > 0 else 0
            next1 = route1[pos1 + seq_len] if pos1 + seq_len < len(route1) else 0
            
            old_links = distances[prev1, seq[0]] + distances[seq[-1], next1]
            new_link = distances[prev1, next1]
            remove_saving = old_links - new_link
            
            # Try inserting into other routes
            for r2 in route_ids:
                if r1 == r2:
                    continue
                
                route2 = routes[r2]
                demand_r2 = sum(demands[c] for c in route2)
                
                if demand_r2 + seq_demand > capacity:
                    continue
                
                # Find best insertion position
                for pos2 in range(len(route2) + 1):
                    prev2 = route2[pos2 - 1] if pos2 > 0 else 0
                    next2 = route2[pos2] if pos2 < len(route2) else 0
                    
                    insert_cost = (distances[prev2, seq[0]] + distances[seq[-1], next2] 
                                   - distances[prev2, next2])
                    
                    saving = remove_saving - insert_cost
                    
                    if saving > best_saving:
                        best_saving = saving
                        best_move = (r1, pos1, seq_len, r2, pos2, seq)
    
    if best_move:
        r1, pos1, seq_len, r2, pos2, seq = best_move
        # Apply move
        new_routes = {k: list(v) for k, v in routes.items()}
        new_routes[r1] = routes[r1][:pos1] + routes[r1][pos1 + seq_len:]
        new_routes[r2] = routes[r2][:pos2] + list(seq) + routes[r2][pos2:]
        
        return new_routes, best_saving, True
    
    return routes, 0, False


def pairwise_optimize(routes: Dict[int, List[int]], distances: np.ndarray,
                      demands: np.ndarray, capacity: int, 
                      max_iterations: int = 10) -> Tuple[Dict[int, List[int]], int]:
    """
    Tối ưu bằng cách gộp từng cặp routes và tối ưu lại.
    """
    print("\n=== Pair-wise Route Optimization ===")
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        print(f"\n  Iteration {iteration}:")
        
        route_ids = list(routes.keys())
        
        for i, j in combinations(route_ids, 2):
            if not routes[i] or not routes[j]:
                continue
            
            # Current cost of pair
            cost_i = calculate_route_cost(routes[i], distances)
            cost_j = calculate_route_cost(routes[j], distances)
            current_cost = cost_i + cost_j
            
            # Skip if too many customers (MaxSAT too slow)
            if len(routes[i]) + len(routes[j]) > 10:
                continue
            
            # Try to optimize this pair
            optimizer = PairwiseRouteOptimizer(
                routes[i], routes[j], distances, demands, capacity
            )
            new_route_i, new_route_j, new_cost = optimizer.optimize()
            
            if new_cost < current_cost:
                print(f"    Routes {i},{j}: {current_cost} -> {new_cost} (IMPROVED -{current_cost - new_cost})")
                routes[i] = new_route_i
                routes[j] = new_route_j
                improved = True
    
    total_cost = sum(calculate_route_cost(r, distances) for r in routes.values())
    return routes, total_cost


def optimize_all_routes(routes, distances, demands, capacity):
    print("\n=== MaxSAT Single Route Optimization ===")
    opt_routes = {}
    total = 0
    
    # Duyệt qua từng tuyến xe (route)
    for v, r in routes.items():
        # [CRITICAL FIX] Ngưỡng an toàn tuyệt đối là 11 điểm.
        # Lý do: Với bộ giải PySAT thuần Python, các bài toán TSP > 11 điểm 
        # có thể gây bùng nổ tổ hợp và treo máy hàng tiếng đồng hồ.
        # Chiến lược: "Thà bỏ sót còn hơn giết nhầm" - Chỉ tối ưu các route ngắn.
        if len(r) > 11: 
            opt_routes[v] = r
            total += calculate_route_cost(r, distances)
            # In dấu chấm than (!) để báo hiệu cho người dùng biết tuyến này đã bị BỎ QUA
            print("!", end="", flush=True) 
            continue
            
        # In dấu chấm (.) để báo hiệu đang chạy Max-SAT cho route ngắn này
        print(".", end="", flush=True) 
        
        # Gọi bộ giải Max-SAT cho bài toán con
        opt = SingleRouteOptimizer(r, distances)
        new_r, cost = opt.optimize()
        
        # Kiểm tra lại tải trọng (Safety check)
        # Mặc dù tối ưu lại thứ tự thường không thay đổi tổng tải trọng, 
        # nhưng bước này đảm bảo tính toàn vẹn dữ liệu.
        if sum(demands[c] for c in new_r) <= capacity:
            opt_routes[v] = new_r
            total += cost
        else:
            # Nếu (rất hiếm) vi phạm tải trọng, giữ lại route cũ
            opt_routes[v] = r
            total += calculate_route_cost(r, distances)
            
    print("\n  Single Route Optimization done.")
    return opt_routes, total


def solve_with_clarke_wright_and_optimize(filepath: str, verbose: bool = True):
    """
    Solve CVRP using Clarke-Wright + MaxSAT route optimization.
    Added 'verbose' parameter to suppress output during benchmarking.
    """
    import re
    from classes.instance import Instance
    from classes.clarke_wright import ClarkeWright
    from classes.two_opt import TwoOpt
    
    if verbose:
        print("=" * 60)
        print("CVRP Solver: Clarke-Wright + MaxSAT Optimization")
        print("=" * 60)
    
    # Load instance
    cvrp = Instance(filepath)
    cvrp.load()
    
    n_vehicles = int(re.search(r'-k(\d+)', filepath).group(1))
    
    if verbose:
        print(f"\nInstance: {filepath}")
        print(f"  Dimension: {cvrp.dimension}")
        print(f"  Vehicles: {n_vehicles}")
        print(f"  Capacity: {cvrp.capacity}")
    
    # Step 1: Clarke-Wright
    if verbose: print("\n=== Step 1: Clarke-Wright Heuristic ===")
    cw_time, cw_routes = ClarkeWright.run(cvrp, n_vehicles)
    
    cw_cost = sum(route.cost for route in cw_routes.values())
    if verbose:
        print(f"  Time: {cw_time:.3f}s")
        print(f"  Cost: {cw_cost}")
    
    # Convert to our format
    routes = {i: list(route.value) for i, (_, route) in enumerate(cw_routes.items())}
    
    # Step 2: Two-Opt improvement
    if verbose: print("\n=== Step 2: Two-Opt Local Search ===")
    two_opt_time, two_opt_routes = TwoOpt.run(cw_routes)
    two_opt_cost = sum(route.cost for route in two_opt_routes.values())
    if verbose:
        print(f"  Time: {two_opt_time:.3f}s")
        print(f"  Cost: {two_opt_cost}")
    
    routes = {i: list(route.value) for i, (_, route) in enumerate(two_opt_routes.items())}
    demands = np.array(cvrp.demands)
    
    # Step 3: Multi-Start Iterated Local Search
    if verbose: print("\n=== Step 3: Multi-Start ILS ===")
    start_time = time.time()
    
    # Adaptive parameters based on instance size - GREEDY settings
    n_customers = len(demands) - 1  # Exclude depot
    if n_customers > 80:
        n_restarts = 50
        time_limit = 300.0  # 5 minutes
    elif n_customers > 50:
        n_restarts = 40
        time_limit = 240.0  # 4 minutes
    else:
        n_restarts = 50  # More restarts for small instances
        time_limit = 180.0  # 3 minutes
    
    ils_routes, ils_cost = multi_start_ils(
        routes.copy(), cvrp.distances, demands, cvrp.capacity,
        n_restarts=n_restarts, time_limit=time_limit
    )
    ils_time = time.time() - start_time
    if verbose:
        print(f"\n  Time: {ils_time:.3f}s")
        print(f"  Cost: {ils_cost}")
    
    # Step 4: Guided Local Search (GLS)
    if verbose: print("\n=== Step 4: Guided Local Search ===")
    start_time = time.time()
    
    # GLS time based on instance size
    gls_time_limit = 60.0 if n_customers > 50 else 30.0
    gls_routes, gls_cost = guided_local_search(
        ils_routes.copy(), cvrp.distances, demands, cvrp.capacity,
        max_iterations=200, time_limit=gls_time_limit
    )
    gls_time = time.time() - start_time
    if verbose:
        print(f"  Time: {gls_time:.3f}s")
        print(f"  Cost: {gls_cost}")
    
    # Step 5: MaxSAT single route optimization
    if verbose: print("\n=== Step 5: MaxSAT Single Route Optimization ===")
    start_time = time.time()
    
    opt_routes, opt_cost = optimize_all_routes(gls_routes, cvrp.distances, demands, cvrp.capacity)
    single_time = time.time() - start_time
    if verbose:
        print(f"  Time: {single_time:.3f}s")
        print(f"  Cost: {opt_cost}")
    
    # Step 6: MaxSAT pair-wise optimization (Inter-route)
    if verbose: print("\n=== Step 6: MaxSAT Pair-wise Optimization ===")
    start_time = time.time()
    
    pair_routes, pair_cost = pairwise_optimize(
        opt_routes.copy(), cvrp.distances, demands, cvrp.capacity, max_iterations=5
    )
    pair_time = time.time() - start_time
    if verbose:
        print(f"\n  Time: {pair_time:.3f}s")
        print(f"  Cost: {pair_cost}")
    
    # Get optimal from solution file
    sol_file = filepath.replace('.vrp', '.sol')
    optimal = None
    if os.path.exists(sol_file):
        with open(sol_file) as f:
            for line in f:
                if line.startswith('Cost'):
                    optimal = int(line.split()[1])
                    break
    
    # Summary
    if verbose:
        total_time = cw_time + two_opt_time + ils_time + gls_time + single_time + pair_time
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Clarke-Wright:    {cw_cost}")
        print(f"  + Two-Opt:        {two_opt_cost} (improved {cw_cost - two_opt_cost})")
        print(f"  + ILS:            {ils_cost} (improved {two_opt_cost - ils_cost})")
        print(f"  + GLS:            {gls_cost} (improved {ils_cost - gls_cost})")
        print(f"  + MaxSAT Single:  {opt_cost} (improved {gls_cost - opt_cost})")
        print(f"  + MaxSAT Pair:    {pair_cost} (improved {opt_cost - pair_cost})")
        print(f"  Total time:       {total_time:.3f}s")
        if optimal:
            print(f"\n  Optimal (from file): {optimal}")
            print(f"  Gap: {(pair_cost - optimal) / optimal * 100:.1f}%")
        print("\nFinal Routes:")
        for v, route in pair_routes.items():
            demand = sum(demands[c] for c in route)
            cost = calculate_route_cost(route, cvrp.distances)
            print(f"  Route {v}: {route} (demand={demand}, cost={cost})")
    
    return pair_routes, pair_cost

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "instances/E-n31-k7.vrp"
    
    solve_with_clarke_wright_and_optimize(filepath)