#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Incremental MaxSAT Solver for CVRP
===================================
Adapted from RTSS (Real-time Taxi-Sharing Service) approach by Zha et al.

Key Ideas from RTSS:
1. Use connection variables l(i,j) and reachability variables r(i,j)
2. Lazy constraint checking (capacity violations added incrementally)
3. Use PySAT RC2 solver with incremental mode
4. Use Clarke-Wright heuristic for initial solution to restrict search space

Author: Adapted for CVRP from RTSS paper
"""

import sys
import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

# Add classes path for Clarke-Wright
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class IncrementalCVRPSolver:
    """
    Incremental MaxSAT solver for CVRP using RTSS methodology.
    
    Key differences from original CVRP approach:
    - Uses reachability variables instead of MTZ for subtour elimination
    - Lazy constraint checking for capacity constraints
    - Incremental solving with PySAT RC2
    - Uses Clarke-Wright for initial solution
    """
    
    def __init__(self, n_customers: int, n_vehicles: int, 
                 distances: np.ndarray, demands: np.ndarray, capacity: int,
                 k_neighbors: int = 5):
        """
        Initialize the solver.
        
        Args:
            n_customers: Number of customers (excluding depot)
            n_vehicles: Number of vehicles
            distances: Distance matrix (n+1 x n+1), index 0 is depot
            demands: Demand array, demands[0] = 0 for depot
            capacity: Vehicle capacity
            k_neighbors: Number of nearest neighbors to consider
        """
        self.n = n_customers + 1  # Including depot
        self.m = n_vehicles
        self.distances = distances
        self.demands = demands
        self.capacity = capacity
        self.k_neighbors = k_neighbors
        
        # Variable counters
        self.var_id = 0
        
        # Connection network: conNet[v][i][j] = variable for edge i->j on vehicle v
        # conNet[v][i][j] = True means vehicle v goes directly from i to j
        self.conNet: List[List[List[int]]] = [
            [[0] * self.n for _ in range(self.n)] for _ in range(self.m)
        ]
        
        # Reachability network: rchNet[v][i][j] = variable for reachability
        # rchNet[v][i][j] = True means i is visited before j on vehicle v's route
        self.rchNet: List[List[List[int]]] = [
            [[0] * self.n for _ in range(self.n)] for _ in range(self.m)
        ]
        
        # K-nearest neighbor mask (which edges to consider)
        self.valid_edges: List[List[List[bool]]] = [
            [[False] * self.n for _ in range(self.n)] for _ in range(self.m)
        ]
        
        # WCNF formula
        self.wcnf = WCNF()
        self.top = 1 + int(np.sum(distances))  # Top weight for hard clauses
        
        # Learnt clauses (for debugging)
        self.learnt_clauses: List[List[int]] = []
        
        # Statistics
        self.n_iterations = 0
        self.n_capacity_violations = 0
        
    def new_var_id(self) -> int:
        """Generate a new variable ID."""
        self.var_id += 1
        return self.var_id
    
    def compute_k_neighbors(self, initial_routes: Optional[Dict[int, List[int]]] = None):
        """
        Compute valid edges based on K-nearest neighbors.
        
        If initial_routes is provided from Clarke-Wright:
        - Edges in the initial routes are always valid
        - Also allow swaps between customers in the same or adjacent routes
        """
        # If we have Clarke-Wright solution, use it to restrict search space
        if initial_routes:
            print("  Using Clarke-Wright routes to restrict search space...")
            
            # First, add all edges from initial routes
            for v, route in initial_routes.items():
                full_route = [0] + route + [0]
                for idx in range(len(full_route) - 1):
                    i, j = full_route[idx], full_route[idx + 1]
                    # Allow this edge on ANY vehicle (for flexibility)
                    for vv in range(self.m):
                        self.valid_edges[vv][i][j] = True
                        self.valid_edges[vv][j][i] = True
                
                # Allow edges between consecutive customers in the route
                for idx1 in range(len(route)):
                    for idx2 in range(idx1 + 1, min(idx1 + 3, len(route))):  # 2-opt neighborhood
                        c1, c2 = route[idx1], route[idx2]
                        for vv in range(self.m):
                            self.valid_edges[vv][c1][c2] = True
                            self.valid_edges[vv][c2][c1] = True
            
            # Add K-nearest neighbors for local improvement
            for i in range(1, self.n):  # Skip depot
                dists = [(j, self.distances[i, j]) for j in range(1, self.n) if i != j]
                dists.sort(key=lambda x: x[1])
                k_nearest = [j for j, _ in dists[:self.k_neighbors]]
                
                for v in range(self.m):
                    for j in k_nearest:
                        self.valid_edges[v][i][j] = True
                        self.valid_edges[v][j][i] = True
        else:
            # Standard K-nearest without initial solution
            for i in range(self.n):
                dists = [(j, self.distances[i, j]) for j in range(self.n) if i != j]
                dists.sort(key=lambda x: x[1])
                k_nearest = [j for j, _ in dists[:self.k_neighbors]]
                
                for v in range(self.m):
                    for j in k_nearest:
                        self.valid_edges[v][i][j] = True
                        self.valid_edges[v][j][i] = True
        
        # Always allow depot connections
        for v in range(self.m):
            for i in range(self.n):
                self.valid_edges[v][0][i] = True
                self.valid_edges[v][i][0] = True
    
    def is_valid_edge(self, v: int, i: int, j: int) -> bool:
        """Check if edge (i,j) is valid for vehicle v."""
        if i == j:
            return False
        return self.valid_edges[v][i][j]
    
    def gen_var_for_con_net(self):
        """Generate variables for connection network."""
        for v in range(self.m):
            for i in range(self.n):
                for j in range(self.n):
                    if self.is_valid_edge(v, i, j):
                        self.conNet[v][i][j] = self.new_var_id()
    
    def gen_var_for_rch_net(self):
        """
        Generate variables for reachability network.
        
        Key insight from RTSS:
        - r(i,j) means "i is visited before j"
        - r(i,j) and r(j,i) are negations of each other for customers
        - This reduces number of variables needed
        """
        for v in range(self.m):
            # For customers only (not depot)
            for i in range(1, self.n):
                for j in range(i + 1, self.n):
                    if self.is_valid_edge(v, i, j) or self.is_valid_edge(v, j, i):
                        var = self.new_var_id()
                        self.rchNet[v][i][j] = var
                        self.rchNet[v][j][i] = -var  # Negation
            
            # For depot-customer reachability (depot is always first)
            for j in range(1, self.n):
                if any(self.is_valid_edge(v, 0, k) or self.is_valid_edge(v, k, 0) for k in range(self.n)):
                    # Depot is always before customers - this is a tautology
                    # We don't need explicit variables for r(0, j)
                    pass
    
    def gen_soft_clauses(self):
        """
        Generate soft clauses for objective function.
        
        For each edge (i,j) on vehicle v, add soft clause:
        (¬conNet[v][i][j], weight=distance[i][j])
        
        Meaning: We pay the distance if we use the edge.
        The solver minimizes the total weight of unsatisfied soft clauses.
        If we use edge (i,j), we falsify the clause ¬conNet[v][i][j], paying distance[i,j].
        """
        for v in range(self.m):
            for i in range(self.n):
                for j in range(self.n):
                    if self.is_valid_edge(v, i, j) and self.conNet[v][i][j] != 0:
                        weight = int(self.distances[i, j])
                        if weight > 0:
                            # Soft clause: prefer NOT using this edge
                            # If edge is used, pay the weight
                            self.wcnf.append([-self.conNet[v][i][j]], weight=weight)
    
    def gen_hard_clause_implication(self):
        """
        If edge (i,j) is used, then i is before j in reachability.
        conNet[v][i][j] → rchNet[v][i][j]
        Equivalent: ¬conNet[v][i][j] ∨ rchNet[v][i][j]
        """
        for v in range(self.m):
            for i in range(1, self.n):  # Skip depot as source
                for j in range(1, self.n):  # Skip depot as target
                    if i != j and self.is_valid_edge(v, i, j):
                        if self.conNet[v][i][j] != 0 and self.rchNet[v][i][j] != 0:
                            self.wcnf.append([-self.conNet[v][i][j], self.rchNet[v][i][j]])
    
    def gen_hard_clause_transitivity(self):
        """
        Transitivity of reachability:
        r(a,b) ∧ r(b,c) → r(a,c)
        Equivalent: ¬r(a,b) ∨ ¬r(b,c) ∨ r(a,c)
        """
        for v in range(self.m):
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
    
    def gen_hard_clause_chain(self):
        """
        Chain law: If a < b and b < c, then edge (c,a) is not allowed.
        r(a,b) ∧ r(b,c) → ¬conNet[c][a]
        Equivalent: ¬r(a,b) ∨ ¬r(b,c) ∨ ¬conNet[c][a]
        """
        for v in range(self.m):
            for a in range(1, self.n):
                for b in range(1, self.n):
                    if a == b:
                        continue
                    for c in range(1, self.n):
                        if c == a or c == b:
                            continue
                        
                        r_ab = self.rchNet[v][a][b]
                        r_bc = self.rchNet[v][b][c]
                        con_ca = self.conNet[v][c][a]
                        
                        if r_ab != 0 and r_bc != 0 and con_ca != 0:
                            self.wcnf.append([-r_ab, -r_bc, -con_ca])
    
    def gen_hard_clause_acyclic(self):
        """
        Acyclic constraint: ¬r(a,b) ∨ ¬r(b,a)
        Since r(b,a) = -r(a,b), this is automatically satisfied
        when we use the negation trick.
        """
        # This is implicitly enforced by our variable encoding
        pass
    
    def gen_hard_clause_depot_out(self):
        """
        Each vehicle must leave the depot exactly once.
        Σ_j conNet[v][0][j] = 1 for each vehicle v
        """
        for v in range(self.m):
            var_list = []
            for j in range(1, self.n):
                if self.is_valid_edge(v, 0, j) and self.conNet[v][0][j] != 0:
                    var_list.append(self.conNet[v][0][j])
            
            if var_list:
                # At least one
                self.wcnf.append(var_list)
                # At most one
                for i in range(len(var_list)):
                    for j in range(i + 1, len(var_list)):
                        self.wcnf.append([-var_list[i], -var_list[j]])
    
    def gen_hard_clause_depot_in(self):
        """
        Each vehicle must return to the depot exactly once.
        Σ_i conNet[v][i][0] = 1 for each vehicle v
        """
        for v in range(self.m):
            var_list = []
            for i in range(1, self.n):
                if self.is_valid_edge(v, i, 0) and self.conNet[v][i][0] != 0:
                    var_list.append(self.conNet[v][i][0])
            
            if var_list:
                # At least one
                self.wcnf.append(var_list)
                # At most one
                for i in range(len(var_list)):
                    for j in range(i + 1, len(var_list)):
                        self.wcnf.append([-var_list[i], -var_list[j]])
    
    def gen_hard_clause_flow_conservation(self):
        """
        Flow conservation: For each customer, exactly one edge in and one edge out.
        GLOBALLY - each customer is visited by exactly one vehicle.
        
        Σ_i,v conNet[v][i][j] = 1 (one edge entering j)
        Σ_k,v conNet[v][j][k] = 1 (one edge leaving j)
        """
        for j in range(1, self.n):  # For each customer
            # Incoming edges (across ALL vehicles)
            in_vars = []
            for v in range(self.m):
                for i in range(self.n):
                    if i != j and self.is_valid_edge(v, i, j) and self.conNet[v][i][j] != 0:
                        in_vars.append(self.conNet[v][i][j])
            
            if in_vars:
                # Exactly one incoming - THIS IS A HARD CONSTRAINT
                self.wcnf.append(in_vars)  # At least one
                for i in range(len(in_vars)):
                    for k in range(i + 1, len(in_vars)):
                        self.wcnf.append([-in_vars[i], -in_vars[k]])  # At most one
            else:
                print(f"WARNING: Customer {j} has no valid incoming edges!")
            
            # Outgoing edges (across ALL vehicles)
            out_vars = []
            for v in range(self.m):
                for k in range(self.n):
                    if k != j and self.is_valid_edge(v, j, k) and self.conNet[v][j][k] != 0:
                        out_vars.append(self.conNet[v][j][k])
            
            if out_vars:
                # Exactly one outgoing
                self.wcnf.append(out_vars)  # At least one
                for i in range(len(out_vars)):
                    for k in range(i + 1, len(out_vars)):
                        self.wcnf.append([-out_vars[i], -out_vars[k]])  # At most one
            else:
                print(f"WARNING: Customer {j} has no valid outgoing edges!")
    
    def gen_hard_clause_vehicle_consistency(self):
        """
        Vehicle consistency: If a customer is visited by vehicle v,
        BOTH the incoming and outgoing edges must belong to vehicle v.
        
        For each customer j (not depot):
        If conNet[v][i][j] = True for some i, then all outgoing edges 
        from j on OTHER vehicles must be False.
        
        conNet[v][i][j] => ¬conNet[v'][j][k] for all v' ≠ v
        Equivalent: ¬conNet[v][i][j] ∨ ¬conNet[v'][j][k]
        """
        for j in range(1, self.n):  # For each customer (not depot)
            for v in range(self.m):
                for v_prime in range(self.m):
                    if v == v_prime:
                        continue
                    
                    # Get all incoming edges to j on vehicle v
                    for i in range(self.n):
                        if i != j and self.is_valid_edge(v, i, j) and self.conNet[v][i][j] != 0:
                            # Get all outgoing edges from j on vehicle v'
                            for k in range(self.n):
                                if k != j and self.is_valid_edge(v_prime, j, k) and self.conNet[v_prime][j][k] != 0:
                                    # If j gets edge from v, then no outgoing edge on v'
                                    self.wcnf.append([-self.conNet[v][i][j], -self.conNet[v_prime][j][k]])
    
    def gen_hard_clause_same_vehicle(self):
        """
        If edge (i,j) is used by vehicle v, both i and j are on vehicle v's route.
        This is enforced by the flow conservation constraints.
        
        Additional constraint: Prevent using both (i,j) and (j,i) on same vehicle.
        """
        for v in range(self.m):
            for i in range(1, self.n):
                for j in range(i + 1, self.n):
                    con_ij = self.conNet[v][i][j]
                    con_ji = self.conNet[v][j][i]
                    if con_ij != 0 and con_ji != 0:
                        # Cannot have both edges
                        self.wcnf.append([-con_ij, -con_ji])
    
    def gen_hard_clause_confluence(self):
        """
        Confluence law from RTSS (Eq. 20):
        If both i and k can reach j, then i and k must be ordered.
        r(i,j) ∧ r(k,j) → (r(i,k) ∨ r(k,i))
        Equivalent: ¬r(i,j) ∨ ¬r(k,j) ∨ r(i,k) ∨ r(k,i)
        
        Since r(k,i) = -r(i,k), this becomes:
        ¬r(i,j) ∨ ¬r(k,j) ∨ r(i,k) ∨ -r(i,k) which is always true.
        
        But due to our encoding where we only create r(a,b) for a < b,
        we need to handle the case where variables exist.
        """
        for v in range(self.m):
            for j in range(1, self.n):  # Target
                for i in range(1, self.n):
                    if i == j:
                        continue
                    for k in range(i + 1, self.n):  # Ensure i < k to avoid duplicates
                        if k == j:
                            continue
                        
                        # Check if variables exist
                        r_ij = self.rchNet[v][i][j]
                        r_kj = self.rchNet[v][k][j]
                        r_ik = self.rchNet[v][i][k]  # This exists for i < k
                        
                        if r_ij != 0 and r_kj != 0 and r_ik != 0:
                            # ¬r(i,j) ∨ ¬r(k,j) ∨ r(i,k) ∨ r(k,i)
                            # Since r(k,i) = -r(i,k), this is: ¬r(i,j) ∨ ¬r(k,j) ∨ r(i,k) ∨ ¬r(i,k)
                            # This is a tautology for the r_ik part, so we only need:
                            # Actually we need to think more carefully...
                            # The constraint says: if r(i,j) and r(k,j), then r(i,k) or r(k,i)
                            # Since exactly one of r(i,k) or r(k,i) is true (by our encoding),
                            # this is always satisfied! So no clause needed.
                            pass
    
    def gen_hard_clause_ramification(self):
        """
        Ramification law from RTSS (Eq. 21):
        If i can reach both j and k, then j and k must be ordered.
        r(i,j) ∧ r(i,k) → (r(j,k) ∨ r(k,j))
        
        Similar to confluence, this is automatically satisfied by our encoding
        since exactly one of r(j,k) or r(k,j) is true for any j ≠ k.
        """
        # By our symmetric encoding, this is automatically satisfied
        pass
    
    def gen_all_hard_clauses(self):
        """Generate all hard clauses for the basic model."""
        print("Generating hard clauses...")
        self.gen_hard_clause_implication()
        self.gen_hard_clause_transitivity()
        self.gen_hard_clause_chain()
        # Confluence and Ramification are automatically satisfied by our symmetric encoding
        # self.gen_hard_clause_confluence()  # Not needed - tautology
        # self.gen_hard_clause_ramification()  # Not needed - tautology
        self.gen_hard_clause_depot_out()
        self.gen_hard_clause_depot_in()
        self.gen_hard_clause_flow_conservation()
        self.gen_hard_clause_vehicle_consistency()  # Ensure same vehicle for in/out
        self.gen_hard_clause_same_vehicle()
        print(f"  Total clauses: {len(self.wcnf.hard) + len(self.wcnf.soft)}")
    
    def decode_route(self, model: List[int], vehicle: int) -> List[int]:
        """
        Decode the route for a specific vehicle from the model.
        
        Returns: List of customer indices in visiting order.
        """
        # Find which conNet variables are true
        route = []
        current = 0  # Start at depot
        visited = {0}
        
        while True:
            found_next = False
            for j in range(self.n):
                if j not in visited and self.is_valid_edge(vehicle, current, j):
                    var = self.conNet[vehicle][current][j]
                    if var != 0 and var in model:
                        if j == 0:  # Back to depot
                            return route
                        route.append(j)
                        visited.add(j)
                        current = j
                        found_next = True
                        break
            
            if not found_next:
                break
        
        return route
    
    def decode_all_routes(self, model: List[int]) -> Dict[int, List[int]]:
        """Decode routes for all vehicles."""
        # Filter to positive conNet variables
        positive_model = [v for v in model if v > 0 and v <= self.var_id]
        
        routes = {}
        for v in range(self.m):
            route = self.decode_route(positive_model, v)
            if route:
                routes[v] = route
        
        return routes
    
    def check_capacity(self, route: List[int]) -> Tuple[bool, int]:
        """
        Check if a route violates capacity constraint.
        
        Returns: (is_valid, total_demand)
        """
        total_demand = sum(self.demands[c] for c in route)
        return total_demand <= self.capacity, total_demand
    
    def check_solution(self, routes: Dict[int, List[int]]) -> Tuple[bool, Optional[Tuple[int, List[int]]]]:
        """
        Check if a solution is feasible.
        
        Returns: (is_feasible, (violating_vehicle, violating_route) or None)
        """
        for v, route in routes.items():
            is_valid, demand = self.check_capacity(route)
            if not is_valid:
                return False, (v, route)
        return True, None
    
    def generate_nogood_clause(self, vehicle: int, route: List[int]) -> List[int]:
        """
        Generate a nogood clause for a violating route.
        
        The clause says: "Don't use all these edges together"
        ¬(e1 ∧ e2 ∧ ... ∧ en) = ¬e1 ∨ ¬e2 ∨ ... ∨ ¬en
        """
        clause = []
        full_route = [0] + route + [0]  # depot -> route -> depot
        
        for i in range(len(full_route) - 1):
            src, dst = full_route[i], full_route[i + 1]
            var = self.conNet[vehicle][src][dst]
            if var != 0:
                clause.append(-var)
        
        return clause
    
    def solve_incremental(self, timeout: int = 300) -> Tuple[Optional[Dict[int, List[int]]], float, int]:
        """
        Solve CVRP using incremental MaxSAT.
        
        Key idea from RTSS:
        1. Solve the base model (without capacity constraints)
        2. Check if solution violates capacity
        3. If violation, add nogood clause and re-solve
        4. Repeat until feasible or UNSAT
        
        Returns: (routes, cost, n_iterations)
        """
        print("\n=== Incremental MaxSAT Solving ===")
        start_time = time.time()
        
        with RC2(self.wcnf, incr=True, verbose=0) as solver:
            best_routes = None
            best_cost = float('inf')
            
            while True:
                self.n_iterations += 1
                elapsed = time.time() - start_time
                
                if elapsed > timeout:
                    print(f"Timeout after {timeout}s")
                    break
                
                print(f"\nIteration {self.n_iterations}...")
                
                # Solve current model
                model = solver.compute()
                
                if model is None:
                    print("  UNSAT - No more solutions")
                    break
                
                # Decode routes
                routes = self.decode_all_routes(model)
                cost = solver.cost
                
                print(f"  Found solution with cost {cost}")
                for v, route in routes.items():
                    demand = sum(self.demands[c] for c in route)
                    print(f"    Vehicle {v}: {route} (demand={demand}/{self.capacity})")
                
                # Check feasibility
                is_feasible, violation = self.check_solution(routes)
                
                if is_feasible:
                    print("  Solution is FEASIBLE!")
                    best_routes = routes
                    best_cost = cost
                    break
                else:
                    v, violating_route = violation
                    demand = sum(self.demands[c] for c in violating_route)
                    print(f"  CAPACITY VIOLATION: Vehicle {v}, demand={demand}/{self.capacity}")
                    
                    self.n_capacity_violations += 1
                    
                    # Generate and add nogood clause
                    nogood = self.generate_nogood_clause(v, violating_route)
                    solver.add_clause(nogood)
                    self.learnt_clauses.append(nogood)
                    
                    print(f"  Added nogood clause with {len(nogood)} literals")
        
        elapsed = time.time() - start_time
        print(f"\n=== Solving Complete ===")
        print(f"Time: {elapsed:.2f}s")
        print(f"Iterations: {self.n_iterations}")
        print(f"Capacity violations found: {self.n_capacity_violations}")
        
        return best_routes, best_cost if best_routes else float('inf'), elapsed
    
    def build_and_solve(self, initial_routes: Optional[Dict[int, List[int]]] = None,
                        timeout: int = 300) -> Tuple[Optional[Dict[int, List[int]]], float, float]:
        """
        Build the model and solve.
        
        Args:
            initial_routes: Optional initial solution for K-neighbors
            timeout: Timeout in seconds
            
        Returns: (routes, cost, time)
        """
        print("=== Building Incremental CVRP Model ===")
        print(f"Customers: {self.n - 1}")
        print(f"Vehicles: {self.m}")
        print(f"Capacity: {self.capacity}")
        print(f"K-neighbors: {self.k_neighbors}")
        
        build_start = time.time()
        
        # Step 1: Compute valid edges
        print("\nComputing K-nearest neighbors...")
        self.compute_k_neighbors(initial_routes)
        
        # Count valid edges
        n_valid = sum(
            1 for v in range(self.m) 
            for i in range(self.n) 
            for j in range(self.n) 
            if self.is_valid_edge(v, i, j)
        )
        print(f"  Valid edges: {n_valid}")
        
        # Step 2: Generate variables
        print("\nGenerating variables...")
        self.gen_var_for_con_net()
        self.gen_var_for_rch_net()
        print(f"  Total variables: {self.var_id}")
        
        # Step 3: Generate clauses
        print("\nGenerating soft clauses (objective)...")
        self.gen_soft_clauses()
        
        self.gen_all_hard_clauses()
        
        build_time = time.time() - build_start
        print(f"\nModel built in {build_time:.2f}s")
        print(f"  Hard clauses: {len(self.wcnf.hard)}")
        print(f"  Soft clauses: {len(self.wcnf.soft)}")
        
        # Step 4: Solve
        routes, cost, solve_time = self.solve_incremental(timeout)
        
        return routes, cost, build_time + solve_time


def load_vrp_instance(filepath: str) -> Tuple[int, int, np.ndarray, np.ndarray, int]:
    """
    Load a VRP instance from a .vrp file.
    Supports both NODE_COORD_SECTION (Euclidean) and EDGE_WEIGHT_SECTION (explicit).
    
    Returns: (n_customers, n_vehicles, distances, demands, capacity)
    """
    import re
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse dimension
    dim_match = re.search(r'DIMENSION\s*:\s*(\d+)', content)
    dimension = int(dim_match.group(1)) if dim_match else 0
    
    # Parse capacity
    cap_match = re.search(r'CAPACITY\s*:\s*(\d+)', content)
    capacity = int(cap_match.group(1)) if cap_match else 0
    
    # Parse vehicles (from name like A-n33-k6 means 6 vehicles)
    k_match = re.search(r'-k(\d+)', filepath)
    n_vehicles = int(k_match.group(1)) if k_match else 1
    
    # Parse demands
    demands = np.zeros(dimension, dtype=int)
    demand_section = re.search(r'DEMAND_SECTION\s*(.*?)\s*DEPOT_SECTION', content, re.DOTALL)
    if demand_section:
        for line in demand_section.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                idx = int(parts[0])
                demand = int(parts[1])
                if idx <= dimension:
                    demands[idx - 1] = demand  # Convert to 0-indexed
    
    # Check edge weight type
    edge_weight_type = re.search(r'EDGE_WEIGHT_TYPE\s*:\s*(\w+)', content)
    edge_weight_format = re.search(r'EDGE_WEIGHT_FORMAT\s*:\s*(\w+)', content)
    
    distances = np.zeros((dimension, dimension), dtype=int)
    
    if edge_weight_type and edge_weight_type.group(1) == 'EXPLICIT':
        # Parse explicit edge weights
        edge_section = re.search(r'EDGE_WEIGHT_SECTION\s*(.*?)\s*DEMAND_SECTION', content, re.DOTALL)
        if edge_section:
            # Parse all numbers from the edge weight section
            numbers = list(map(int, edge_section.group(1).split()))
            
            fmt = edge_weight_format.group(1) if edge_weight_format else 'LOWER_ROW'
            
            if fmt == 'LOWER_ROW':
                # Lower triangular matrix without diagonal
                idx = 0
                for i in range(1, dimension):
                    for j in range(i):
                        if idx < len(numbers):
                            distances[i, j] = numbers[idx]
                            distances[j, i] = numbers[idx]  # Symmetric
                            idx += 1
            elif fmt == 'FULL_MATRIX':
                idx = 0
                for i in range(dimension):
                    for j in range(dimension):
                        if idx < len(numbers):
                            distances[i, j] = numbers[idx]
                            idx += 1
    else:
        # Parse node coordinates and compute Euclidean distance
        coords = {}
        coord_section = re.search(r'NODE_COORD_SECTION\s*(.*?)\s*DEMAND_SECTION', content, re.DOTALL)
        if coord_section:
            for line in coord_section.group(1).strip().split('\n'):
                parts = line.split()
                if len(parts) >= 3:
                    idx = int(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    coords[idx] = (x, y)
        
        for i in range(dimension):
            for j in range(dimension):
                if i != j and (i + 1) in coords and (j + 1) in coords:
                    xi, yi = coords[i + 1]
                    xj, yj = coords[j + 1]
                    distances[i, j] = int(np.sqrt((xi - xj)**2 + (yi - yj)**2) + 0.5)
    
    return dimension - 1, n_vehicles, distances, demands, capacity


def get_clarke_wright_solution(filepath: str):
    """
    Get initial solution using Clarke-Wright heuristic.
    
    Returns: (routes_dict, total_cost, cvrp_instance)
    """
    try:
        from classes.instance import Instance
        from classes.clarke_wright import ClarkeWright
        
        # Load instance
        cvrp = Instance(filepath)
        cvrp.load()
        
        # Get number of vehicles from filename (e.g., E-n31-k7 -> 7 vehicles)
        import re
        match = re.search(r'-k(\d+)', filepath)
        n_vehicles = int(match.group(1)) if match else 7
        
        # Run Clarke-Wright
        print(f"\n=== Running Clarke-Wright Heuristic ===")
        result = ClarkeWright.run(cvrp, n_vehicles)
        
        # ClarkeWright.run returns (time, routes) due to @Utils.timer decorator
        if isinstance(result, tuple) and len(result) == 2:
            cw_time, routes = result
            print(f"  Clarke-Wright time: {cw_time:.3f}s")
        else:
            routes = result
        
        # Convert to our format: {vehicle_id: [customer_list]}
        routes_dict = {}
        total_cost = 0
        
        for i, (key, route) in enumerate(routes.items()):
            routes_dict[i] = list(route.value)  # Convert Route to list
            total_cost += route.cost
        
        print(f"Clarke-Wright solution:")
        print(f"  Total cost: {total_cost}")
        print(f"  Number of routes: {len(routes_dict)}")
        for v, r in routes_dict.items():
            print(f"    Route {v}: {r}")
        
        return routes_dict, total_cost, cvrp
        
    except Exception as e:
        import traceback
        print(f"Clarke-Wright failed: {e}")
        traceback.print_exc()
        return None, None, None


def solve_vrp_file(filepath: str, k_neighbors: int = 5, timeout: int = 300, 
                   use_clarke_wright: bool = True):
    """
    Load and solve a VRP instance from file.
    
    Args:
        filepath: Path to VRP file
        k_neighbors: Number of nearest neighbors for search space
        timeout: Timeout in seconds
        use_clarke_wright: Use Clarke-Wright for initial solution
    """
    print(f"Loading instance: {filepath}")
    n_customers, n_vehicles, distances, demands, capacity = load_vrp_instance(filepath)
    
    print(f"Instance loaded:")
    print(f"  Customers: {n_customers}")
    print(f"  Vehicles: {n_vehicles}")
    print(f"  Capacity: {capacity}")
    print(f"  Total demand: {sum(demands)}")
    
    # Get Clarke-Wright initial solution
    initial_routes = None
    cw_cost = None
    
    if use_clarke_wright:
        initial_routes, cw_cost, _ = get_clarke_wright_solution(filepath)
        if initial_routes:
            print(f"\nUsing Clarke-Wright solution (cost={cw_cost}) to restrict search space")
    
    solver = IncrementalCVRPSolver(
        n_customers=n_customers,
        n_vehicles=n_vehicles,
        distances=distances,
        demands=demands,
        capacity=capacity,
        k_neighbors=k_neighbors
    )
    
    routes, cost, time_taken = solver.build_and_solve(
        initial_routes=initial_routes,
        timeout=timeout
    )
    
    # Compare with Clarke-Wright
    if cw_cost is not None and routes:
        if cost < cw_cost:
            print(f"\n*** MaxSAT IMPROVED Clarke-Wright: {cw_cost} -> {cost} ***")
        elif cost == cw_cost:
            print(f"\n*** MaxSAT matched Clarke-Wright: {cost} ***")
        else:
            print(f"\n*** Clarke-Wright was better: {cw_cost} vs {cost} ***")
    
    return routes, cost, time_taken, demands, distances


def main():
    """Example usage."""
    import sys
    
    print("=" * 60)
    print("Incremental MaxSAT Solver for CVRP")
    print("Adapted from RTSS methodology + Clarke-Wright")
    print("=" * 60)
    
    # Check if a VRP file is provided as argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        k_neighbors = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 300
        
        routes, cost, time_taken, demands, distances = solve_vrp_file(
            filepath, k_neighbors=k_neighbors, timeout=timeout
        )
        
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        
        if routes:
            print(f"Cost: {cost}")
            print(f"Time: {time_taken:.2f}s")
            print("Routes:")
            total_cost = 0
            for v, route in sorted(routes.items()):
                demand = sum(demands[c] for c in route)
                route_cost = (distances[0, route[0]] + 
                             sum(distances[route[i], route[i+1]] for i in range(len(route)-1)) +
                             distances[route[-1], 0])
                total_cost += route_cost
                # Convert to 1-indexed for display (matching VRP convention)
                route_display = [c for c in route]  # Keep 0-indexed internally
                print(f"  Route #{v+1}: {' '.join(map(str, route_display))} (demand={demand}, dist={route_cost})")
            print(f"Total computed cost: {total_cost}")
        else:
            print("No feasible solution found!")
        return
    
    # Default: small test instance
    # 5 customers + 1 depot
    n_customers = 5
    n_vehicles = 2
    capacity = 10
    
    # Demands: [depot, c1, c2, c3, c4, c5]
    demands = np.array([0, 3, 4, 3, 4, 3])
    
    # Distance matrix (symmetric)
    distances = np.array([
        [0, 10, 15, 20, 25, 30],
        [10, 0, 12, 18, 22, 28],
        [15, 12, 0, 10, 16, 24],
        [20, 18, 10, 0, 8, 14],
        [25, 22, 16, 8, 0, 10],
        [30, 28, 24, 14, 10, 0]
    ])
    
    # Create solver
    solver = IncrementalCVRPSolver(
        n_customers=n_customers,
        n_vehicles=n_vehicles,
        distances=distances,
        demands=demands,
        capacity=capacity,
        k_neighbors=4
    )
    
    # Solve
    routes, cost, time_taken = solver.build_and_solve(timeout=60)
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    
    if routes:
        print(f"Cost: {cost}")
        print(f"Time: {time_taken:.2f}s")
        print("Routes:")
        for v, route in routes.items():
            demand = sum(demands[c] for c in route)
            route_cost = (distances[0, route[0]] + 
                         sum(distances[route[i], route[i+1]] for i in range(len(route)-1)) +
                         distances[route[-1], 0])
            print(f"  Vehicle {v}: 0 -> {' -> '.join(map(str, route))} -> 0")
            print(f"    Demand: {demand}/{capacity}, Distance: {route_cost}")
    else:
        print("No feasible solution found!")


if __name__ == "__main__":
    main()
