import numpy as np
from typing import List, Tuple
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

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
