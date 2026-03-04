import numpy as np
from typing import List, Tuple
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

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
