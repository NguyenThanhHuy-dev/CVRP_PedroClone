import numpy as np
from typing import List, Tuple, Dict, Optional
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from src.common.utils import calculate_route_cost

class SingleRouteOptimizer:
    """Tối ưu thứ tự các điểm trong một route (TSP nhỏ)."""
    def __init__(self, customers: List[int], distances: np.ndarray):
        self.customers = customers
        self.n = len(customers) + 1
        self.distances = distances
        self.local_to_global = {0: 0}
        for i, c in enumerate(customers): self.local_to_global[i + 1] = c
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
        self.var_id += 1; return self.var_id

    def gen_model(self):
        # Variables
        for i in range(self.n):
            for j in range(self.n):
                if i != j: self.conNet[i][j] = self.new_var_id()
        for i in range(1, self.n):
            for j in range(i + 1, self.n):
                var = self.new_var_id()
                self.rchNet[i][j] = var; self.rchNet[j][i] = -var
        
        # Soft clauses
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.local_dist[i, j] > 0:
                    self.wcnf.append([-self.conNet[i][j]], weight=self.local_dist[i, j])
        
        # Hard clauses (Simplified TSP)
        for i in range(self.n):
            out_v = [self.conNet[i][j] for j in range(self.n) if i!=j]
            in_v = [self.conNet[j][i] for j in range(self.n) if i!=j]
            self.wcnf.append(out_v); self.wcnf.append(in_v) # At least one
            for x in range(len(out_v)):
                for y in range(x+1, len(out_v)): self.wcnf.append([-out_v[x], -out_v[y]])
            for x in range(len(in_v)):
                for y in range(x+1, len(in_v)): self.wcnf.append([-in_v[x], -in_v[y]])
        
        # Subtour elimination
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i!=j and self.rchNet[i][j]!=0:
                    self.wcnf.append([-self.conNet[i][j], self.rchNet[i][j]])
        
        for a in range(1, self.n):
            for b in range(1, self.n):
                if a==b: continue
                for c in range(1, self.n):
                    if c==a or c==b: continue
                    if self.rchNet[a][b] and self.rchNet[b][c]:
                        self.wcnf.append([-self.rchNet[a][b], -self.rchNet[b][c], self.rchNet[a][c]])
                        self.wcnf.append([-self.rchNet[a][b], -self.rchNet[b][c], -self.conNet[a][c]])

    def decode(self, model):
        pos = set(v for v in model if v > 0)
        route = []
        curr, visited = 0, {0}
        for _ in range(self.n - 1):
            for j in range(self.n):
                if j not in visited and self.conNet[curr][j] in pos:
                    if self.local_to_global[j] != 0: route.append(self.local_to_global[j])
                    visited.add(j); curr = j; break
        return route
    
    def calculate_initial_cost(self):
        # Tính cost của route ban đầu để làm fallback
        c = 0
        local_route = [0] + list(range(1, self.n)) + [0] 
        for k in range(len(local_route)-1):
            c += self.local_dist[local_route[k], local_route[k+1]]
        return c

    def optimize(self):
        fallback_cost = self.calculate_initial_cost()
        self.gen_model()
        with RC2(self.wcnf, verbose=0) as solver:
            model = solver.compute()
            if model: 
                return self.decode(model), solver.cost
            return self.customers, fallback_cost

class PairwiseRouteOptimizer:
    """Tối ưu 2 routes: Assignment + Ordering."""
    def __init__(self, c1: List[int], c2: List[int], dists: np.ndarray, dems: np.ndarray, cap: int):
        self.c1, self.c2 = c1, c2
        self.all_c = c1 + c2
        self.n = len(self.all_c) + 1
        self.dists, self.dems, self.cap = dists, dems, cap
        self.l2g = {0:0}
        for i, c in enumerate(self.all_c): self.l2g[i+1] = c
        self.ldist = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                self.ldist[i,j] = dists[self.l2g[i], self.l2g[j]]
        self.wcnf = WCNF()
        self.var_id = 0
        self.con = [[[0]*self.n for _ in range(self.n)] for _ in range(2)]
        self.rch = [[[0]*self.n for _ in range(self.n)] for _ in range(2)]
        self.assign = [0]*self.n

    def nid(self) -> int: self.var_id+=1; return self.var_id

    def optimize(self):
        # Gen vars
        for i in range(1, self.n): self.assign[i] = self.nid()
        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i!=j: self.con[v][i][j] = self.nid()
        for v in range(2):
            for i in range(1, self.n):
                for j in range(i+1, self.n):
                    x = self.nid(); self.rch[v][i][j]=x; self.rch[v][j][i]=-x
        
        # Soft clauses
        for v in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if i!=j and self.ldist[i,j]>0:
                        self.wcnf.append([-self.con[v][i][j]], weight=self.ldist[i,j])
        
        # Hard clauses
        for i in range(1, self.n):
            for j in range(self.n):
                if i!=j:
                    self.wcnf.append([self.assign[i], -self.con[1][i][j]])
                    self.wcnf.append([self.assign[i], -self.con[1][j][i]])
                    self.wcnf.append([-self.assign[i], -self.con[0][i][j]])
                    self.wcnf.append([-self.assign[i], -self.con[0][j][i]])

        for c in range(1, self.n):
            in_v = [self.con[v][i][c] for v in range(2) for i in range(self.n) if i!=c]
            out_v = [self.con[v][c][j] for v in range(2) for j in range(self.n) if j!=c]
            self.wcnf.append(in_v); self.wcnf.append(out_v)
            for i in range(len(in_v)):
                for j in range(i+1, len(in_v)): self.wcnf.append([-in_v[i], -in_v[j]])
            for i in range(len(out_v)):
                for j in range(i+1, len(out_v)): self.wcnf.append([-out_v[i], -out_v[j]])

        for v in range(2):
            d_out = [self.con[v][0][j] for j in range(1, self.n)]
            d_in = [self.con[v][i][0] for i in range(1, self.n)]
            self.wcnf.append(d_out); self.wcnf.append(d_in)
            for i in range(len(d_out)):
                for j in range(i+1, len(d_out)): self.wcnf.append([-d_out[i], -d_out[j]])
            for i in range(len(d_in)):
                for j in range(i+1, len(d_in)): self.wcnf.append([-d_in[i], -d_in[j]])

        for v in range(2):
            for i in range(1, self.n):
                for j in range(1, self.n):
                    if i!=j and self.rch[v][i][j]!=0:
                        self.wcnf.append([-self.con[v][i][j], self.rch[v][i][j]])
            for a in range(1, self.n):
                for b in range(1, self.n):
                    if a==b: continue
                    for c in range(1, self.n):
                        if c==a or c==b: continue
                        if self.rch[v][a][b] and self.rch[v][b][c]:
                            self.wcnf.append([-self.rch[v][a][b], -self.rch[v][b][c], self.rch[v][a][c]])
                            self.wcnf.append([-self.rch[v][a][b], -self.rch[v][b][c], -self.con[v][a][c]])

        with RC2(self.wcnf, verbose=0) as solver:
            model = solver.compute()
            if model:
                pos = set(m for m in model if m>0)
                routes = [[], []]
                for v in range(2):
                    curr, visited = 0, {0}
                    for _ in range(self.n):
                        for j in range(self.n):
                            if j not in visited and self.con[v][curr][j] in pos:
                                if self.l2g[j]!=0: routes[v].append(self.l2g[j])
                                visited.add(j); curr=j; break
                
                if sum(self.dems[c] for c in routes[0]) <= self.cap and \
                   sum(self.dems[c] for c in routes[1]) <= self.cap:
                    return routes[0], routes[1], solver.cost
        
        return self.c1, self.c2, float('inf')