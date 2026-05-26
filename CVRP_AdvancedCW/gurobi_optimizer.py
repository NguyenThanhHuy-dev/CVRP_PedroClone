import numpy as np
from typing import List, Tuple
import gurobipy as gp
from gurobipy import GRB

class GurobiSingleRouteOptimizer:
    """Tối ưu 1 route (bài toán TSP) bằng Gurobi sử dụng mô hình MTZ."""
    
    def __init__(self, customers: List[int], distances: np.ndarray, timeout: float = 5.0):
        self.customers = customers
        self.n = len(customers) + 1
        self.timeout = timeout
        
        self.local_to_global = {0: 0}
        for i, c in enumerate(customers):
            self.local_to_global[i + 1] = c

        self.dist = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                gi, gj = self.local_to_global[i], self.local_to_global[j]
                self.dist[i, j] = distances[gi, gj]

    def optimize(self) -> Tuple[List[int], int]:
        try:
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            
            m = gp.Model("TSP_Single_Route", env=env)
            m.setParam('TimeLimit', self.timeout)
            m.setParam('Threads', 1)

            x = {}
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        x[i, j] = m.addVar(vtype=GRB.BINARY, obj=self.dist[i, j])
            
            u = {}
            for i in range(1, self.n):
                u[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=self.n-1)

            for j in range(self.n):
                m.addConstr(gp.quicksum(x[i, j] for i in range(self.n) if i != j) == 1)
            for i in range(self.n):
                m.addConstr(gp.quicksum(x[i, j] for j in range(self.n) if i != j) == 1)

            for i in range(1, self.n):
                for j in range(1, self.n):
                    if i != j:
                        m.addConstr(u[i] - u[j] + (self.n - 1) * x[i, j] <= self.n - 2)

            m.optimize()

            if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
                route = []
                current = 0
                for _ in range(self.n - 1):
                    for j in range(self.n):
                        if current != j and x[current, j].X > 0.5:
                            if j != 0: route.append(self.local_to_global[j])
                            current = j
                            break
                return route, int(round(m.ObjVal))
                
        except Exception:
            pass
            
        orig_cost = self.dist[0, 1]
        for i in range(1, self.n - 1):
            orig_cost += self.dist[i, i + 1]
        orig_cost += self.dist[self.n - 1, 0]
        return self.customers, orig_cost


class GurobiPairwiseRouteOptimizer:
    """Tối ưu 2 routes cùng lúc bằng Gurobi (Chia tải trọng & Sắp xếp đường đi)."""
    
    def __init__(self, c1: List[int], c2: List[int], distances: np.ndarray, demands: np.ndarray, capacity: int, timeout: float = 10.0):
        self.customers = c1 + c2
        self.n = len(self.customers) + 1
        self.timeout = timeout
        self.capacity = capacity
        
        self.local_to_global = {0: 0}
        for i, c in enumerate(self.customers):
            self.local_to_global[i + 1] = c

        self.dist = np.zeros((self.n, self.n), dtype=int)
        self.dem = np.zeros(self.n, dtype=int)
        
        for i in range(self.n):
            gi = self.local_to_global[i]
            if i > 0:
                self.dem[i] = demands[gi]
            for j in range(self.n):
                gj = self.local_to_global[j]
                self.dist[i, j] = distances[gi, gj]

    def optimize(self) -> Tuple[List[int], List[int], int, bool]:
        try:
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            
            m = gp.Model("CVRP_Pairwise", env=env)
            m.setParam('TimeLimit', self.timeout)
            m.setParam('Threads', 1)

            # y[i, v]: =1 nếu khách hàng i đi xe v
            y = m.addVars(self.n, 2, vtype=GRB.BINARY, name="y")
            # x[i, j, v]: =1 nếu xe v đi từ i đến j
            x = m.addVars(self.n, self.n, 2, vtype=GRB.BINARY, name="x")
            
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        for v in range(2): x[i, j, v].Obj = self.dist[i, j]
                    else:
                        for v in range(2): x[i, j, v].ub = 0 

            u = m.addVars(self.n, 2, vtype=GRB.CONTINUOUS, lb=1, ub=self.n-1, name="u")

            m.addConstr(y[0, 0] == 1)
            m.addConstr(y[0, 1] == 1)

            for i in range(1, self.n):
                m.addConstr(y[i, 0] + y[i, 1] == 1)

            for v in range(2):
                m.addConstr(gp.quicksum(self.dem[i] * y[i, v] for i in range(1, self.n)) <= self.capacity)

            for v in range(2):
                for i in range(self.n):
                    m.addConstr(gp.quicksum(x[j, i, v] for j in range(self.n) if j != i) == y[i, v])
                    m.addConstr(gp.quicksum(x[i, j, v] for j in range(self.n) if j != i) == y[i, v])

            for v in range(2):
                for i in range(1, self.n):
                    for j in range(1, self.n):
                        if i != j:
                            m.addConstr(u[i, v] - u[j, v] + self.n * x[i, j, v] <= self.n - 1)

            m.optimize()

            if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
                routes = [[], []]
                for v in range(2):
                    current = 0
                    for _ in range(self.n - 1):
                        found = False
                        for j in range(self.n):
                            if current != j and x[current, j, v].X > 0.5:
                                if j != 0: routes[v].append(self.local_to_global[j])
                                current = j
                                found = True
                                break
                        if not found or current == 0: break
                
                return routes[0], routes[1], int(round(m.ObjVal)), True
            else:
                return [], [], float('inf'), False
                
        except Exception:
            return [], [], float('inf'), False