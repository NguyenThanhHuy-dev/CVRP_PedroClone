#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Gurobi-based Optimizers cho CVRP
================================
"""

import numpy as np
from typing import List, Tuple
import gurobipy as gp
from gurobipy import GRB

class GurobiSingleRouteOptimizer:
    """Tối ưu 1 route (bài toán TSP) bằng Gurobi sử dụng mô hình MTZ."""
    
    def __init__(self, customers: List[int], distances: np.ndarray, timeout: float = 5.0):
        self.customers = customers
        self.n = len(customers) + 1  # Tính cả kho (depot)
        self.timeout = timeout
        
        # Ánh xạ index cục bộ (0..n-1) sang index toàn cục
        self.local_to_global = {0: 0}
        for i, c in enumerate(customers):
            self.local_to_global[i + 1] = c

        self.dist = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                gi = self.local_to_global[i]
                gj = self.local_to_global[j]
                self.dist[i, j] = distances[gi, gj]

    def optimize(self) -> Tuple[List[int], int]:
        try:
            # Khởi tạo môi trường Gurobi im lặng
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

            # Degree = 1
            for j in range(self.n):
                m.addConstr(gp.quicksum(x[i, j] for i in range(self.n) if i != j) == 1)
            for i in range(self.n):
                m.addConstr(gp.quicksum(x[i, j] for j in range(self.n) if i != j) == 1)

            # Subtour elimination (MTZ)
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
            pass # Fallback bên dưới
            
        # Fallback: Trả về route cũ
        orig_cost = self.dist[0, 1]
        for i in range(1, self.n - 1):
            orig_cost += self.dist[i, i + 1]
        orig_cost += self.dist[self.n - 1, 0]
        return self.customers, orig_cost