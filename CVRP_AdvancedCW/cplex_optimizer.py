#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
CPLEX-based Optimizers cho CVRP
===============================
Sử dụng IBM CPLEX (thông qua docplex) để giải Single và Pairwise.
Lưu ý: Nếu dùng bản Community, giới hạn là 1000 biến.
"""

import numpy as np
from typing import List, Tuple
from docplex.mp.model import Model

class CplexSingleRouteOptimizer:
    """Tối ưu 1 route (TSP) bằng CPLEX sử dụng mô hình MTZ."""
    
    def __init__(self, customers: List[int], distances: np.ndarray, timeout: float = 5.0):
        self.customers = customers
        self.n = len(customers) + 1  # Tính cả kho (depot)
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
            # Khởi tạo model CPLEX
            mdl = Model(name="TSP_Single_Route")
            mdl.parameters.timelimit = self.timeout
            mdl.parameters.threads = 1

            # 1. BIẾN QUYẾT ĐỊNH
            x = {(i, j): mdl.binary_var(name=f"x_{i}_{j}") 
                 for i in range(self.n) for j in range(self.n) if i != j}
            
            u = {i: mdl.continuous_var(lb=1, ub=self.n-1, name=f"u_{i}") 
                 for i in range(1, self.n)}

            # Hàm mục tiêu
            mdl.minimize(mdl.sum(self.dist[i, j] * x[i, j] 
                                 for i in range(self.n) for j in range(self.n) if i != j))

            # 2. RÀNG BUỘC
            # Vào = 1, Ra = 1
            for j in range(self.n):
                mdl.add_constraint(mdl.sum(x[i, j] for i in range(self.n) if i != j) == 1)
            for i in range(self.n):
                mdl.add_constraint(mdl.sum(x[i, j] for j in range(self.n) if i != j) == 1)

            # Khử chu trình con (MTZ)
            for i in range(1, self.n):
                for j in range(1, self.n):
                    if i != j:
                        mdl.add_constraint(u[i] - u[j] + (self.n - 1) * x[i, j] <= self.n - 2)

            # 3. GIẢI MÔ HÌNH
            sol = mdl.solve(log_output=False)

            if sol:
                route = []
                current = 0
                for _ in range(self.n - 1):
                    for j in range(self.n):
                        if current != j and sol.get_value(x[current, j]) > 0.5:
                            if j != 0: route.append(self.local_to_global[j])
                            current = j
                            break
                return route, int(round(sol.objective_value))
                
        except Exception as e:
            pass # Sẽ rơi xuống phần Fallback bên dưới
            
        # Fallback (Trả về tuyến gốc nếu CPLEX lỗi/timeout không ra nghiệm)
        orig_cost = self.dist[0, 1]
        for i in range(1, self.n - 1):
            orig_cost += self.dist[i, i + 1]
        orig_cost += self.dist[self.n - 1, 0]
        return self.customers, orig_cost


class CplexPairwiseRouteOptimizer:
    """Tối ưu 2 routes bằng CPLEX (Chia tải trọng & Sắp xếp đường đi)."""
    
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
            mdl = Model(name="CVRP_Pairwise")
            mdl.parameters.timelimit = self.timeout
            mdl.parameters.threads = 1

            # 1. BIẾN QUYẾT ĐỊNH
            y = {(i, v): mdl.binary_var(name=f"y_{i}_{v}") for i in range(self.n) for v in range(2)}
            x = {(i, j, v): mdl.binary_var(name=f"x_{i}_{j}_{v}") 
                 for i in range(self.n) for j in range(self.n) if i != j for v in range(2)}
            u = {(i, v): mdl.continuous_var(lb=1, ub=self.n-1, name=f"u_{i}_{v}") 
                 for i in range(1, self.n) for v in range(2)}

            # Hàm mục tiêu
            mdl.minimize(mdl.sum(self.dist[i, j] * x[i, j, v] 
                                 for i in range(self.n) for j in range(self.n) if i != j for v in range(2)))

            # 2. RÀNG BUỘC
            mdl.add_constraint(y[0, 0] == 1)
            mdl.add_constraint(y[0, 1] == 1)

            for i in range(1, self.n):
                mdl.add_constraint(y[i, 0] + y[i, 1] == 1)

            for v in range(2):
                mdl.add_constraint(mdl.sum(self.dem[i] * y[i, v] for i in range(1, self.n)) <= self.capacity)

            for v in range(2):
                for i in range(self.n):
                    mdl.add_constraint(mdl.sum(x[j, i, v] for j in range(self.n) if j != i) == y[i, v])
                    mdl.add_constraint(mdl.sum(x[i, j, v] for j in range(self.n) if j != i) == y[i, v])

            for v in range(2):
                for i in range(1, self.n):
                    for j in range(1, self.n):
                        if i != j:
                            mdl.add_constraint(u[i, v] - u[j, v] + self.n * x[i, j, v] <= self.n - 1)

            # 3. GIẢI MÔ HÌNH
            sol = mdl.solve(log_output=False)

            if sol:
                routes = [[], []]
                for v in range(2):
                    current = 0
                    for _ in range(self.n - 1):
                        found = False
                        for j in range(self.n):
                            if current != j and sol.get_value(x[current, j, v]) > 0.5:
                                if j != 0: routes[v].append(self.local_to_global[j])
                                current = j
                                found = True
                                break
                        if not found or current == 0: break
                
                return routes[0], routes[1], int(round(sol.objective_value)), True
            else:
                return [], [], float('inf'), False
                
        except Exception as e:
            return [], [], float('inf'), False