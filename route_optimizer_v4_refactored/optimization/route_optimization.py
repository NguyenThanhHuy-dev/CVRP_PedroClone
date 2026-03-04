import time
from itertools import combinations
import numpy as np
from typing import List, Dict, Tuple

from core.cost_utils import calculate_route_cost
from sat.pairwise_route import PairwiseRouteOptimizer
from sat.single_route import SingleRouteOptimizer

def pairwise_optimize(routes: Dict[int, List[int]], distances: np.ndarray,
                      demands: np.ndarray, capacity: int, 
                      max_iterations: int = 2) -> Tuple[Dict[int, List[int]], int]:
    """
    Tối ưu bằng cách gộp từng cặp routes và tối ưu lại.
    [FIXED] Giảm ngưỡng và số vòng lặp để tránh treo máy.
    """
    print("\n=== Pair-wise Route Optimization ===")
    
    improved = True
    iteration = 0
    global_start = time.time()
    
    # Giới hạn tổng thời gian cho bước này là 3 phút
    MAX_STEP_TIME = 180 
    
    while improved and iteration < max_iterations:
        if time.time() - global_start > MAX_STEP_TIME:
            print("  [TIMEOUT] Pair-wise step took too long. Skipping remaining iterations.")
            break

        improved = False
        iteration += 1
        print(f"\n  Iteration {iteration}:")
        
        route_ids = list(routes.keys())
        # Sắp xếp để ưu tiên các route nhỏ trước
        route_ids.sort(key=lambda rid: len(routes[rid]))
        
        for i, j in combinations(route_ids, 2):
            if time.time() - global_start > MAX_STEP_TIME: break
            if not routes[i] or not routes[j]: continue
            
            # [CRITICAL FIX] Giảm ngưỡng từ 10 xuống 8 để an toàn
            # MaxSAT với N>9 thường rất rủi ro về thời gian
            if len(routes[i]) + len(routes[j]) > 8:
                continue
            
            # Current cost of pair
            cost_i = calculate_route_cost(routes[i], distances)
            cost_j = calculate_route_cost(routes[j], distances)
            current_cost = cost_i + cost_j
            
            # In dấu chấm để biết chương trình vẫn đang chạy
            print(".", end="", flush=True)
            
            # Try to optimize this pair
            optimizer = PairwiseRouteOptimizer(
                routes[i], routes[j], distances, demands, capacity
            )
            # Vì hàm optimize bên trong chưa implement timeout thực sự cho RC2,
            # việc giảm ngưỡng xuống 8 ở trên là biện pháp bảo vệ chính.
            new_route_i, new_route_j, new_cost = optimizer.optimize()
            
            if new_cost < current_cost:
                print(f"\n    Routes {i},{j} (Sz {len(routes[i])}+{len(routes[j])}): {current_cost} -> {new_cost} (IMPROVED -{current_cost - new_cost})")
                routes[i] = new_route_i
                routes[j] = new_route_j
                improved = True
                
    total_cost = sum(calculate_route_cost(r, distances) for r in routes.values())
    print(f"\n  Pair-wise done. Final cost: {total_cost}")
    return routes, total_cost


def optimize_all_routes(routes, distances, demands, capacity):
    print("\n=== MaxSAT Single Route Optimization ===")
    opt_routes = {}
    total = 0
    
    # Duyệt qua từng tuyến xe (route)
    for v, r in routes.items():
        # 1. Bỏ qua tuyến quá dài (để tránh treo máy do MaxSAT)
        if len(r) > 11: 
            opt_routes[v] = r
            total += calculate_route_cost(r, distances)
            print("!", end="", flush=True) 
            continue
        
        # 2. [FIX QUAN TRỌNG] Bỏ qua tuyến quá ngắn (0 hoặc 1 khách)
        # Tuyến 1 khách luôn là tối ưu (Depot -> Khách -> Depot), không cần chạy SAT
        if len(r) <= 1:
            opt_routes[v] = r
            total += calculate_route_cost(r, distances)
            # In dấu "-" để báo hiệu là skip do quá ngắn
            print("-", end="", flush=True)
            continue

        print(".", end="", flush=True)
        
        # 3. [DEBUG] Bọc trong Try-Except để không crash chương trình
        try:
            # Gọi bộ giải Max-SAT cho bài toán con
            opt = SingleRouteOptimizer(r, distances)
            new_r, cost = opt.optimize()
            
            # Kiểm tra lại tải trọng (Safety check)
            if sum(demands[c] for c in new_r) <= capacity:
                opt_routes[v] = new_r
                total += cost
            else:
                opt_routes[v] = r
                total += calculate_route_cost(r, distances)
                
        except Exception as e:
            # Nếu gặp lỗi (như lỗi numpy index), in log và giữ nguyên route cũ
            # Không dừng chương trình!
            print(f"\n[⚠️ SKIP Route {v}] Lỗi: {e}. Giữ nguyên route cũ.")
            opt_routes[v] = r
            total += calculate_route_cost(r, distances)
            
    print("\n  Single Route Optimization done.")
    return opt_routes, total
