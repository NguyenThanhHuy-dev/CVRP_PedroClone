# classes/alns.py
import random
import numpy as np
from typing import Dict, List, Tuple, Set

class ALNS:
    """
    Tập hợp các toán tử Phá hủy (Destroy), Sửa chữa (Repair) và GLS 
    dành cho Chặng 3 (ALNS + GLS) của hệ thống tối ưu.
    """

    # =========================================================================
    # 1. CƠ CHẾ GLS (GUIDED LOCAL SEARCH)
    # =========================================================================
    @staticmethod
    def get_penalized_cost(i: int, j: int, distances: np.ndarray, penalties: np.ndarray, lambda_val: float) -> float:
        """ Tính chi phí mở rộng: c~(S) = c(S) + lambda * p_ij """
        return float(distances[i, j]) + lambda_val * float(penalties[i, j])

    @staticmethod
    def update_gls_penalties(routes: Dict[int, List[int]], distances: np.ndarray, penalties: np.ndarray):
        """ 
        Tính độ hữu ích u_ij = c_ij / (1 + p_ij) 
        và tăng phạt p_ij cho các cạnh có độ hữu ích cao nhất (edges dài nhất/ngoan cố nhất).
        """
        max_util = -1.0
        edges_to_penalize = []

        for route in routes.values():
            if not route: continue
            full_route = [0] + route + [0] # Bao gồm cả Depot
            for i in range(len(full_route) - 1):
                u, v = full_route[i], full_route[i+1]
                util = distances[u, v] / (1.0 + penalties[u, v])
                
                if util > max_util + 1e-5:
                    max_util = util
                    edges_to_penalize = [(u, v)]
                elif abs(util - max_util) <= 1e-5:
                    edges_to_penalize.append((u, v))

        # Tăng phạt cho các cạnh lọt vào danh sách đen (Đồ thị vô hướng nên phạt cả 2 chiều)
        for u, v in edges_to_penalize:
            penalties[u, v] += 1
            penalties[v, u] += 1

    # =========================================================================
    # 2. CÁC TOÁN TỬ PHÁ HỦY (DESTROY OPERATORS)
    # =========================================================================
    @staticmethod
    def destroy_random(routes: Dict[int, List[int]], q: int) -> Tuple[Dict[int, List[int]], List[int]]:
        """ Phá hủy Ngẫu nhiên: Chọn ngẫu nhiên q khách hàng để rút khỏi tuyến """
        new_routes = {k: list(v) for k, v in routes.items()}
        all_customers = [c for r in new_routes.values() for c in r]
        
        if q >= len(all_customers):
            q = len(all_customers)
            
        removed = random.sample(all_customers, q)
        
        for r_id in new_routes:
            new_routes[r_id] = [c for c in new_routes[r_id] if c not in removed]
            
        return new_routes, removed

    @staticmethod
    def destroy_worst(routes: Dict[int, List[int]], distances: np.ndarray, q: int) -> Tuple[Dict[int, List[int]], List[int]]:
        """ Phá hủy Tồi nhất: Rút q khách hàng làm tốn nhiều chi phí quãng đường nhất """
        new_routes = {k: list(v) for k, v in routes.items()}
        savings = [] # Lưu lượng cost tiết kiệm được nếu tháo khách hàng i ra
        
        for r_id, route in new_routes.items():
            for i, cust in enumerate(route):
                prev_c = 0 if i == 0 else route[i-1]
                next_c = 0 if i == len(route)-1 else route[i+1]
                
                # Chi phí tiết kiệm = (Đường đi cũ) - (Đường đi nếu nối thẳng bỏ qua cust)
                save = distances[prev_c, cust] + distances[cust, next_c] - distances[prev_c, next_c]
                savings.append((save, r_id, cust))
                
        # Sắp xếp giảm dần theo lượng tiết kiệm (Đỉnh tốn kém nhất nằm đầu)
        savings.sort(key=lambda x: x[0], reverse=True)
        
        # Thêm một chút ngẫu nhiên để tránh việc cứ xóa đi xóa lại 1 tập đỉnh cố định
        pool_size = min(len(savings), int(q * 1.5))
        selected = random.sample(savings[:pool_size], q)
        removed = [x[2] for x in selected]
        
        for r_id in new_routes:
            new_routes[r_id] = [c for c in new_routes[r_id] if c not in removed]
            
        return new_routes, removed

    @staticmethod
    def destroy_related(routes: Dict[int, List[int]], distances: np.ndarray, q: int) -> Tuple[Dict[int, List[int]], List[int]]:
        """ Phá hủy Tương đồng: Chọn 1 đỉnh ngẫu nhiên, sau đó rút q-1 đỉnh GẦN nó nhất """
        new_routes = {k: list(v) for k, v in routes.items()}
        all_customers = [c for r in new_routes.values() for c in r]
        
        if q >= len(all_customers):
            return ALNS.destroy_random(routes, q)
            
        seed_customer = random.choice(all_customers)
        all_customers.remove(seed_customer)
        
        # Sắp xếp các khách hàng còn lại theo khoảng cách tới seed_customer
        all_customers.sort(key=lambda c: distances[seed_customer, c])
        removed = [seed_customer] + all_customers[:q-1]
        
        for r_id in new_routes:
            new_routes[r_id] = [c for c in new_routes[r_id] if c not in removed]
            
        return new_routes, removed

    # =========================================================================
    # 3. TOÁN TỬ SỬA CHỮA (REPAIR OPERATOR)
    # =========================================================================
    @staticmethod
    def repair_regret_2(routes: Dict[int, List[int]], removed: List[int], distances: np.ndarray, 
                       demands: np.ndarray, capacity: int, penalties: np.ndarray, lambda_val: float) -> Dict[int, List[int]]:
        """ 
        Sửa chữa Regret-2: Chèn lại khách hàng bằng cách so sánh vị trí Tốt nhất và Tốt thứ hai.
        Lưu ý: Dùng Chi phí Mở rộng c~(S) (Penalized Cost) để đưa ra quyết định!
        """
        repaired_routes = {k: list(v) for k, v in routes.items()}
        unassigned = list(removed)
        
        while unassigned:
            best_regret = -float('inf')
            best_customer = -1
            best_insertion = None # Tuple: (r_id, position)
            
            for cust in unassigned:
                # Tìm 2 vị trí chèn tốt nhất cho 'cust' trên toàn bộ các xe
                insertion_costs = []
                
                for r_id, route in repaired_routes.items():
                    # Bỏ qua xe nếu vi phạm tải trọng
                    if sum(demands[c] for c in route) + demands[cust] > capacity:
                        continue
                        
                    for i in range(len(route) + 1):
                        prev_c = 0 if i == 0 else route[i-1]
                        next_c = 0 if i == len(route) else route[i]
                        
                        # Tính phí chèn gia tăng BẰNG PENALIZED COST
                        cost_increase = (
                            ALNS.get_penalized_cost(prev_c, cust, distances, penalties, lambda_val) +
                            ALNS.get_penalized_cost(cust, next_c, distances, penalties, lambda_val) -
                            ALNS.get_penalized_cost(prev_c, next_c, distances, penalties, lambda_val)
                        )
                        insertion_costs.append((cost_increase, r_id, i))
                        
                # Nếu khách hàng này KHÔNG THỂ chèn vào đâu (do sức chứa kẹt cứng)
                # Tạm thời ép chèn vào xe rỗng nhất để tránh chết code (Local Search sẽ dọn dẹp sau)
                if not insertion_costs:
                    # LỖI KẸT TẢI TRỌNG: Trả về None để hệ thống từ chối hoàn toàn vòng lặp này!
                    return None
                    
                insertion_costs.sort(key=lambda x: x[0])
                
                # Chi phí chèn Tốt nhất (delta f1) và Tốt thứ 2 (delta f2)
                f1 = insertion_costs[0][0]
                f2 = insertion_costs[1][0] if len(insertion_costs) > 1 else f1 + 1000.0
                
                regret_value = f2 - f1
                
                if regret_value > best_regret:
                    best_regret = regret_value
                    best_customer = cust
                    best_insertion = (insertion_costs[0][1], insertion_costs[0][2])
                    
            # Thực hiện chèn khách hàng có Regret lớn nhất vào vị trí Tốt nhất của nó
            target_r_id, target_pos = best_insertion
            repaired_routes[target_r_id].insert(target_pos, best_customer)
            unassigned.remove(best_customer)
            
        return repaired_routes