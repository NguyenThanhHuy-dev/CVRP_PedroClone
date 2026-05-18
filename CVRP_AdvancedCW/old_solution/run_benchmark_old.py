#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import sys
import time
import logging
import csv
import re
import argparse

# Cô lập môi trường
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR) 

from classes.instance import Instance
from classes.clarke_wright import ClarkeWright
from classes.two_opt import TwoOpt
from classes.k_neighbors import KNeighbors
from classes.solver import Solver

INSTANCE_DIR_BASE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "instances"))
LOG_DIR = os.path.join(CURRENT_DIR, "logs")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

GLOBAL_TIMEOUT = 1200.0

# ============================================================================
# TRÍCH XUẤT BKS BẰNG CƠ CHẾ ĐỌC KÉP (DUAL-READ)
# ============================================================================
def extract_bks(vrp_filepath: str, sol_filepath: str) -> float:
    """ 
    Ưu tiên 1: Đọc file .sol tìm dòng 'Cost X' (Chuẩn nhất cho mọi bộ dữ liệu).
    Ưu tiên 2: Nếu không có file .sol, đọc file .vrp quét phần COMMENT.
    """
    # 1. Thử quét file .sol
    if os.path.exists(sol_filepath):
        try:
            with open(sol_filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Tìm dòng cuối cùng hoặc bất cứ đâu có "Cost 1234" hoặc "Cost: 1234"
                match = re.search(r'Cost\s*[:=]?\s*(\d+\.?\d*)', content, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        except Exception as e:
            logging.warning(f"Lỗi khi đọc file {sol_filepath}: {e}")

    # 2. Thử quét file .vrp (Phương án dự phòng)
    if os.path.exists(vrp_filepath):
        try:
            with open(vrp_filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Bắt các mẫu "Optimal value: 123", "Best cost: 123"
                match = re.search(r'(?:Optimal|Best)\s*(?:known\s*)?(?:value|cost)?\s*[:=]?\s*(\d+\.?\d*)', content, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        except Exception as e:
            logging.warning(f"Lỗi khi đọc file {vrp_filepath}: {e}")
            
    return 0.0

# ============================================================================
# CƠ CHẾ GHI ĐÈ KẾT QUẢ TỐT NHẤT (UPSERT LOGIC)
# ============================================================================
def upsert_to_csv(csv_path: str, new_data: dict):
    headers = [
        "Instance", "N", "K", "BKS", "Neighbor_Num", 
        "CW_Cost", "Solver_Cost", "Gap_%", 
        "Total_Time_s", "Status", "Global_Timeout", "Runs"
    ]
    
    existing_data = {}
    if os.path.isfile(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_data[row["Instance"]] = row

    instance = new_data["Instance"]
    
    if instance in existing_data:
        old_row = existing_data[instance]
        runs = int(old_row.get("Runs", 0)) + 1
        
        try:
            old_cost = float(old_row.get("Solver_Cost")) if old_row.get("Solver_Cost") != "" else float('inf')
        except ValueError:
            old_cost = float('inf')
            
        try:
            new_cost_str = new_data.get("Solver_Cost", "")
            new_cost_val = float(new_cost_str) if new_cost_str != "" else float('inf')
        except ValueError:
            new_cost_val = float('inf')

        # Ghi đè nếu chi phí MỚI nhỏ hơn
        if new_cost_val < old_cost:
            logging.info(f"  [CSV] Cập nhật thành công! ({new_cost_val} < {old_cost}). Ghi đè dữ liệu!")
            new_data["Runs"] = runs
            existing_data[instance] = new_data
        else:
            logging.info(f"  [CSV] Nghiệm mới không tốt hơn ({new_cost_val} >= {old_cost}). Bỏ qua, cộng Runs.")
            old_row["Runs"] = runs
            existing_data[instance] = old_row
    else:
        new_data["Runs"] = 1
        existing_data[instance] = new_data

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in existing_data.values():
            writer.writerow(row)

# ============================================================================
# HÀM BENCHMARK CHÍNH
# ============================================================================
def run_benchmark(target_set: str, neighbor_num: int):
    log_filename = os.path.join(LOG_DIR, f"benchmark_{target_set}_old_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )
    
    csv_file = os.path.join(RESULTS_DIR, f"benchmark_{target_set}_ferreira.csv")
    instance_folder = os.path.join(INSTANCE_DIR_BASE, target_set)
    
    if not os.path.exists(instance_folder):
        logging.error(f"❌ Thư mục không tồn tại: {instance_folder}")
        return

    # Chỉ quét các file .vrp (bỏ qua file .sol) để tránh chạy lặp 2 lần
    files = sorted([f for f in os.listdir(instance_folder) if f.endswith(".vrp")])
    logging.info(f"=== BẮT ĐẦU BENCHMARK OLD SOTA (FERREIRA) | TẬP {target_set} ===")

    for filename in files:
        vrp_filepath = os.path.join(instance_folder, filename)
        name = filename.replace(".vrp", "")
        sol_filepath = os.path.join(instance_folder, f"{name}.sol")
        
        match = re.search(r'-n(\d+)-k(\d+)', name)
        n_nodes = int(match.group(1)) if match else 0
        k_vehicles = int(match.group(2)) if match else 0
        
        # Đọc BKS qua hàm mới
        bks = extract_bks(vrp_filepath, sol_filepath)

        logging.info("-" * 60)
        logging.info(f"Đang xử lý: {name} (N={n_nodes}, K={k_vehicles}) | BKS: {bks} | Neighbors: {neighbor_num}")
        
        start_total = time.time()
        cw_cost = float('inf')
        solver_cost = float('inf')
        gap = 0.0
        status = "FAILED"
        is_timeout = False

        try:
            cvrp = Instance(vrp_filepath).load()

            _, routes = ClarkeWright.run(cvrp, k_vehicles)
            _, routes = TwoOpt.run(routes)
            cw_cost = sum(r.cost for r in routes.values())
            
            _, matrices = KNeighbors.run(cvrp, neighbor_num, routes)

            # [PATCH 1]: Tính toán thời gian trừ hao trước khi gọi bộ giải chính xác
            elapsed_before_solver = time.time() - start_total
            # Cắt bớt 1 giây buffer để an toàn cho độ trễ hệ điều hành
            remaining_time = int(GLOBAL_TIMEOUT - elapsed_before_solver - 1) 
            
            if remaining_time <= 0:
                solver_cost = float('inf')
            else:
                # Truyền quỹ thời gian thực tế còn lại cho clasp
                solver_time, solver_cost, _ = Solver.run(cvrp, matrices, use_lima=False, timeout=remaining_time)
            
            total_time = time.time() - start_total
            # Ép cờ timeout nếu hệ thống chạy sát hoặc vượt ngưỡng (trừ hao 2 giây)
            is_timeout = (total_time >= GLOBAL_TIMEOUT - 2.0)

            # [PATCH 2]: Logic State Consistency cho Gap và Status 
            gap = "" # Bỏ giá trị mặc định 0.0

            if solver_cost == 0:
                # Đồ thị đứt gãy do K-Neighbors (Infeasible ngầm)
                status = "INFEASIBLE"
                gap = -100.0
            elif solver_cost < float('inf'):
                # Tìm được nghiệm
                status = "TIMEOUT_WITH_SOL" if is_timeout else "SUCCESS"
                if bks > 0:
                    gap = ((solver_cost - bks) / bks) * 100
            else:
                # Hoàn toàn không tìm được nghiệm
                status = "TIMEOUT_NO_SOL" if is_timeout else "FAILED"

            gap_str = f"{gap:.2f}%" if isinstance(gap, float) else "N/A"
            logging.info(f"  [Kết quả] Cost = {solver_cost:.2f} | Gap = {gap_str} | Time: {total_time:.1f}s | Trạng thái: {status}")
        except Exception as e:
            total_time = time.time() - start_total
            is_timeout = (total_time >= GLOBAL_TIMEOUT)
            error_msg = str(e)
            
            if "Cannot reduce the number of routes" in error_msg:
                status = "CW_UNSAT" 
            elif is_timeout:
                status = "TIMEOUT_NO_SOL"
            else:
                status = "UNSAT" if "Cannot find a solution" in error_msg else "ERROR"
                
            logging.error(f"  ❌ [LỖI] {error_msg} - Trạng thái: {status}")

        row_data = {
            "Instance": name,
            "N": n_nodes, "K": k_vehicles, "BKS": bks,
            "Neighbor_Num": neighbor_num,
            "CW_Cost": int(cw_cost) if cw_cost != float('inf') else "",
            "Solver_Cost": int(solver_cost) if solver_cost != float('inf') else "",
            "Gap_%": round(gap, 2) if solver_cost != float('inf') else "",
            "Total_Time_s": round(total_time, 2),
            "Status": status,
            "Global_Timeout": is_timeout
        }
        upsert_to_csv(csv_file, row_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, default="X")
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()
    run_benchmark(target_set=args.set, neighbor_num=args.neighbors)