import argparse
import yaml
import os
import sys
import time
import csv
from datetime import datetime

# Thêm đường dẫn src vào hệ thống
sys.path.append(os.getcwd())

from src.solvers.solver_v1 import SolverV1
from src.solvers.solver_v2 import SolverV2

def save_result_to_csv(output_dir, instance_name, algo, cost, time_elapsed, config_path):
    """Hàm lưu kết quả vào file CSV"""
    # 1. Tạo thư mục results nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_file = os.path.join(output_dir, "experiment_log.csv")
    file_exists = os.path.isfile(csv_file)
    
    # 2. Ghi dữ liệu
    with open(csv_file, mode='a', newline='') as f:
        fieldnames = ['Date', 'Instance', 'Algorithm', 'Cost', 'Time(s)', 'Config']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Ghi tiêu đề nếu file mới tạo
            
        writer.writerow({
            'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Instance': instance_name,
            'Algorithm': algo,
            'Cost': cost,
            'Time(s)': f"{time_elapsed:.2f}",
            'Config': config_path
        })
    print(f"  [Saved] Kết quả đã được lưu vào: {csv_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, required=True, help="Đường dẫn file .vrp")
    parser.add_argument('--algo', type=str, default='v2', choices=['v1', 'v2'], help="Chọn thuật toán")
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f: config = yaml.safe_load(f)
    
    # Lấy tham số output directory từ config (mặc định là 'results')
    output_dir = config.get('experiment', {}).get('output_dir', 'results')

    # Chọn Solver
    if args.algo == 'v1':
        solver = SolverV1(config.get('algorithm', {}))
    else:
        solver = SolverV2(config.get('algorithm', {}))

    # Chạy
    print(f"--- STARTING {args.algo.upper()} on {os.path.basename(args.instance)} ---")
    start = time.time()
    
    try:
        routes, cost = solver.solve(args.instance)
        elapsed = time.time() - start
        
        # In ra màn hình
        print("\n" + "="*40)
        print(f"FINAL RESULT ({args.algo.upper()}): {cost}")
        print(f"TIME: {elapsed:.2f}s")
        print("="*40)
        
        # Lưu vào file
        save_result_to_csv(output_dir, os.path.basename(args.instance), args.algo, cost, elapsed, args.config)

    except Exception as e:
        print(f"\n[ERROR] Chạy thất bại: {e}")

if __name__ == "__main__":
    main()