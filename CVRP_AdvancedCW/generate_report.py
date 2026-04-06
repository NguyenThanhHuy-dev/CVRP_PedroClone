#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Yêu cầu cài đặt thư viện: 
# pip install pandas matplotlib seaborn tabulate

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Tìm đường dẫn file (hỗ trợ cả ở thư mục gốc và thư mục results/)
def get_file_path(filename):
    if os.path.exists(filename): 
        return filename
    if os.path.exists(f"results/{filename}"): 
        return f"results/{filename}"
    return None

files = {
    "PySAT": "benchmark_B_pysat.csv",
    "Gurobi": "benchmark_B_gurobi.csv",
    "CPLEX": "benchmark_B_cplex.csv"
}

dfs = []
for method, filename in files.items():
    path = get_file_path(filename)
    if path:
        try:
            df = pd.read_csv(path)
            if not df.empty:
                df['Method'] = method
                dfs.append(df)
        except Exception as e:
            print(f"[LỖI] Không thể đọc file {path}: {e}")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy file của {method} ({filename})")

if not dfs:
    print("[LỖI] Không có file CSV nào được nạp. Vui lòng kiểm tra lại đường dẫn!")
    exit()

# Gộp dữ liệu từ 3 phương pháp
data = pd.concat(dfs, ignore_index=True)

# 2. Sử dụng đúng tên cột từ file CSV 
time_col = 'Last_Time(s)'
cost_col = 'Best_Cost'
bks_col = 'BKS'
last_gap_col = 'Last_Gap(%)' # Cột bổ sung theo yêu cầu

print(f"[INFO] Bắt đầu phân tích báo cáo nâng cao (4 Biểu đồ)...")

# Đảm bảo các cột định dạng số
data[cost_col] = pd.to_numeric(data[cost_col], errors='coerce')
data[bks_col] = pd.to_numeric(data[bks_col], errors='coerce')
data[time_col] = pd.to_numeric(data[time_col], errors='coerce')
data[last_gap_col] = pd.to_numeric(data[last_gap_col], errors='coerce')

# Tính Gap (%) tổng quát và xác định Optimal (Cho phép sai số làm tròn 0.001)
data['Gap(%)'] = ((data[cost_col] - data[bks_col]) / data[bks_col]) * 100
data['Is_Optimal'] = data['Gap(%)'] <= 0.001

# Sắp xếp lại dữ liệu theo N và K để biểu đồ đường hiển thị chuẩn xác (từ bài dễ đến bài khó)
data['N_val'] = data['Instance'].str.extract(r'-n(\d+)').astype(float)
data.sort_values(by=['N_val', 'Instance'], inplace=True)

# ==========================================
# PHẦN 1: XUẤT BẢNG THỐNG KÊ (Markdown)
# ==========================================
print("\n" + "="*50)
print(" BẢNG TỔNG HỢP HIỆU NĂNG SO SÁNH 3 PHƯƠNG PHÁP")
print("="*50)

summary = data.groupby('Method').agg(
    Avg_Gap=('Gap(%)', 'mean'),
    Max_Gap=('Gap(%)', 'max'),
    Avg_Time=(time_col, 'mean'),
    Opt_Count=('Is_Optimal', 'sum'),
    Total_Instances=('Instance', 'count')
).reset_index()

summary['Opt_Rate(%)'] = (summary['Opt_Count'] / summary['Total_Instances']) * 100

try:
    print(summary.to_markdown(index=False, floatfmt=".2f"))
except ImportError:
    print(summary.to_string(index=False, float_format="%.2f"))

# ==========================================
# PHẦN 2: VẼ BIỂU ĐỒ (4 Subplots)
# ==========================================
os.makedirs("results", exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# Khởi tạo khung biểu đồ 2x2 với kích thước lớn
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
palette = ["#1f77b4", "#ff7f0e", "#2ca02c"] # Xanh lam, Cam, Xanh lá

# 1. Boxplot (Phân phối Gap) - Góc trên trái
sns.boxplot(data=data, x='Method', y='Gap(%)', ax=axes[0, 0], palette=palette)
axes[0, 0].set_title('Phân phối sai số Best Gap (%)', fontweight='bold', pad=10)
axes[0, 0].set_ylabel('Gap (%)')

# 2. Barplot (Thời gian trung bình) - Góc trên phải
sns.barplot(data=summary, x='Method', y='Avg_Time', ax=axes[0, 1], palette=palette)
axes[0, 1].set_title('Thời gian tính toán trung bình', fontweight='bold', pad=10)
axes[0, 1].set_ylabel('Thời gian (s)')

# 3. Lineplot (Last Gap qua từng Instances) - Góc dưới trái
sns.lineplot(data=data, x='Instance', y=last_gap_col, hue='Method', style='Method', 
             markers=True, dashes=False, ax=axes[1, 0], palette=palette, linewidth=2, markersize=8)
axes[1, 0].set_title('Biến động Last Gap (%) qua từng Instances', fontweight='bold', pad=10)
axes[1, 0].set_ylabel('Last Gap (%)')
axes[1, 0].set_xlabel('Instances (Tăng dần theo kích thước)')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Lineplot (Last Time qua từng Instances) - Góc dưới phải
sns.lineplot(data=data, x='Instance', y=time_col, hue='Method', style='Method', 
             markers=True, dashes=False, ax=axes[1, 1], palette=palette, linewidth=2, markersize=8)
axes[1, 1].set_title('Thời gian giải (Last Time) qua từng Instances', fontweight='bold', pad=10)
axes[1, 1].set_ylabel('Thời gian giải (s)')
axes[1, 1].set_xlabel('Instances (Tăng dần theo kích thước)')
axes[1, 1].tick_params(axis='x', rotation=45)

# Tối ưu hóa layout để chữ không đè lên nhau
plt.tight_layout()

# Lưu và hiển thị
save_path = os.path.join("results", "scientific_comparison_full.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n[INFO] Đã lưu biểu đồ khoa học (4 khung) tại: {save_path}")

plt.show()