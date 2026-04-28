#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Yêu cầu cài đặt thư viện:
# pip install pandas matplotlib seaborn tabulate

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# 1. Tìm đường dẫn file
def get_file_path(filename):
    if os.path.exists(filename):
        return filename
    if os.path.exists(f"results/{filename}"):
        return f"results/{filename}"
    return None


files = {
    "PySAT": "benchmark_E_pysat.csv",
    "Gurobi": "benchmark_E_gurobi.csv",
    "CPLEX": "benchmark_E_cplex.csv",
}

dfs = []
for method, filename in files.items():
    path = get_file_path(filename)
    if path:
        try:
            df = pd.read_csv(path)
            if not df.empty:
                df["Method"] = method
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

# 2. Xử lý dữ liệu số
time_col = "Last_Time(s)"
cost_col = "Best_Cost"
bks_col = "BKS"
last_gap_col = "Last_Gap(%)"

print(f"[INFO] Bắt đầu phân tích báo cáo nâng cao (Highlight Max-SAT)...")

for col in [cost_col, bks_col, time_col, last_gap_col]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Tính Best Gap (%)
data["Best_Gap(%)"] = ((data[cost_col] - data[bks_col]) / data[bks_col]) * 100
data["Is_Optimal"] = data["Best_Gap(%)"] <= 0.001

# Sắp xếp lại dữ liệu theo N
data["N_val"] = data["Instance"].str.extract(r"-n(\d+)").astype(float)
data.sort_values(by=["N_val", "Instance"], inplace=True)

# ==========================================
# ĐIỂM NHẤN: TÌM CÁC INSTANCES MÀ MAX-SAT CÓ ĐÓNG GÓP
# ==========================================
# Tự động quét các cột lưu thông số improvements (s-imp, p-imp, single_imp_count, v.v...)
imp_cols = [col for col in data.columns if "imp" in col.lower()]
data["MaxSAT_Contribution"] = False

if imp_cols:
    for col in imp_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        # Nếu phương pháp là PySAT và có đóng góp > 0 thì bật cờ True
        data.loc[data["Method"] == "PySAT", "MaxSAT_Contribution"] |= data[col] > 0

pysat_contrib_data = data[
    (data["Method"] == "PySAT") & (data["MaxSAT_Contribution"] == True)
]
if not pysat_contrib_data.empty:
    print(
        f"[INFO] Phát hiện {len(pysat_contrib_data)} instances có sự cải thiện trực tiếp từ Max-SAT!"
    )

# Cố định thứ tự và màu sắc
method_order = ["PySAT", "Gurobi", "CPLEX"]
color_palette = {
    "PySAT": "#1f77b4",  # Xanh lam
    "Gurobi": "#ff7f0e",  # Cam
    "CPLEX": "#2ca02c",  # Xanh lá
}

# ==========================================
# PHẦN 1: XUẤT BẢNG THỐNG KÊ
# ==========================================
summary = (
    data.groupby("Method")
    .agg(
        Avg_Best_Gap=("Best_Gap(%)", "mean"),
        Max_Best_Gap=("Best_Gap(%)", "max"),
        Avg_Last_Time=(time_col, "mean"),
        Opt_Count=("Is_Optimal", "sum"),
        Total_Instances=("Instance", "count"),
    )
    .reindex(method_order)
    .reset_index()
)

summary["Opt_Rate(%)"] = (summary["Opt_Count"] / summary["Total_Instances"]) * 100

print("\n" + "=" * 50)
print(" BẢNG TỔNG HỢP HIỆU NĂNG SO SÁNH 3 PHƯƠNG PHÁP")
print("=" * 50)
try:
    print(summary.to_markdown(index=False, floatfmt=".2f"))
except ImportError:
    print(summary.to_string(index=False, float_format="%.2f"))

# ==========================================
# PHẦN 2: VẼ BIỂU ĐỒ LẠI THEO YÊU CẦU MỚI
# Hàng 1: Thời gian (Barplot & Last Time)
# Hàng 2: Sai số (Best Gap & Last Gap cạnh nhau)
# ==========================================
os.makedirs("results", exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- HÀNG 1: BIỂU ĐỒ THỜI GIAN (TIME) ---
# 1. Barplot (Avg Last Time) - Góc trên trái
sns.barplot(
    data=summary,
    x="Method",
    y="Avg_Last_Time",
    ax=axes[0, 0],
    palette=color_palette,
    order=method_order,
)
axes[0, 0].set_title(
    "Thời gian tính toán trung bình (Last Time)", fontweight="bold", pad=10
)
axes[0, 0].set_ylabel("Thời gian (s)")

# 2. Lineplot (Last Time qua từng Instances) - Góc trên phải
sns.lineplot(
    data=data,
    x="Instance",
    y=time_col,
    hue="Method",
    style="Method",
    hue_order=method_order,
    style_order=method_order,
    markers=True,
    dashes=False,
    ax=axes[0, 1],
    palette=color_palette,
    linewidth=2,
    markersize=8,
)
axes[0, 1].set_title(
    "Thời gian kết thúc thuật toán (Last Time)", fontweight="bold", pad=10
)
axes[0, 1].set_ylabel("Last Time (s)")
axes[0, 1].set_xlabel("")
axes[0, 1].tick_params(axis="x", rotation=45)

# --- HÀNG 2: BIỂU ĐỒ SAI SỐ (GAP) ---
# 3. Lineplot (Best Gap) - Góc dưới trái
sns.lineplot(
    data=data,
    x="Instance",
    y="Best_Gap(%)",
    hue="Method",
    style="Method",
    hue_order=method_order,
    style_order=method_order,
    markers=True,
    dashes=False,
    ax=axes[1, 0],
    palette=color_palette,
    linewidth=2,
    markersize=8,
)

if not pysat_contrib_data.empty:
    sns.scatterplot(
        data=pysat_contrib_data,
        x="Instance",
        y="Best_Gap(%)",
        ax=axes[1, 0],
        color="red",
        edgecolor="black",
        linewidth=1,
        zorder=10,
        label="Max-SAT Cải thiện (S-Imp/P-Imp > 0)",
    )
    axes[1, 0].legend(fontsize="small", title="Chú thích", title_fontsize="small")

axes[1, 0].set_title(
    "Biến động Best Gap (%) qua từng Instances", fontweight="bold", pad=10
)
axes[1, 0].set_ylabel("Best Gap (%)")
axes[1, 0].set_xlabel("Instances (Tăng dần theo kích thước)")
axes[1, 0].tick_params(axis="x", rotation=45)

# 4. Lineplot (Last Gap) - Góc dưới phải
sns.lineplot(
    data=data,
    x="Instance",
    y=last_gap_col,
    hue="Method",
    style="Method",
    hue_order=method_order,
    style_order=method_order,
    markers=True,
    dashes=False,
    ax=axes[1, 1],
    palette=color_palette,
    linewidth=2,
    markersize=8,
)
axes[1, 1].set_title(
    "Biến động Last Gap (%) qua từng Instances", fontweight="bold", pad=10
)
axes[1, 1].set_ylabel("Last Gap (%)")
axes[1, 1].set_xlabel("Instances (Tăng dần theo kích thước)")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
# Lưu và hiển thị
save_path = os.path.join("results", "result_comparison_E.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"\n[INFO] Đã lưu biểu đồ khoa học tại: {save_path}")

plt.show()
