import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# CẤU HÌNH BAN ĐẦU
# ==========================================
TARGET_SET = "X"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
    }
)

# ==========================================
# CLEAN NUMERIC
# ==========================================
INVALID_TEXT = ["inf", "nan", "-", "", "cw_unsat", "oom", "infeasible"]


def clean_numeric(val):
    if pd.isna(val):
        return np.nan

    s = str(val).strip().lower()

    if s in INVALID_TEXT:
        return np.nan

    try:
        return float(val)
    except:
        return np.nan


# ==========================================
# CLEAN DỮ LIỆU FERREIRA
# ==========================================
def preprocess_ferreira(df_fer):

    df_fer["Solver_Cost"] = df_fer["Solver_Cost"].apply(clean_numeric)
    df_fer["Gap_%"] = df_fer["Gap_%"].apply(clean_numeric)
    df_fer["Total_Time_s"] = df_fer["Total_Time_s"].apply(clean_numeric)

    invalid_status = ["CW_UNSAT", "OOM", "INFEASIBLE"]

    mask_invalid = (
        df_fer["Status"].isin(invalid_status)
        | (df_fer["Solver_Cost"] <= 0)
        | (df_fer["Gap_%"] <= -100)
    )

    df_fer.loc[mask_invalid, "Solver_Cost"] = np.nan
    df_fer.loc[mask_invalid, "Gap_%"] = np.nan

    return df_fer


# ==========================================
# MAIN
# ==========================================
def generate_summary_and_charts(dataset_name):

    print(f"Đang xử lý bộ dữ liệu: {dataset_name}")

    file_fer = f"../old_solution/results/benchmark_{dataset_name}_ferreira.csv"
    file_cplex = f"benchmark_{dataset_name}_cplex.csv"
    file_gurobi = f"benchmark_{dataset_name}_gurobi.csv"
    file_pysat = f"benchmark_{dataset_name}_pysat.csv"

    # ======================================
    # ĐỌC FILE
    # ======================================

    df_fer = pd.read_csv(file_fer)
    df_cplex = pd.read_csv(file_cplex)
    df_gurobi = pd.read_csv(file_gurobi)
    df_pysat = pd.read_csv(file_pysat)

    # ======================================
    # LÀM SẠCH DỮ LIỆU FERREIRA
    # ======================================

    df_fer = preprocess_ferreira(df_fer)

    # ======================================
    # CHUẨN HÓA TÊN CỘT
    # ======================================

    df_fer = df_fer[["Instance", "Solver_Cost", "Gap_%", "Total_Time_s"]].rename(
        columns={
            "Solver_Cost": "Cost_Fer",
            "Gap_%": "Gap_Fer",
            "Total_Time_s": "Time_Fer",
        }
    )

    df_cpx = df_cplex[["Instance", "Best_Cost", "Best_Gap(%)", "Last_Time(s)"]].rename(
        columns={
            "Best_Cost": "Cost_CPLEX",
            "Best_Gap(%)": "Gap_CPLEX",
            "Last_Time(s)": "Time_CPLEX",
        }
    )

    df_gur = df_gurobi[["Instance", "Best_Cost", "Best_Gap(%)", "Last_Time(s)"]].rename(
        columns={
            "Best_Cost": "Cost_Gurobi",
            "Best_Gap(%)": "Gap_Gurobi",
            "Last_Time(s)": "Time_Gurobi",
        }
    )

    df_sat = df_pysat[["Instance", "Best_Cost", "Best_Gap(%)", "Last_Time(s)"]].rename(
        columns={
            "Best_Cost": "Cost_PySAT",
            "Best_Gap(%)": "Gap_PySAT",
            "Last_Time(s)": "Time_PySAT",
        }
    )

    # ======================================
    # GHÉP DỮ LIỆU
    # ======================================

    df_final = (
        df_fer.merge(df_cpx, on="Instance", how="outer")
        .merge(df_gur, on="Instance", how="outer")
        .merge(df_sat, on="Instance", how="outer")
    )

    methods = ["Fer", "CPLEX", "Gurobi", "PySAT"]

    for m in methods:

        df_final[f"Cost_{m}"] = df_final[f"Cost_{m}"].apply(clean_numeric)
        df_final[f"Gap_{m}"] = df_final[f"Gap_{m}"].apply(clean_numeric)
        df_final[f"Time_{m}"] = df_final[f"Time_{m}"].apply(clean_numeric)

    total_instances = len(df_final)

    # ======================================
    # TẠO BẢNG TÓM TẮT
    # ======================================

    summary_data = []

    for m in methods:

        feasible_mask = df_final[f"Cost_{m}"].notna() & (df_final[f"Cost_{m}"] > 0)

        feasible = feasible_mask.sum()

        optimal = (feasible_mask & (df_final[f"Gap_{m}"] <= 0.0)).sum()

        failed = total_instances - feasible

        avg_gap = df_final.loc[feasible_mask, f"Gap_{m}"].mean()

        avg_time = df_final.loc[feasible_mask, f"Time_{m}"].mean()

        summary_data.append(
            {
                "Phương pháp": "Ferreira" if m == "Fer" else m,
                "Số nghiệm hợp lệ": feasible,
                "Nghiệm tối ưu": optimal,
                "Thất bại / OOM": failed,
                "Gap trung bình (%)": round(avg_gap, 2),
                "Thời gian TB (s)": round(avg_time, 2),
            }
        )

    df_summary = pd.DataFrame(summary_data)

    print(df_summary)

    # ======================================
    # XUẤT BẢNG TÓM TẮT LATEX
    # ======================================

    latex_summary = ""

    latex_summary += "\\begin{table}[H]\n"
    latex_summary += "\\centering\n"

    latex_summary += f"\\caption{{Thống kê tổng hợp trên bộ dữ liệu {dataset_name}}}\n"

    latex_summary += f"\\label{{tab:summary_{dataset_name.lower()}}}\n"

    latex_summary += "\\begin{tabular}{lccccc}\n"
    latex_summary += "\\toprule\n"

    latex_summary += (
        "\\textbf{Phương pháp} & "
        "\\textbf{Nghiệm hợp lệ} & "
        "\\textbf{Nghiệm tối ưu} & "
        "\\textbf{Thất bại/OOM} & "
        "\\textbf{Gap TB (\\%)} & "
        "\\textbf{Time TB (s)} \\\\\n"
    )

    latex_summary += "\\midrule\n"

    for _, row in df_summary.iterrows():

        latex_summary += (
            f"{row['Phương pháp']} & "
            f"{row['Số nghiệm hợp lệ']} & "
            f"{row['Nghiệm tối ưu']} & "
            f"{row['Thất bại / OOM']} & "
            f"{row['Gap trung bình (%)']} & "
            f"{row['Thời gian TB (s)']} \\\\\n"
        )

    latex_summary += "\\bottomrule\n"
    latex_summary += "\\end{tabular}\n"

    latex_summary += (
        "\\vspace{0.2cm}\n"
        "\\footnotesize{"
        "OOM: vượt quá bộ nhớ; "
        "CW\\_UNSAT: Clarke-Wright không sinh được nghiệm khả thi; "
        "INFEASIBLE: không tìm được nghiệm thoả mãn ràng buộc."
        "}\n"
    )

    latex_summary += "\\end{table}"

    with open(
        f"{FIGURES_DIR}/summary_table_{dataset_name}.tex", "w", encoding="utf-8"
    ) as f:

        f.write(latex_summary)
    # ======================================
    # BIỂU ĐỒ BAR CHART
    # ======================================

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df_summary))
    width = 0.25

    ax.bar(x - width, df_summary["Nghiệm tối ưu"], width, label="Nghiệm tối ưu")

    ax.bar(x, df_summary["Số nghiệm hợp lệ"], width, label="Nghiệm hợp lệ")

    ax.bar(x + width, df_summary["Thất bại / OOM"], width, label="Thất bại / OOM")

    ax.set_xticks(x)
    ax.set_xticklabels(df_summary["Phương pháp"])

    ax.set_ylabel("Số lượng bài toán")

    ax.set_title(f"Tỷ lệ giải thành công trên bộ dữ liệu {dataset_name}")

    ax.legend()

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.savefig(f"{FIGURES_DIR}/bar_status_{dataset_name}.pdf")

    plt.close()

    # ======================================
    # BIỂU ĐỒ BOXPLOT
    # ======================================

    gap_data = pd.DataFrame(
        {
            "Ferreira": df_final["Gap_Fer"],
            "CPLEX": df_final["Gap_CPLEX"],
            "Gurobi": df_final["Gap_Gurobi"],
            "PySAT": df_final["Gap_PySAT"],
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(data=gap_data, ax=ax, showfliers=True)

    ax.set_ylabel("Gap (%)")

    ax.set_title(f"Phân bố Gap trên bộ dữ liệu {dataset_name}")

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Không hiển thị các giá trị âm vô nghĩa
    q99 = np.nanpercentile(gap_data.values.flatten(), 99)

    ax.set_ylim(0, q99 * 1.1)

    plt.tight_layout()

    plt.savefig(f"{FIGURES_DIR}/boxplot_gap_{dataset_name}.pdf")

    plt.close()

    print("Hoàn tất sinh biểu đồ.")


if __name__ == "__main__":
    generate_summary_and_charts(TARGET_SET)
