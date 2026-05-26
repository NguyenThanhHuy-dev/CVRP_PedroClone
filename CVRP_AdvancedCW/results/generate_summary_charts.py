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
                "Phương pháp": (
                    "Phương pháp đề xuất"
                    if m == "PySAT"
                    else "Ferreira" if m == "Fer" else m
                ),
                "Số nghiệm khả thi": feasible,
                "Số nghiệm tối ưu": optimal,
                "Số bài không giải được": failed,
                "Gap trung bình (%)": round(avg_gap, 2),
                "Thời gian trung bình (s)": round(avg_time, 2),
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
    latex_summary += "\\small\n\n"

    latex_summary += (
        f"\\caption{{Kết quả tổng hợp thực nghiệm trên bộ dữ liệu {dataset_name}}}\n"
    )

    latex_summary += f"\\label{{tab:summary_{dataset_name.lower()}}}\n\n"

    latex_summary += "\\resizebox{\\textwidth}{!}{%\n"

    latex_summary += "\\begin{tabular}{lccccc}\n"
    latex_summary += "\\toprule\n"

    latex_summary += (
        "\\textbf{Phương pháp} & "
        "\\textbf{Số nghiệm khả thi} & "
        "\\textbf{Số nghiệm tối ưu} & "
        "\\textbf{Số bài không giải được} & "
        "\\textbf{Gap trung bình (\\%)} & "
        "\\textbf{Thời gian trung bình (s)} \\\\\n"
    )

    latex_summary += "\\midrule\n"

    for _, row in df_summary.iterrows():

        latex_summary += (
            f"{row['Phương pháp']} & "
            f"{row['Số nghiệm khả thi']} & "
            f"{row['Số nghiệm tối ưu']} & "
            f"{row['Số bài không giải được']} & "
            f"{row['Gap trung bình (%)']} & "
            f"{row['Thời gian trung bình (s)']} \\\\\n"
        )

    latex_summary += "\\bottomrule\n"
    latex_summary += "\\end{tabular}%\n"
    latex_summary += "}\n\n"

    latex_summary += (
        "\\vspace{0.2cm}\n"
        "\\footnotesize{"
        "OOM: vượt quá giới hạn bộ nhớ; "
        "CW\\_UNSAT: thuật toán Clarke--Wright không sinh được nghiệm khả thi; "
        "INFEASIBLE: không tìm được nghiệm thoả mãn các ràng buộc bài toán."
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

    ax.bar(x - width, df_summary["Số nghiệm tối ưu"], width, label="Nghiệm tối ưu")

    ax.bar(x, df_summary["Số nghiệm khả thi"], width, label="Nghiệm khả thi")

    ax.bar(x + width, df_summary["Số bài không giải được"], width, label="Không giải được")

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
            "Phương pháp đề xuất": df_final["Gap_PySAT"],
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
    return df_summary

# ==========================================
# TẠO BẢNG TỔNG HỢP CHUNG
# ==========================================
def generate_overall_summary_table(all_summaries):

    methods = [
        "Ferreira",
        "CPLEX",
        "Gurobi",
        "Phương pháp đề xuất",
    ]

    datasets = ["A", "B", "E", "F", "P", "X"]

    latex = ""

    latex += "\\begin{table*}[t]\n"
    latex += "\\centering\n"
    latex += "\\small\n\n"

    latex += (
        "\\caption{Tổng hợp kết quả thực nghiệm "
        "trên các bộ dữ liệu benchmark}\n"
    )

    latex += "\\label{tab:overall_summary}\n\n"

    latex += "\\resizebox{\\textwidth}{!}{%\n"

    # ======================================
    # HEADER
    # ======================================

    latex += "\\begin{tabular}{l"

    for _ in datasets:
        latex += "ccc|"

    latex = latex[:-1]  # bỏ dấu | cuối

    latex += "}\n"

    latex += "\\toprule\n"

    # Dataset header
    latex += "& "

    for i, d in enumerate(datasets):

        if i < len(datasets) - 1:
            latex += (
                f"\\multicolumn{{3}}{{c|}}{{\\textbf{{{d}}}}} & "
            )
        else:
            latex += (
                f"\\multicolumn{{3}}{{c}}{{\\textbf{{{d}}}}}"
            )

    latex += " \\\\\n"

    # cmidrule
    start = 2

    for i in range(len(datasets)):

        end = start + 2

        latex += f"\\cmidrule(lr){{{start}-{end}}}\n"

        start += 3

    # metric header
    latex += "\\textbf{Phương pháp} "

    for _ in datasets:
        latex += (
            "& Opt & Gap & Time "
        )

    latex += "\\\\\\n"

    latex += "\\midrule\n"

    # ======================================
    # BODY
    # ======================================

    for method in methods:

        latex += method

        for dataset in datasets:

            df = all_summaries[dataset]

            row = df[df["Phương pháp"] == method].iloc[0]

            opt = row["Số nghiệm tối ưu"]
            gap = row["Gap trung bình (%)"]
            time = row["Thời gian trung bình (s)"]

            latex += (
                f" & {opt}"
                f" & {gap:.2f}"
                f" & {time:.0f}"
            )

        latex += " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}%\n"
    latex += "}\n\n"

    latex += (
        "\\vspace{0.2cm}\n"
        "\\footnotesize{"
        "Opt: số lượng nghiệm tối ưu tìm được; "
        "Gap: độ lệch trung bình so với cận tốt nhất (\\%); "
        "Time: thời gian giải trung bình (giây); "
        "OOM: vượt quá giới hạn bộ nhớ; "
        "CW\\_UNSAT: thuật toán Clarke--Wright không sinh được nghiệm khả thi; "
        "INFEASIBLE: không tìm được nghiệm thoả mãn các ràng buộc bài toán."
        "}\n"
    )

    latex += "\\end{table*}\n"

    # ======================================
    # WRITE FILE
    # ======================================

    with open(
        f"{FIGURES_DIR}/overall_summary_table.tex",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(latex)

    print("Đã sinh bảng tổng hợp overall_summary_table.tex")
    
# ==========================================
# BIỂU ĐỒ TỔNG HỢP OVERALL SUMMARY
# ==========================================
def generate_overall_summary_charts(all_summaries):

    datasets = ["A", "B", "E", "F", "P", "X"]

    methods = [
        "Ferreira",
        "CPLEX",
        "Gurobi",
        "Phương pháp đề xuất",
    ]

    # ======================================
    # CHUẨN BỊ DỮ LIỆU
    # ======================================

    opt_data = {m: [] for m in methods}
    gap_data = {m: [] for m in methods}
    time_data = {m: [] for m in methods}

    for dataset in datasets:

        df = all_summaries[dataset]

        for method in methods:

            row = df[df["Phương pháp"] == method].iloc[0]

            opt_data[method].append(
                row["Số nghiệm tối ưu"]
            )

            gap_data[method].append(
                row["Gap trung bình (%)"]
            )

            time_data[method].append(
                row["Thời gian trung bình (s)"]
            )

    # ======================================
    # HÀM VẼ CHUNG
    # ======================================

    def grouped_bar_chart(data_dict, ylabel, title, filename):

        x = np.arange(len(datasets))

        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))

        offsets = [-1.5, -0.5, 0.5, 1.5]

        for i, method in enumerate(methods):

            ax.bar(
                x + offsets[i] * width,
                data_dict[method],
                width,
                label=method,
            )

        ax.set_xticks(x)

        ax.set_xticklabels(datasets)

        ax.set_ylabel(ylabel)

        ax.set_title(title)

        ax.legend()

        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()

        plt.savefig(
            f"{FIGURES_DIR}/{filename}.pdf"
        )

        plt.close()

    # ======================================
    # VẼ CÁC BIỂU ĐỒ
    # ======================================

    grouped_bar_chart(
        opt_data,
        "Số nghiệm tối ưu",
        "So sánh số nghiệm tối ưu",
        "overall_optimal_comparison",
    )

    grouped_bar_chart(
        gap_data,
        "Gap trung bình (%)",
        "So sánh Gap trung bình",
        "overall_gap_comparison",
    )

    grouped_bar_chart(
        time_data,
        "Thời gian trung bình (s)",
        "So sánh thời gian giải trung bình",
        "overall_time_comparison",
    )

    print("Đã sinh biểu đồ tổng hợp overall summary.")


if __name__ == "__main__":

    datasets = ["A", "B", "E", "F", "P", "X"]

    all_summaries = {}

    for dataset in datasets:

        df_summary = generate_summary_and_charts(dataset)

        all_summaries[dataset] = df_summary

    generate_overall_summary_table(all_summaries)

    generate_overall_summary_charts(all_summaries)