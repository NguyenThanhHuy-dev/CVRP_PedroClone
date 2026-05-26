import pandas as pd
import numpy as np
import os

# =========================================================
# CHỈ CẦN THAY ĐỔI BIẾN NÀY ĐỂ RENDER BẢNG CHO BỘ DỮ LIỆU KHÁC
# (Ví dụ: 'A', 'B', 'E', 'P', 'X')
TARGET_SET = 'F'  
# =========================================================

def format_val(val, is_gap=False, is_pysat=False):
    """Hàm định dạng số: làm tròn 2 chữ số thập phân, xử lý NaN"""
    if pd.isna(val) or val == 'N/A' or val == '':
        return "-"
    try:
        f_val = float(val)
        formatted = f"{f_val:.2f}"
        
        # Bôi đậm nếu là thuật toán của bạn và đạt mốc tối ưu (Gap <= 0.00)
        if is_gap and is_pysat and f_val <= 0.0:
            return f"\\textbf{{{formatted}}}"
        return formatted
    except ValueError:
        return str(val)

def generate_latex_table(dataset_name):
    print(f"Đang xử lý tạo bảng LaTeX cho bộ dữ liệu: {dataset_name}")
    
    # 1. Đọc dữ liệu từ các file CSV dựa trên biến dataset_name
    file_ferreira = f'../old_solution/results/benchmark_{dataset_name}_ferreira.csv'
    file_cplex = f'benchmark_{dataset_name}_cplex.csv'
    file_gurobi = f'benchmark_{dataset_name}_gurobi.csv'
    file_pysat = f'benchmark_{dataset_name}_pysat.csv'

    # Kiểm tra xem file có tồn tại không trước khi đọc
    if not os.path.exists(file_ferreira):
        print(f"Cảnh báo: Không tìm thấy file {file_ferreira}")
        return
        
    df_fer = pd.read_csv(file_ferreira)
    df_cplex = pd.read_csv(file_cplex)
    df_gurobi = pd.read_csv(file_gurobi)
    df_pysat = pd.read_csv(file_pysat)

    # 2. Trích xuất và đổi tên các cột
    df_fer = df_fer[['Instance', 'BKS', 'Solver_Cost', 'Gap_%', 'Total_Time_s', 'Status']].copy()
    df_fer.columns = ['Instance', 'BKS', 'Cost_Fer', 'Gap_Fer', 'Time_Fer', 'Status_Fer']

    def extract_my_data(df, suffix):
        res = df[['Instance', 'Best_Cost', 'Best_Gap(%)', 'Last_Time(s)']].copy()
        res.columns = ['Instance', f'Cost_{suffix}', f'Gap_{suffix}', f'Time_{suffix}']
        return res

    df_c = extract_my_data(df_cplex, 'CPLEX')
    df_g = extract_my_data(df_gurobi, 'Gurobi')
    df_p = extract_my_data(df_pysat, 'PySAT')

    # 3. Merge dữ liệu (Outer join để không sót instance nào)
    df_final = df_fer.merge(df_c, on='Instance', how='outer')\
                     .merge(df_g, on='Instance', how='outer')\
                     .merge(df_p, on='Instance', how='outer')

    # Sắp xếp lại dataframe theo tên Instance cho đẹp mắt
    df_final = df_final.sort_values(by='Instance').reset_index(drop=True)

    # 4. Sinh mã LaTeX
    latex_str = "% Cần thêm các package: \\usepackage{booktabs}, \\usepackage{multirow}, \\usepackage{graphicx}\n\n"
    latex_str += "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += "\\resizebox{\\textwidth}{!}{\n"
    latex_str += "\\begin{tabular}{l c | r r r | r r r | r r r | r r r}\n"
    latex_str += "\\toprule\n"
    
    latex_str += "\\multirow{2}{*}{\\textbf{Instance}} & \\multirow{2}{*}{\\textbf{BKS}} & "
    latex_str += "\\multicolumn{3}{c|}{\\textbf{Ferreira et al.}} & "
    latex_str += "\\multicolumn{3}{c|}{\\textbf{CPLEX}} & "
    latex_str += "\\multicolumn{3}{c|}{\\textbf{Gurobi}} & "
    latex_str += "\\multicolumn{3}{c}{\\textbf{PySAT (Đề xuất)}} \\\\\n"
    
    latex_str += "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14}\n"
    latex_str += " & & Cost & Gap(\\%) & Time(s) & Cost & Gap(\\%) & Time(s) & Cost & Gap(\\%) & Time(s) & Cost & Gap(\\%) & Time(s) \\\\\n"
    latex_str += "\\midrule\n"

    for index, row in df_final.iterrows():
        instance = str(row['Instance']).replace('_', '\\_')
        bks = format_val(row['BKS'])
        
        # --- XỬ LÝ RIÊNG CHO FERREIRA ĐỂ BẮT LỖI CW_UNSAT, OOM ---
        status_fer = str(row['Status_Fer'])
        # Nếu Cost bị trống và Status mang thông báo lỗi
        if pd.isna(row['Cost_Fer']) and status_fer not in ['nan', 'None', 'SUCCESS', 'TIMEOUT_WITH_SOL']:
            safe_status = status_fer.replace('_', '\\_')
            fer_str = f"\\multicolumn{{3}}{{c|}}{{{safe_status}}}"
        else:
            c_fer = format_val(row['Cost_Fer'])
            g_fer = format_val(row['Gap_Fer'], is_gap=True)
            t_fer = format_val(row['Time_Fer'])
            fer_str = f"{c_fer} & {g_fer} & {t_fer}"
        
        # CPLEX
        c_cpx = format_val(row['Cost_CPLEX'])
        g_cpx = format_val(row['Gap_CPLEX'], is_gap=True)
        t_cpx = format_val(row['Time_CPLEX'])
        
        # Gurobi
        c_gur = format_val(row['Cost_Gurobi'])
        g_gur = format_val(row['Gap_Gurobi'], is_gap=True)
        t_gur = format_val(row['Time_Gurobi'])
        
        # PySAT
        c_sat = format_val(row['Cost_PySAT'])
        g_sat = format_val(row['Gap_PySAT'], is_gap=True, is_pysat=True)
        t_sat = format_val(row['Time_PySAT'])

        # Gắn vào dòng LaTeX
        latex_str += f"{instance} & {bks} & {fer_str} & {c_cpx} & {g_cpx} & {t_cpx} & {c_gur} & {g_gur} & {t_gur} & {c_sat} & {g_sat} & {t_sat} \\\\\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "}\n"
    
    # Cập nhật caption và label tự động theo tên biến
    latex_str += f"\\caption{{So sánh chất lượng nghiệm và thời gian giải trên bộ dữ liệu {dataset_name}}}\n"
    latex_str += f"\\label{{tab:benchmark_{dataset_name}}}\n"
    latex_str += "\\end{table}\n"

    # Tên file xuất ra cũng tự động
    output_filename = f"table_{dataset_name}_latex.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    print(f"-> Hoàn tất! Đã lưu mã LaTeX vào file {output_filename}")

if __name__ == "__main__":
    generate_latex_table(TARGET_SET)