import pandas as pd
import numpy as np
import os

TARGET_SET = 'X'  # Chạy cho bộ X

def format_val(val, is_gap=False, is_pysat=False):
    if pd.isna(val) or val == 'N/A' or val == '':
        return "-"
    try:
        f_val = float(val)
        formatted = f"{f_val:.2f}"
        if is_gap and is_pysat and f_val <= 0.0:
            return f"\\textbf{{{formatted}}}"
        return formatted
    except ValueError:
        return str(val)

def generate_latex_table(dataset_name):
    print(f"Đang xử lý tạo bảng LaTeX ngắt trang (xoay ngang) cho bộ dữ liệu: {dataset_name}")
    
    file_ferreira = f'../old_solution/results/benchmark_{dataset_name}_ferreira.csv'
    file_cplex = f'benchmark_{dataset_name}_cplex.csv'
    file_gurobi = f'benchmark_{dataset_name}_gurobi.csv'
    file_pysat = f'benchmark_{dataset_name}_pysat.csv'

    if not os.path.exists(file_ferreira):
        print(f"Cảnh báo: Không tìm thấy file {file_ferreira}")
        return
        
    df_fer = pd.read_csv(file_ferreira)
    df_cplex = pd.read_csv(file_cplex)
    df_gurobi = pd.read_csv(file_gurobi)
    df_pysat = pd.read_csv(file_pysat)

    df_fer = df_fer[['Instance', 'BKS', 'Solver_Cost', 'Gap_%', 'Total_Time_s', 'Status']].copy()
    df_fer.columns = ['Instance', 'BKS', 'Cost_Fer', 'Gap_Fer', 'Time_Fer', 'Status_Fer']

    def extract_my_data(df, suffix):
        res = df[['Instance', 'Best_Cost', 'Best_Gap(%)', 'Last_Time(s)']].copy()
        res.columns = ['Instance', f'Cost_{suffix}', f'Gap_{suffix}', f'Time_{suffix}']
        return res

    df_c = extract_my_data(df_cplex, 'CPLEX')
    df_g = extract_my_data(df_gurobi, 'Gurobi')
    df_p = extract_my_data(df_pysat, 'PySAT')

    df_final = df_fer.merge(df_c, on='Instance', how='outer')\
                     .merge(df_g, on='Instance', how='outer')\
                     .merge(df_p, on='Instance', how='outer')

    # --- CẬP NHẬT CƠ CHẾ SẮP XẾP TOÁN HỌC ---
    df_final['N'] = df_final['Instance'].str.extract(r'-n(\d+)').astype(float)
    df_final['K'] = df_final['Instance'].str.extract(r'-k(\d+)').astype(float)
    df_final = df_final.sort_values(by=['N', 'K']).reset_index(drop=True)
    df_final = df_final.drop(columns=['N', 'K'])
    # ----------------------------------------

    # 4. Sinh mã LaTeX bằng LONGTABLE (Xoay ngang trang)
    latex_str = "% Cần thêm các package: \\usepackage{booktabs}, \\usepackage{multirow}, \\usepackage{longtable}, \\usepackage{pdflscape}\n\n"
    latex_str += "\\begin{landscape}\n"
    latex_str += "{\\small % Tăng cỡ chữ lên small vì đã có không gian ngang\n"
    latex_str += "\\setlength{\\tabcolsep}{4pt} % Khoảng cách cột vừa phải\n"
    latex_str += "\\begin{longtable}{l c | r r r | r r r | r r r | r r r}\n"
    latex_str += f"\\caption{{So sánh chất lượng nghiệm và thời gian giải trên bộ dữ liệu {dataset_name}}}\n"
    latex_str += f"\\label{{tab:benchmark_{dataset_name}}} \\\\\n"
    latex_str += "\\toprule\n"
    
    header_str = "\\multirow{2}{*}{\\textbf{Instance}} & \\multirow{2}{*}{\\textbf{BKS}} & "
    header_str += "\\multicolumn{3}{c|}{\\textbf{Ferreira et al.}} & "
    header_str += "\\multicolumn{3}{c|}{\\textbf{CPLEX}} & "
    header_str += "\\multicolumn{3}{c|}{\\textbf{Gurobi}} & "
    header_str += "\\multicolumn{3}{c}{\\textbf{PySAT (Đề xuất)}} \\\\\n"
    header_str += "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14}\n"
    header_str += " & & Cost & Gap(\\%) & Time(s) & Cost & Gap(\\%) & Time(s) & Cost & Gap(\\%) & Time(s) & Cost & Gap(\\%) & Time(s) \\\\\n"
    header_str += "\\midrule\n"

    latex_str += header_str
    latex_str += "\\endfirsthead\n\n"

    latex_str += "\\multicolumn{14}{c}{\\tablename\\ \\thetable{} -- Tiếp tục từ trang trước} \\\\\n"
    latex_str += "\\toprule\n"
    latex_str += header_str
    latex_str += "\\endhead\n\n"

    latex_str += "\\midrule\n"
    latex_str += "\\multicolumn{14}{r}{\\textit{Tiếp tục ở trang sau...}} \\\\\n"
    latex_str += "\\endfoot\n\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\endlastfoot\n\n"

    # Điền dữ liệu
    for index, row in df_final.iterrows():
        instance = str(row['Instance']).replace('_', '\\_')
        bks = format_val(row['BKS'])
        
        status_fer = str(row['Status_Fer'])
        try:
            cost_fer_val = float(row['Cost_Fer'])
        except:
            cost_fer_val = -1.0

        # [PATCH 1]: Xử lý lỗi False Optimality
        try:
            bks_val = float(row['BKS'])
        except:
            bks_val = 0.0
            
        if bks_val == 0.0 and cost_fer_val > 0:
            row['Gap_Fer'] = "N/A" 

        # [PATCH 2]: Nhánh xử lý INFEASIBLE và OOM / CW_UNSAT cho Ferreira
        if pd.isna(row['Cost_Fer']) or cost_fer_val == 0.0 or status_fer in ['CW_UNSAT', 'OOM']:
            if status_fer in ['CW_UNSAT', 'OOM']:
                safe_status = status_fer.replace('_', '\\_')
            else:
                safe_status = "INFEASIBLE"
            fer_str = "\\multicolumn{3}{c|}{" + safe_status + "}"
            
        # [PATCH 3]: Nhánh xử lý in nghiệm và đánh dấu TIMEOUT cho Ferreira
        else:
            c_fer = format_val(row['Cost_Fer'])
            g_fer = format_val(row['Gap_Fer'], is_gap=True)
            t_fer = format_val(row['Time_Fer'])
            
            if status_fer == "TIMEOUT_WITH_SOL":
                t_fer = f"{t_fer}*"
                
            fer_str = f"{c_fer} & {g_fer} & {t_fer}"
        
        # =====================================================================
        # [PATCH MỚI]: HÀM XỬ LÝ CHUNG CHO CPLEX, GUROBI, PYSAT BẮT LỖI INF
        # =====================================================================
        def get_solver_str(cost_val, gap_val, time_val, is_pysat=False):
            # Nếu Pandas đọc được chữ 'inf' hoặc float('inf')
            if str(cost_val).strip().lower() == 'inf':
                return "\\multicolumn{3}{c|}{CW\\_UNSAT}"
            
            c_str = format_val(cost_val)
            g_str = format_val(gap_val, is_gap=True, is_pysat=is_pysat)
            t_str = format_val(time_val)
            return f"{c_str} & {g_str} & {t_str}"

        cpx_str = get_solver_str(row['Cost_CPLEX'], row['Gap_CPLEX'], row['Time_CPLEX'])
        gur_str = get_solver_str(row['Cost_Gurobi'], row['Gap_Gurobi'], row['Time_Gurobi'])
        sat_str = get_solver_str(row['Cost_PySAT'], row['Gap_PySAT'], row['Time_PySAT'], is_pysat=True)

        latex_str += f"{instance} & {bks} & {fer_str} & {cpx_str} & {gur_str} & {sat_str} \\\\\n"
        # =====================================================================

    latex_str += "\\end{longtable}\n"
    latex_str += "}\n"
    latex_str += "\\end{landscape}\n"
    
    output_filename = f"table_{dataset_name}_landscape.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    print(f"-> Hoàn tất! Đã lưu mã LaTeX xoay ngang vào file {output_filename}")

if __name__ == "__main__":
    generate_latex_table(TARGET_SET)