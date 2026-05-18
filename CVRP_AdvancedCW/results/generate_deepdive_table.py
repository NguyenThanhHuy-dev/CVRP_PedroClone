import pandas as pd
import os

def generate_deep_dive_table_all():
    datasets = ['A', 'B', 'E', 'P', 'X']
    all_data = []

    # Đọc tất cả các file benchmark của PySAT
    for ds in datasets:
        file_path = f'benchmark_{ds}_pysat.csv'
        if os.path.exists(file_path):
            df_temp = pd.read_csv(file_path)
            all_data.append(df_temp)
    
    if not all_data:
        print("Không tìm thấy file CSV nào!")
        return

    # Gộp thành 1 dataframe tổng
    df = pd.concat(all_data, ignore_index=True)

    # Đổi tên cột cho an toàn (loại bỏ khoảng trắng)
    df.columns = df.columns.str.strip()

    # Tính tổng số lần Max-SAT can thiệp
    df['Total_Imp'] = df['Single_Imp'] + df['Pair_Imp']

    # Lọc ra các bài có Max-SAT can thiệp VÀ sắp xếp lấy Top 10
    df_filtered = df[df['Total_Imp'] > 0].sort_values(by=['Total_Imp', 'Pair_Imp'], ascending=[False, False]).head(10)

    if df_filtered.empty:
        print("Không có bài nào Max-SAT can thiệp. (S-Imp và P-Imp đều = 0).")
        return

    # Sinh mã LaTeX
    latex_str = "\\begin{table}[H]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Số lần cải thiện nghiệm cục bộ bởi Max-SAT trên các bài toán tiêu biểu}\n"
    latex_str += "\\label{tab:deep_dive_maxsat}\n"
    latex_str += "\\begin{tabular}{l c c | r r | r}\n"
    latex_str += "\\toprule\n"
    latex_str += "\\textbf{Instance} & \\textbf{N} & \\textbf{K} & \\textbf{S-Imp (Đơn tuyến)} & \\textbf{P-Imp (Cặp tuyến)} & \\textbf{Gap cuối cùng} \\\\\n"
    latex_str += "\\midrule\n"

    for _, row in df_filtered.iterrows():
        instance = str(row['Instance']).replace('_', '\\_')
        n = int(row['N'])
        k = int(row['K'])
        s_imp = int(row['Single_Imp'])
        p_imp = int(row['Pair_Imp'])
        gap = float(row['Best_Gap(%)'])
        
        gap_str = f"\\textbf{{{gap:.2f}\\%}}" if gap <= 0.0 else f"{gap:.2f}\\%"
        
        latex_str += f"{instance} & {n} & {k} & {s_imp} & {p_imp} & {gap_str} \\\\\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"

    print(latex_str)

if __name__ == "__main__":
    generate_deep_dive_table_all()