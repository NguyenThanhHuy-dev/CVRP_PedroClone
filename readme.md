# CVRP Hybrid Solver: Metaheuristic + MaxSAT 

> **Khóa luận Tốt nghiệp:** Phương pháp kết hợp tìm kiếm theo kinh nghiệm và biểu diễn MaxSAT cho bài toán định tuyến xe có ràng buộc trọng tải.  
> **Trường:** Đại học Công nghệ, Đại học Quốc gia Hà Nội (VNU-UET)  
> **Tác giả:** Nguyễn Thành Huy  
---

# Các tính năng chính

---

# Cấu trúc thư mục (Project Structure)
```text
CVRP_AdvancedCW/
├── classes/                     # Chứa các thuật toán Metaheuristic (ALNS, GLS, 2-opt, Cross-Exchange...)
├── instances/                   # Các bộ dữ liệu chuẩn từ CVRPLIB (A, B, E, F, P, X)
├── logs/                        # Log cho từng thuật toán (pysat/, gurobi/, cplex/)
├── results/
│   ├── benchmark_*.csv          # Bảng benchmark
├── advanced_optimizer_pysat.py  # Phương pháp đề xuất
├── advanced_optimizer_gurobi.py # Phương pháp đối chứng: CW -> Gurobi MILP (MTZ)
├── advanced_optimizer_cplex.py  # Phương pháp đối chứng: CW -> CPLEX MILP (MTZ)
├── run_benchmark_X.py           # Script chạy thực nghiệm hàng loạt cho bộ dữ liệu X 
├── generate_summary_charts.py   # Script tự động đọc CSV và xuất biểu đồ LaTeX/PDF
└── requirements.txt             # Danh sách thư viện Python

```
# Cài đặt
1. Yêu cầu hệ thống
Python 3.10+
Hệ điều hành:
Linux/Ubuntu (Khuyến nghị)
2. Cài đặt thư viện
pip install -r requirements.txt

- Các thư viện chính
python-sat
numpy
pandas
matplotlib
seaborn
networkx
3. Bản quyền bộ giải thương mại

Để chạy các mô hình đối chứng, hệ thống yêu cầu cài đặt và kích hoạt bản quyền tương ứng:

Gurobi Optimizer
- Hỗ trợ giấy phép Academic/Student
- IBM ILOG CPLEX Optimization Studio


Hệ thống cung cấp các script run_benchmark_*.py để tự động hóa hoàn toàn quá trình thực nghiệm và ghi nhận kết quả.

Chạy thực nghiệm hàng loạt

Bạn có thể khởi chạy bộ dữ liệu mong muốn và chỉ định phương pháp giải qua đối số dòng lệnh:

pysat (Mặc định)
gurobi
cplex
- Ví dụ
# Chạy kiến trúc lai MaxSAT đề xuất trên bộ dữ liệu X
python run_benchmark_X.py pysat

# Chạy mô hình đối chứng MILP Gurobi trên bộ dữ liệu P
python run_benchmark_P.py gurobi

# Chạy mô hình đối chứng MILP CPLEX trên bộ dữ liệu A
python run_benchmark_A.py cplex

# Kết quả đầu ra

Toàn bộ quá trình đánh giá được lưu vết minh bạch tại thư mục results/.

1. Chỉ số thống kê
benchmark_X_<method>.csv

Ghi nhận:

Tổng chi phí tuyến đường (Cost)
Độ sai lệch tối ưu (Gap)
Thời gian thực thi (Time)

cho từng bài toán thực nghiệm.

2. Nhật ký hệ thống
logs/<method>/*.log

Ghi nhận chi tiết:

Lịch sử sinh mệnh đề lười
Số lần phá vỡ cực tiểu địa phương
Thời gian CPU cấp phát
Quá trình tìm kiếm và hậu tối ưu
# Kiến trúc Hybrid tổng quát

Clarke-Wright Initialization

-> Route Reduction Heuristics
-> ALNS + GLS Metaheuristic Search
-> K-Route Extraction (K = 2)
-> Incremental MaxSAT Encoding
-> RC2 Optimization (PySAT)
-> Lazy DFJ & Capacity Cuts
-> Final CVRP Solution
# Bộ dữ liệu thực nghiệm

Hệ thống hỗ trợ benchmark từ CVRPLIB:

A
B
E
F
P
X

# Mục tiêu nghiên cứu

Dự án hướng tới xây dựng một kiến trúc lai có khả năng:

- Mở rộng trên các bài toán CVRP quy mô lớn
- Giảm chi phí tuyến đường so với heuristic truyền thống

- Khai thác ưu thế của suy luận logic MaxSAT

- Kết hợp hiệu quả giữa:
-   Metaheuristic
-   SAT/MaxSAT
