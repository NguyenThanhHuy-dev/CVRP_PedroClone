#!/bin/bash

# ==========================================
# SCRIPT CHẠY BENCHMARK CVRP SONG SONG
# ==========================================

# Kích hoạt môi trường ảo (Sửa lại đường dẫn nếu cần)
source venv/bin/activate

# Tạo thư mục chứa log của runlim (nếu chưa có)
mkdir -p runlim_logs

echo "BẮT ĐẦU CHẠY BENCHMARK 3 LUỒNG (Gurobi, CPLEX, PySAT)..."

# Lệnh 1: Chạy GUROBI chạy ngầm (&)
# runlim -o [file_log] -t [max_time_tổng] python ...
runlim -o runlim_logs/gurobi_runlim.log -r 14400 python run_benchmark_B.py gurobi > runlim_logs/gurobi_console.log 2>&1 &
PID1=$!
echo "[LUỒNG 1] Gurobi đã khởi chạy (PID: $PID1)"

# Lệnh 2: Chạy CPLEX chạy ngầm (&)
runlim -o runlim_logs/cplex_runlim.log -r 14400 python run_benchmark_B.py cplex > runlim_logs/cplex_console.log 2>&1 &
PID2=$!
echo "[LUỒNG 2] CPLEX đã khởi chạy (PID: $PID2)"

# Lệnh 3: Chạy PYSAT chạy ngầm (&)
runlim -o runlim_logs/pysat_runlim.log -r 14400 python run_benchmark_B.py pysat > runlim_logs/pysat_console.log 2>&1 &
PID3=$!
echo "[LUỒNG 3] PySAT đã khởi chạy (PID: $PID3)"

echo "------------------------------------------------------"
echo "Cả 3 phương pháp đang chạy ngầm. Bạn có thể làm việc khác."
echo "Để xem tiến độ trực tiếp, hãy mở terminal khác và gõ lệnh:"
echo "  tail -f runlim_logs/pysat_console.log"
echo "  tail -f runlim_logs/gurobi_console.log"
echo "------------------------------------------------------"

# Lệnh wait giúp script không thoát ngay mà chờ cả 3 luồng hoàn thành
wait $PID1
wait $PID2
wait $PID3

echo "TẤT CẢ BENCHMARK ĐÃ HOÀN THÀNH!"