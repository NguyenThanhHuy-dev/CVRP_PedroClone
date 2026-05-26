import os
import sys
import time
import re
import logging
from csv_upsert import load_csv, save_csv, upsert_row
import math

VALID_METHODS = ("gurobi", "cplex", "pysat")

def parse_method() -> str:
    if len(sys.argv) < 2:
        print(f"[INFO] Không có METHOD, dùng mặc định: pysat")
        return "pysat"
    method = sys.argv[1].strip().lower()
    if method not in VALID_METHODS:
        print(f"[LỖI] METHOD không hợp lệ: '{method}'")
        sys.exit(1)
    return method

METHOD = parse_method()
print(f"[INFO] Đang nạp solver: {METHOD.upper()}...")

if METHOD == "gurobi":
    from advanced_optimizer_gurobi import solve_advanced
elif METHOD == "cplex":
    from advanced_optimizer_cplex import solve_advanced
else:  
    from advanced_optimizer_pysat import solve_advanced

INSTANCE_DIR = os.path.join("instances", "B")
RESULTS_DIR  = "results"
RESULT_FILE = os.path.join(RESULTS_DIR, f"benchmark_B_{METHOD}.csv")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# SINGLE SOURCE OF TRUTH (CẤU HÌNH TRUNG TÂM)
ALNS_ITERATIONS = 150 

# Thời gian tối đa cho hàm Max-SAT Single Route
BASE_CONFIGS = {
    "gurobi": {"single_timeout":  40.0},
    "cplex":  {"single_timeout":  40.0},
    "pysat":  {"single_timeout":  40.0},
}

def build_dynamic_config(n: int, k: int) -> dict:
    base = BASE_CONFIGS[METHOD]
    avg  = n / k if k > 0 else float(n)

    if METHOD in ("gurobi", "cplex"):
        max_single   = max(20, min(35, int(avg * 2.0)))
        max_pairwise = max(16, min(28, int(avg * 3.0)))
        pair_timeout = float(math.ceil(min(120.0, max(30.0, avg * 9.0))))
    else:  # pysat
        max_single   = 15
        max_pairwise = 16
        pair_timeout = float(math.ceil(min(60.0, max(60.0, avg * 4.0))))
    return {
        "max_single_size":   max_single,
        "single_timeout":    base["single_timeout"],
        "max_pairwise_size": max_pairwise,
        "pairwise_timeout":  pair_timeout,
        "global_timeout":    1200.0, 
    }

# HELPERS
def get_bks_from_sol(sol_filepath: str) -> int:
    if not os.path.exists(sol_filepath): return 0
    try:
        with open(sol_filepath, 'r') as f:
            for line in f:
                if line.strip().lower().startswith('cost'):
                    parts = line.replace(':', '').split()
                    if len(parts) >= 2: return int(parts[1])
    except Exception: pass
    return 0

def get_instance_info(filename: str):
    name = filename.replace(".vrp", "")
    n_match = re.search(r'-n(\d+)', name)
    k_match = re.search(r'-k(\d+)', name)
    n = int(n_match.group(1)) if n_match else 0
    k = int(k_match.group(1)) if k_match else 0
    return name, n, k

def suppress_logging_to_console():
    root_logger = logging.getLogger()
    suppressed = []
    for h in root_logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.CRITICAL)
            suppressed.append(h)
    return suppressed

def restore_logging_to_console(suppressed_handlers):
    for h in suppressed_handlers:
        h.setLevel(logging.INFO)

# HÀM CHÍNH
def run_b_benchmark():
    if not os.path.exists(INSTANCE_DIR): return

    b_files = sorted(
        [f for f in os.listdir(INSTANCE_DIR) if f.startswith("B-") and f.endswith(".vrp")],
        key=lambda f: get_instance_info(f)[1] 
    )

    if not b_files: return

    print("\n" + "=" * 70)
    print(f"BENCHMARK B-SET  |  METHOD: {METHOD.upper()}  |  {len(b_files)} instances")
    print(f"ALNS Iterations  : {ALNS_ITERATIONS}")
    print(f"Single Timeout   : {BASE_CONFIGS[METHOD]['single_timeout']}s")
    print(f"KẾT QUẢ LƯU VÀO  : {RESULT_FILE}")
    print("=" * 70)

    csv_data = load_csv(RESULT_FILE)

    for idx, filename in enumerate(b_files):
        name, n, k = get_instance_info(filename)
        filepath     = os.path.join(INSTANCE_DIR, filename)
        sol_filepath = os.path.join(INSTANCE_DIR, filename.replace(".vrp", ".sol"))
        bks = get_bks_from_sol(sol_filepath)

        existing_key = (name, METHOD)
        if existing_key in csv_data:
            old_runs = csv_data[existing_key].get("Runs", "0")
            old_best = csv_data[existing_key].get("Best_Cost", "?")
            print(f"\n[{idx+1:02d}/{len(b_files)}] {name} (N={n}, K={k}, BKS={bks}) ← đã có {old_runs} lần chạy, best={old_best}")
        else:
            print(f"\n[{idx+1:02d}/{len(b_files)}] {name} (N={n}, K={k}, BKS={bks}) ← lần đầu chạy")

        cfg = build_dynamic_config(n, k)
        print(f"  [Config] single={cfg['max_single_size']} pair={cfg['max_pairwise_size']} "
              f"s_to={cfg['single_timeout']:.0f}s p_to={cfg['pairwise_timeout']:.0f}s")

        start_t = time.time()
        suppressed = []
        try:
            suppressed = suppress_logging_to_console()

            opt_routes, opt_cost, stats = solve_advanced(
                filepath, config=cfg, max_iterations=ALNS_ITERATIONS, target_cost=float(bks)
            )

            restore_logging_to_console(suppressed)
            elapsed = time.time() - start_t
            gap = ((opt_cost - bks) / bks) * 100 if bks > 0 else 0.0

            print(f"  -> Cost: {opt_cost} | Gap: {gap:+.2f}% | Time: {elapsed:.1f}s")
            print(f"  -> S-Imp: {stats.get('single_imp_count', 0)} | P-Imp: {stats.get('pairwise_imp_count', 0)}")

            csv_data = upsert_row(
                data=csv_data, instance=name, n=n, k=k, bks=bks,
                new_cost=opt_cost, elapsed=elapsed, cfg=cfg, stats=stats,
                method=METHOD, max_iterations=ALNS_ITERATIONS,
            )
            save_csv(RESULT_FILE, csv_data)

            row = csv_data[(name, METHOD)]
            print(f"  -> Runs={row['Runs']} | Best={row['Best_Cost']} ({row['Best_Gap(%)']}%) | Avg={row['Avg_Cost']} ({row['Avg_Gap(%)']}%)")

        except Exception as e:
            restore_logging_to_console(suppressed)
            print(f"  -> LỖI: {e}")

    print("\nHOÀN THÀNH.")

if __name__ == "__main__":
    run_b_benchmark()