#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
csv_upsert.py
=============
Helper module: đọc toàn bộ CSV vào memory, upsert một row theo
key (Instance, Method), ghi lại toàn bộ file.

Logic upsert:
  - Nếu (Instance, Method) chưa có  → thêm row mới, Runs=1
  - Nếu (Instance, Method) đã có    → cập nhật row:
      Runs     += 1
      Avg_Cost  = (old_avg * old_runs + new_cost) / new_runs
      Best_Cost = min(old_best, new_cost)
      Best_Gap  = tính lại từ Best_Cost và BKS
      Last_Cost = new_cost
      Last_Gap  = gap lần chạy vừa rồi
      Last_Time = time lần chạy vừa rồi
      Config    → luôn ghi theo lần chạy mới nhất

Ghi chú cột:
  - Dùng chung cho cả 3 solver: gurobi, cplex, pysat.
  - Gurobi/CPLEX: Single_Imp và Pair_Imp là số lần MIP cải thiện được tuyến.
  - PySAT      : Single_Imp và Pair_Imp là số lần MaxSAT cải thiện được tuyến.
  - Không còn cột Solver (đã gộp vào cột Method).
  - G-Timeout  : True/False – có bị dừng do global timeout 1200s không.
"""

import csv
import os
from typing import Dict, Any


_KEY_COLS = ("Instance", "Method")

FIELDNAMES = [
    # Định danh
    "Instance", "N", "K", "BKS",
    # Thống kê tổng hợp
    "Runs",
    "Best_Cost", "Best_Gap(%)",
    "Avg_Cost",  "Avg_Gap(%)",
    # Kết quả lần chạy mới nhất
    "Last_Cost", "Last_Gap(%)", "Last_Time(s)",
    # Tham số config (lần chạy mới nhất)
    "Max_Single", "Max_Pair", "Num_Pairs", "Patience", "Max_Iter",
    "Single_Timeout(s)", "Pair_Timeout(s)", "Global_Timeout(s)",
    # Thống kê solver
    "Single_Imp", "Pair_Imp",
    "S-Timeout", "P-Timeout", "G-Timeout",
    # Phân loại
    "Method",
]


def _make_key(row: Dict[str, str]) -> tuple:
    return (row.get("Instance", ""), row.get("Method", ""))


def load_csv(filepath: str) -> Dict[tuple, Dict[str, str]]:
    """Đọc file CSV vào dict keyed by (Instance, Method)."""
    data: Dict[tuple, Dict[str, str]] = {}
    if not os.path.exists(filepath):
        return data
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _make_key(row)
            data[key] = dict(row)
    return data


def save_csv(filepath: str, data: Dict[tuple, Dict[str, str]]) -> None:
    """Ghi toàn bộ dict ra file CSV."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in data.values():
            writer.writerow(row)


def upsert_row(
    data:           Dict[tuple, Dict[str, str]],
    instance:       str,
    n:              int,
    k:              int,
    bks:            int,
    new_cost:       float,
    elapsed:        float,
    cfg:            Dict[str, Any],
    stats:          Dict[str, Any],
    method:         str,
    max_iterations: int,
) -> Dict[tuple, Dict[str, str]]:
    """
    Upsert một kết quả vào dict.
    """
    bks_str = str(bks) if bks > 0 else "N/A"
    gap_str = lambda cost: f"{((cost - bks) / bks * 100):.2f}" if bks > 0 else "N/A"

    key = (instance, method)

    # Tham số config để ghi vào CSV
    def _cfg(k_name, default=""):
        return str(cfg.get(k_name, default))

    row_cfg = {
        "Max_Single":        _cfg("max_single_size"),
        "Max_Pair":          _cfg("max_pairwise_size"),
        "Num_Pairs":         _cfg("n_closest_pairs"),
        "Patience":          _cfg("patience"),
        "Max_Iter":          str(max_iterations),
        "Single_Timeout(s)": _cfg("single_timeout"),
        "Pair_Timeout(s)":   _cfg("pairwise_timeout"),
        "Global_Timeout(s)": _cfg("global_timeout", 1200.0),
    }

    row_stats = {
        "Single_Imp": str(stats.get("single_imp_count",    0)),
        "Pair_Imp":   str(stats.get("pairwise_imp_count",  0)),
        "S-Timeout":  str(stats.get("single_timeouts",     0)),
        "P-Timeout":  str(stats.get("pairwise_timeouts",   0)),
        "G-Timeout":  str(stats.get("global_timeout",  False)),
    }

    if key not in data:
        data[key] = {
            "Instance":        instance,
            "N":               str(n),
            "K":               str(k),
            "BKS":             bks_str,
            "Runs":            "1",
            "Best_Cost":       f"{new_cost:.0f}",
            "Best_Gap(%)":     gap_str(new_cost),
            "Avg_Cost":        f"{new_cost:.2f}",
            "Avg_Gap(%)":      gap_str(new_cost),
            "Last_Cost":       f"{new_cost:.0f}",
            "Last_Gap(%)":     gap_str(new_cost),
            "Last_Time(s)":    f"{elapsed:.2f}",
            "Method":          method,
            **row_cfg,
            **row_stats,
        }
    else:
        old      = data[key]
        old_runs = int(old.get("Runs", "1"))
        old_avg  = float(old.get("Avg_Cost", str(new_cost)))
        old_best = float(old.get("Best_Cost", str(new_cost)))

        new_runs = old_runs + 1
        new_avg  = (old_avg * old_runs + new_cost) / new_runs
        new_best = min(old_best, new_cost)

        old.update({
            "Runs":          str(new_runs),
            "Best_Cost":     f"{new_best:.0f}",
            "Best_Gap(%)":   gap_str(new_best),
            "Avg_Cost":      f"{new_avg:.2f}",
            "Avg_Gap(%)":    gap_str(new_avg),
            "Last_Cost":     f"{new_cost:.0f}",
            "Last_Gap(%)":   gap_str(new_cost),
            "Last_Time(s)":  f"{elapsed:.2f}",
            **row_cfg,
            **row_stats,
        })
        data[key] = old

    return data