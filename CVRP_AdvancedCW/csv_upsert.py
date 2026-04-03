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
      Avg_Cost  = (old_avg * old_runs + new_cost) / new_runs   (tính lại)
      Best_Cost = min(old_best, new_cost)
      Best_Gap  = tính lại từ Best_Cost và BKS
      Last_Cost = new_cost  (kết quả lần chạy vừa rồi)
      Last_Gap  = gap lần chạy vừa rồi
      Last_Time = time lần chạy vừa rồi
      Config, Solver → luôn ghi theo lần chạy mới nhất
"""

import csv
import os
from typing import Dict, Any, Optional


# Tên cột key để nhận dạng một instance+method
_KEY_COLS = ("Instance", "Method")

# Tất cả tên cột theo đúng thứ tự trong file CSV
FIELDNAMES = [
    # Định danh
    "Instance", "N", "K", "BKS",
    # Thống kê tổng hợp (được tính lại mỗi lần upsert)
    "Runs",
    "Best_Cost", "Best_Gap(%)",
    "Avg_Cost",  "Avg_Gap(%)",
    # Kết quả của lần chạy mới nhất
    "Last_Cost", "Last_Gap(%)", "Last_Time(s)",
    # Config đã dùng (lần chạy mới nhất)
    "Max_Single", "Max_Pair", "Num_Pairs", "Patience", "Max_Iter",
    # Thống kê solver
    "Single_Imp_Count", "Pair_Imp_Count",
    # Phân loại
    "Method", "Solver",
]


def _make_key(row: Dict[str, str]) -> tuple:
    return (row.get("Instance", ""), row.get("Method", ""))


def load_csv(filepath: str) -> Dict[tuple, Dict[str, str]]:
    """
    Đọc file CSV vào dict keyed by (Instance, Method).
    Trả về dict rỗng nếu file chưa tồn tại.
    """
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
    """
    Ghi toàn bộ dict ra file CSV, giữ thứ tự cột theo FIELDNAMES.
    Các cột không có trong FIELDNAMES sẽ bị bỏ qua (tránh crash).
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in data.values():
            writer.writerow(row)


def upsert_row(
    data: Dict[tuple, Dict[str, str]],
    instance:  str,
    n: int, k: int,
    bks:       int,
    new_cost:  float,
    elapsed:   float,
    cfg:       Dict[str, Any],
    stats:     Dict[str, Any],
    method:    str,
    max_iterations: int,
) -> Dict[tuple, Dict[str, str]]:
    """
    Upsert một kết quả vào dict.

    - Lần đầu chạy  → tạo row mới với Runs=1, Best=Avg=Last=new_cost.
    - Lần chạy lại  → cập nhật Runs, tính lại Avg, cập nhật Best nếu tốt hơn,
                       luôn ghi Last theo lần chạy vừa rồi.
    """
    bks_str  = str(bks) if bks > 0 else "N/A"
    new_gap  = ((new_cost - bks) / bks * 100) if bks > 0 else 0.0
    gap_str  = lambda cost: f"{((cost - bks) / bks * 100):.2f}" if bks > 0 else "N/A"

    key = (instance, method)

    if key not in data:
        # ── Lần đầu: tạo mới ──────────────────────────────────────────
        data[key] = {
            "Instance":         instance,
            "N":                str(n),
            "K":                str(k),
            "BKS":              bks_str,
            "Runs":             "1",
            "Best_Cost":        f"{new_cost:.0f}",
            "Best_Gap(%)":      gap_str(new_cost),
            "Avg_Cost":         f"{new_cost:.2f}",
            "Avg_Gap(%)":       gap_str(new_cost),
            "Last_Cost":        f"{new_cost:.0f}",
            "Last_Gap(%)":      gap_str(new_cost),
            "Last_Time(s)":     f"{elapsed:.2f}",
            "Max_Single":       str(cfg.get("max_single_size",   "")),
            "Max_Pair":         str(cfg.get("max_pairwise_size", "")),
            "Num_Pairs":        str(cfg.get("n_closest_pairs",   "")),
            "Patience":         str(cfg.get("patience",          "")),
            "Max_Iter":         str(max_iterations),
            "Single_Imp_Count": str(stats.get("single_imp_count",   0)),
            "Pair_Imp_Count":   str(stats.get("pairwise_imp_count", 0)),
            "Method":           method,
            "Solver":           stats.get("solver_name", "N/A"),
        }
    else:
        # ── Lần chạy lại: upsert ──────────────────────────────────────
        old = data[key]
        old_runs = int(old.get("Runs", "1"))
        old_avg  = float(old.get("Avg_Cost", str(new_cost)))
        old_best = float(old.get("Best_Cost", str(new_cost)))

        new_runs = old_runs + 1
        new_avg  = (old_avg * old_runs + new_cost) / new_runs
        new_best = min(old_best, new_cost)

        old.update({
            "Runs":             str(new_runs),
            "Best_Cost":        f"{new_best:.0f}",
            "Best_Gap(%)":      gap_str(new_best),
            "Avg_Cost":         f"{new_avg:.2f}",
            "Avg_Gap(%)":       gap_str(new_avg),
            "Last_Cost":        f"{new_cost:.0f}",
            "Last_Gap(%)":      gap_str(new_cost),
            "Last_Time(s)":     f"{elapsed:.2f}",
            # Config luôn cập nhật theo lần chạy mới nhất
            "Max_Single":       str(cfg.get("max_single_size",   "")),
            "Max_Pair":         str(cfg.get("max_pairwise_size", "")),
            "Num_Pairs":        str(cfg.get("n_closest_pairs",   "")),
            "Patience":         str(cfg.get("patience",          "")),
            "Max_Iter":         str(max_iterations),
            "Single_Imp_Count": str(stats.get("single_imp_count",   0)),
            "Pair_Imp_Count":   str(stats.get("pairwise_imp_count", 0)),
            "Solver":           stats.get("solver_name", "N/A"),
        })
        data[key] = old

    return data