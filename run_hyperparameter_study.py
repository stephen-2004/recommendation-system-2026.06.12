#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoSeRec Section 5.5.1 Hyperparameter Study Runner
==================================================
Runs all hyperparameter sensitivity experiments on Sports and Beauty datasets.

Paper hyperparameter → code argument mapping:
  α (substitute ratio)   →  --substitute_rate
  β (insert ratio)       →  --insert_rate
  K (short seq thresh)   →  --augment_threshold
  λ (CL loss weight)     →  --cf_weight

Usage:
  python run_hyperparameter_study.py              # run all experiments
  python run_hyperparameter_study.py --gpu 1      # specify GPU
  python run_hyperparameter_study.py --dry-run    # print commands only
"""

import subprocess
import os
import csv
import ast
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

# ─────────────────────────── Experiment Definitions ───────────────────────────

# Common fixed hyperparameters shared across ALL experiments
COMMON_FIXED = {
    "--num_hidden_layers":         2,
    "--num_attention_heads":       2,
    "--hidden_size":               64,
    "--max_seq_length":            50,
    "--lr":                        0.001,
    "--batch_size":                256,
    "--adam_beta1":                0.9,
    "--adam_beta2":                0.999,
    "--augmentation_warm_up_epoches": 160,  # E in the paper
    "--epochs":                    300,
    "--patience":                  40,      # early stopping patience
    "--augment_type_for_short":   "SIM",   # Substitute + Insert + Mask for short seqs
}

# Each experiment dict:
#   param       : name shown in CSV / plots (alpha / beta / K / lambda)
#   dataset     : data_name argument value
#   values      : list of values for the varied hyperparameter
#   fixed       : dict of {--arg: value} fixed for this group
#   varied_arg  : which --arg is being varied
EXPERIMENTS = [
    # ── α (substitute_rate) study ────────────────────────────────────────────
    {
        "param":      "alpha",
        "dataset":    "Sports_and_Outdoors",
        "values":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "fixed":      {"--augment_threshold": 4,  "--insert_rate": 0.4, "--cf_weight": 0.1},
        "varied_arg": "--substitute_rate",
    },
    {
        "param":      "alpha",
        "dataset":    "Beauty",
        "values":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "fixed":      {"--augment_threshold": 12, "--insert_rate": 0.5, "--cf_weight": 0.1},
        "varied_arg": "--substitute_rate",
    },

    # ── β (insert_rate) study ─────────────────────────────────────────────────
    {
        "param":      "beta",
        "dataset":    "Sports_and_Outdoors",
        "values":     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "fixed":      {"--augment_threshold": 4,  "--substitute_rate": 0.1, "--cf_weight": 0.1},
        "varied_arg": "--insert_rate",
    },
    {
        "param":      "beta",
        "dataset":    "Beauty",
        "values":     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "fixed":      {"--augment_threshold": 12, "--substitute_rate": 0.1, "--cf_weight": 0.1},
        "varied_arg": "--insert_rate",
    },

    # ── K (augment_threshold) study ───────────────────────────────────────────
    {
        "param":      "K",
        "dataset":    "Sports_and_Outdoors",
        "values":     [4, 8, 12, 16, 20],
        "fixed":      {"--substitute_rate": 0.4, "--insert_rate": 0.1, "--cf_weight": 0.1},
        "varied_arg": "--augment_threshold",
    },
    {
        "param":      "K",
        "dataset":    "Beauty",
        "values":     [4, 8, 12, 16, 20],
        "fixed":      {"--substitute_rate": 0.5, "--insert_rate": 0.1, "--cf_weight": 0.1},
        "varied_arg": "--augment_threshold",
    },

    # ── λ (cf_weight) study ───────────────────────────────────────────────────
    {
        "param":      "lambda",
        "dataset":    "Sports_and_Outdoors",
        "values":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "fixed":      {"--augment_threshold": 4,  "--substitute_rate": 0.4, "--insert_rate": 0.1},
        "varied_arg": "--cf_weight",
    },
    {
        "param":      "lambda",
        "dataset":    "Beauty",
        "values":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "fixed":      {"--augment_threshold": 12, "--substitute_rate": 0.5, "--insert_rate": 0.1},
        "varied_arg": "--cf_weight",
    },
]

# ─────────────────────────── Paths ────────────────────────────────────────────

ROOT_DIR    = Path(__file__).parent
SRC_DIR     = ROOT_DIR / "src"
RESULTS_CSV = ROOT_DIR / "hyperparameter_study_results.csv"

CSV_FIELDS = [
    "param", "dataset", "value",
    "ndcg5", "hit5", "ndcg10", "hit10", "ndcg20", "hit20",
    "model_idx", "log_file", "timestamp",
]

# model_idx base: use 1000+ to avoid collisions with existing experiments
MODEL_IDX_BASE = 1000

# ─────────────────────────── Helpers ──────────────────────────────────────────

def build_task_list():
    """Return list of (exp, value, model_idx) with stable, reproducible indices."""
    tasks = []
    idx = MODEL_IDX_BASE
    for exp in EXPERIMENTS:
        for value in exp["values"]:
            tasks.append((exp, value, idx))
            idx += 1
    return tasks


def load_completed():
    """Return dict of (param, dataset, str(value)) -> result row for finished runs."""
    completed = {}
    if not RESULTS_CSV.exists():
        return completed
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ndcg5"):   # has a result → finished
                key = (row["param"], row["dataset"], row["value"])
                completed[key] = row
    return completed


def append_result(row):
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_log(log_path):
    """
    Extract the last full-sort evaluation result from the log file.
    Returns a dict with NDCG@5 etc., or None if not found.
    """
    if not log_path.exists():
        return None
    content = log_path.read_text(encoding="utf-8", errors="ignore")
    # Match dict literals that contain 'NDCG@5'
    matches = re.findall(r"\{[^{}]*'NDCG@5'[^{}]*\}", content)
    if not matches:
        return None
    try:
        return ast.literal_eval(matches[-1])
    except Exception:
        return None


def build_command(exp, value, model_idx, gpu_id):
    cmd = [sys.executable, "main.py"]
    cmd += ["--data_name",  exp["dataset"]]
    cmd += ["--model_idx",  str(model_idx)]
    cmd += ["--gpu_id",     str(gpu_id)]

    # Group-specific fixed args
    for arg, val in exp["fixed"].items():
        cmd += [arg, str(val)]

    # The varied argument
    cmd += [exp["varied_arg"], str(value)]

    # Common fixed args
    for arg, val in COMMON_FIXED.items():
        cmd += [arg, str(val)]

    return cmd


def run_one(cmd, model_idx, dataset, dry_run=False):
    """Run a single training job. Returns the log file path."""
    log_name = f"CoSeRec-{dataset}-{model_idx}.txt"
    log_path  = SRC_DIR / "output" / log_name

    print(f"  CMD : {' '.join(cmd)}")
    print(f"  LOG : {log_path}")

    if dry_run:
        print("  [DRY RUN] Skipping actual execution.\n")
        return log_path

    result = subprocess.run(cmd, cwd=str(SRC_DIR))
    if result.returncode != 0:
        print(f"  [WARNING] Process exited with code {result.returncode}")
    return log_path


# ─────────────────────────── Main ─────────────────────────────────────────────

def main():
    cli = argparse.ArgumentParser(description="Run CoSeRec hyperparameter study")
    cli.add_argument("--gpu",     default="0",    help="GPU ID (default: 0)")
    cli.add_argument("--dry-run", action="store_true", help="Print commands without running")
    cli.add_argument("--param",   default=None,
                     help="Only run a specific param group: alpha / beta / K / lambda")
    cli.add_argument("--dataset", default=None,
                     help="Only run a specific dataset: Sports_and_Outdoors / Beauty")
    args = cli.parse_args()

    tasks     = build_task_list()
    completed = load_completed()

    # Optional filtering
    if args.param:
        tasks = [(e, v, m) for (e, v, m) in tasks if e["param"] == args.param]
    if args.dataset:
        tasks = [(e, v, m) for (e, v, m) in tasks if e["dataset"] == args.dataset]

    total     = len(tasks)
    done_cnt  = sum(1 for (e, v, m) in tasks
                    if (e["param"], e["dataset"], str(v)) in completed)

    print("=" * 72)
    print(f"CoSeRec Hyperparameter Study  —  {total} experiments total")
    print(f"Already completed: {done_cnt}  |  Remaining: {total - done_cnt}")
    print(f"GPU: {args.gpu}  |  Results CSV: {RESULTS_CSV}")
    print("=" * 72)

    for task_no, (exp, value, model_idx) in enumerate(tasks, 1):
        key = (exp["param"], exp["dataset"], str(value))
        prefix = f"[{task_no:>3}/{total}]  {exp['param']:6s}={str(value):<4}  {exp['dataset']}"

        if key in completed:
            print(f"{prefix}  →  SKIP (NDCG@5={completed[key]['ndcg5']})")
            continue

        print(f"\n{prefix}  →  RUNNING  (model_idx={model_idx})")
        cmd      = build_command(exp, value, model_idx, args.gpu)
        log_path = run_one(cmd, model_idx, exp["dataset"], dry_run=args.dry_run)

        if args.dry_run:
            continue

        # Parse result from log
        metrics = parse_log(log_path)
        if metrics:
            row = {
                "param":     exp["param"],
                "dataset":   exp["dataset"],
                "value":     str(value),
                "ndcg5":     metrics.get("NDCG@5", ""),
                "hit5":      metrics.get("HIT@5",  ""),
                "ndcg10":    metrics.get("NDCG@10",""),
                "hit10":     metrics.get("HIT@10", ""),
                "ndcg20":    metrics.get("NDCG@20",""),
                "hit20":     metrics.get("HIT@20", ""),
                "model_idx": str(model_idx),
                "log_file":  str(log_path),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            append_result(row)
            print(f"  DONE  NDCG@5={metrics.get('NDCG@5')}  HIT@5={metrics.get('HIT@5')}")
        else:
            print(f"  WARNING: could not parse NDCG@5 from {log_path}")

    print("\n" + "=" * 72)
    print("All experiments finished. Results saved to:")
    print(f"  {RESULTS_CSV}")
    print("\nNext steps:")
    print("  python plot_hyperparameter_study.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
