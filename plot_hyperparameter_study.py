#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoSeRec Section 5.5.1 — Hyperparameter Study Plot & Summary
============================================================
Reads hyperparameter_study_results.csv and produces:
  1. A printed table of all results
  2. Figure 8 replica: 8 subplots (4 params × 2 datasets)

Usage:
  python plot_hyperparameter_study.py            # plot + save PDF/PNG
  python plot_hyperparameter_study.py --table    # print table only (no plot)
"""

import csv
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).parent
RESULTS_CSV = ROOT_DIR / "hyperparameter_study_results.csv"
OUT_PNG     = ROOT_DIR / "hyperparameter_study_figure8.png"
OUT_PDF     = ROOT_DIR / "hyperparameter_study_figure8.pdf"

# ── Display labels ─────────────────────────────────────────────────────────────

PARAM_LABELS = {
    "alpha":  r"$\alpha$ (Substitute Rate)",
    "beta":   r"$\beta$ (Insert Rate)",
    "K":      r"$K$ (Short Seq Threshold)",
    "lambda": r"$\lambda$ (CL Loss Weight)",
}
PARAM_ORDER = ["alpha", "beta", "K", "lambda"]

DATASET_LABELS = {
    "Sports_and_Outdoors": "Sports",
    "Beauty":              "Beauty",
}
DATASET_ORDER = ["Sports_and_Outdoors", "Beauty"]

# ── Data loading ───────────────────────────────────────────────────────────────

def load_results():
    """
    Returns nested dict:
      data[(param, dataset)][value] = {ndcg5, hit5, ndcg10, hit10, ...}
    """
    data = {}
    if not RESULTS_CSV.exists():
        print(f"[ERROR] Results file not found: {RESULTS_CSV}")
        print("  Run `python run_hyperparameter_study.py` first.")
        sys.exit(1)

    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key   = (row["param"], row["dataset"])
            try:
                value  = float(row["value"])
                ndcg5  = float(row["ndcg5"])
            except (ValueError, KeyError):
                continue   # skip incomplete rows
            if key not in data:
                data[key] = {}
            data[key][value] = {
                "ndcg5":  ndcg5,
                "hit5":   _safe_float(row.get("hit5")),
                "ndcg10": _safe_float(row.get("ndcg10")),
                "hit10":  _safe_float(row.get("hit10")),
            }
    return data


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Table printing ─────────────────────────────────────────────────────────────

def print_table(data):
    print()
    print("=" * 70)
    print("  CoSeRec Hyperparameter Study — Results Summary (NDCG@5)")
    print("=" * 70)

    for param in PARAM_ORDER:
        print(f"\n── {PARAM_LABELS[param]} ──────────────────────────────────")
        # Collect all values for this param across both datasets
        all_values = set()
        for ds in DATASET_ORDER:
            key = (param, ds)
            if key in data:
                all_values.update(data[key].keys())
        all_values = sorted(all_values)

        # Header
        header = f"  {'Value':>8}" + "".join(f"  {DATASET_LABELS[ds]:>12}" for ds in DATASET_ORDER)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for v in all_values:
            row = f"  {v:>8.2f}"
            for ds in DATASET_ORDER:
                key = (param, ds)
                ndcg5 = data.get(key, {}).get(v, {}).get("ndcg5")
                if ndcg5 is not None:
                    row += f"  {ndcg5:>12.4f}"
                else:
                    row += f"  {'—':>12}"
            print(row)

    print()


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_figure8(data):
    matplotlib.rcParams.update({
        "font.size":       10,
        "axes.titlesize":  10,
        "axes.labelsize":  9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6.5))
    fig.suptitle(
        "Hyperparameter Sensitivity Study  (Figure 8 of CoSeRec paper)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    colors = {"Sports_and_Outdoors": "#2171b5", "Beauty": "#cb181d"}
    markers = {"Sports_and_Outdoors": "o", "Beauty": "s"}

    for col_idx, param in enumerate(PARAM_ORDER):
        for row_idx, dataset in enumerate(DATASET_ORDER):
            ax  = axes[row_idx][col_idx]
            key = (param, dataset)

            ax.set_title(f"{DATASET_LABELS[dataset]}", pad=4)
            if row_idx == n_rows - 1:
                ax.set_xlabel(PARAM_LABELS[param])
            if col_idx == 0:
                ax.set_ylabel("NDCG@5")

            if key not in data or not data[key]:
                ax.text(0.5, 0.5, "No data yet",
                        ha="center", va="center", transform=ax.transAxes,
                        color="gray", fontsize=9)
                ax.set_xticks([])
                continue

            values    = sorted(data[key].keys())
            ndcg_vals = [data[key][v]["ndcg5"] for v in values]

            ax.plot(
                values, ndcg_vals,
                color=colors[dataset],
                marker=markers[dataset],
                markersize=5, linewidth=1.5,
                label=DATASET_LABELS[dataset],
            )

            # Mark the best point
            best_idx = int(np.argmax(ndcg_vals))
            ax.plot(values[best_idx], ndcg_vals[best_idx],
                    marker="*", color="gold", markersize=10, zorder=5,
                    markeredgecolor=colors[dataset], markeredgewidth=0.8)

            ax.set_xticks(values)
            if param == "K":
                ax.set_xticklabels([str(int(v)) for v in values])

            # Tighten y-axis around actual range
            ymin, ymax = min(ndcg_vals), max(ndcg_vals)
            margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.001
            ax.set_ylim(ymin - margin, ymax + margin)

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(axis="x", rotation=30)

    # Add column titles (param names) above row 0
    for col_idx, param in enumerate(PARAM_ORDER):
        axes[0][col_idx].set_title(
            f"{PARAM_LABELS[param]}\n{DATASET_LABELS[DATASET_ORDER[0]]}",
            pad=4,
        )

    plt.tight_layout()

    # Save
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")
    print(f"Saved → {OUT_PDF}")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    cli = argparse.ArgumentParser(description="Plot CoSeRec hyperparameter study results")
    cli.add_argument("--table", action="store_true",
                     help="Print summary table only (skip plot)")
    cli.add_argument("--no-show", action="store_true",
                     help="Save figures without displaying them interactively")
    args = cli.parse_args()

    data = load_results()

    # How many results are loaded
    total = sum(len(v) for v in data.values())
    print(f"Loaded {total} result(s) from {RESULTS_CSV}")

    print_table(data)

    if not args.table:
        if args.no_show:
            matplotlib.use("Agg")
        plot_figure8(data)


if __name__ == "__main__":
    main()
