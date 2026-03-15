"""
05_evaluate.py
--------------
Load walk-forward results and produce:
  1. Per-variable MAE summary + competition score
  2. Actual vs Predicted time-series plots for each target
  3. MAE-over-steps plot (convergence check)

All plots saved to results/plots/.

Run: python 05_evaluate.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(BASE_DIR, "results", "walk_forward_results.csv")
PLOTS_DIR   = os.path.join(BASE_DIR, "results", "plots")
REPORT_TXT  = os.path.join(BASE_DIR, "results", "main_metrics_report.txt")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Target metadata
# ---------------------------------------------------------------------------
TARGETS = [
    "target_temperature_2m",
    "target_rain",
]

TARGET_LABELS = {
    "target_temperature_2m": "Temperature 2m (°C)",
    "target_rain":           "Rain (mm)",
}

COLORS = {
    "target_temperature_2m": ("#e63946", "#457b9d"),
    "target_rain":           ("#6a4c93", "#1982c4"),
}

plt.rcParams.update({
    "figure.facecolor" : "#1a1a2e",
    "axes.facecolor"   : "#16213e",
    "axes.edgecolor"   : "#444",
    "axes.labelcolor"  : "#e0e0e0",
    "xtick.color"      : "#aaa",
    "ytick.color"      : "#aaa",
    "text.color"       : "#e0e0e0",
    "grid.color"       : "#2a2a4a",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "legend.framealpha": 0.3,
    "legend.edgecolor" : "#555",
    "font.family"      : "sans-serif",
})

# ---------------------------------------------------------------------------
# 1. Load results
# ---------------------------------------------------------------------------
print(f"Loading results from {RESULTS_CSV} …")
df = pd.read_csv(RESULTS_CSV, parse_dates=["time"])
print(f"  Rows: {len(df):,}")

# ---------------------------------------------------------------------------
# 2. MAE summary per target
# ---------------------------------------------------------------------------
print("\n=== Per-Target MAE ===")
metrics_dict = {}
for target in TARGETS:
    actual = df[f"actual_{target}"].dropna()
    pred   = df[f"pred_{target}"].loc[actual.index]
    mae    = float(mean_absolute_error(actual, pred))
    rmse   = float(np.sqrt(mean_squared_error(actual, pred)))
    r2     = float(r2_score(actual, pred))
    bias   = float(np.mean(pred - actual))
    corr   = float(np.corrcoef(actual, pred)[0, 1])

    step_mae = df.groupby("step").apply(
        lambda g: mean_absolute_error(g[f"actual_{target}"], g[f"pred_{target}"]),
        include_groups=False,
    )

    metrics_dict[target] = {
        "mae": mae, "rmse": rmse, "r2": r2, "bias": bias, "corr": corr,
        "step_mae_mean": float(step_mae.mean()),
        "step_mae_std":  float(step_mae.std()),
        "step_mae_min":  float(step_mae.min()),
        "step_mae_max":  float(step_mae.max()),
    }

    print(f"  {TARGET_LABELS[target]:<35} MAE = {mae:.4f}")

mae_global = np.mean([m["mae"] for m in metrics_dict.values()])
score      = (2.5 / (1 + mae_global)) * (len(TARGETS) / 17) * 100
score_competition_equivalent = (2.5 / (1 + mae_global)) * (5 / 17) * 100

print(f"\n  Global MAE  : {mae_global:.4f}")
print(f"  Score ({len(TARGETS)} targets)               : {score:.4f}")
print(f"  Score (competition-equivalent 5/17) : {score_competition_equivalent:.4f}")
print(f"  Formula active: (2.5 / (1 + {mae_global:.4f})) × ({len(TARGETS)}/17) × 100")
print(f"  Formula comp. : (2.5 / (1 + {mae_global:.4f})) × (5/17) × 100")

# ---------------------------------------------------------------------------
# 3. Actual vs Predicted time-series plots (one per target)
# ---------------------------------------------------------------------------
print("\nGenerating actual vs predicted plots …")

for target in TARGETS:
    label    = TARGET_LABELS[target]
    col_act  = f"actual_{target}"
    col_pred = f"pred_{target}"
    c_act, c_pred = COLORS[target]

    # Aggregate by time (average across locations for readability)
    ts = df.groupby("time")[[col_act, col_pred]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(ts["time"], ts[col_act],  color=c_act,  lw=1.2, alpha=0.9,  label="Actual")
    ax.plot(ts["time"], ts[col_pred], color=c_pred, lw=1.0, alpha=0.75, label="Predicted", linestyle="--")

    mae = metrics_dict[target]["mae"]
    ax.set_title(f"{label}  —  MAE = {mae:.4f}", fontsize=13, pad=10)
    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"{target}_actual_vs_pred.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# 4. MAE evolution across walk-forward steps (convergence check)
# ---------------------------------------------------------------------------
print("\nGenerating MAE-over-steps plot …")

fig, axes = plt.subplots(len(TARGETS), 1, figsize=(14, 3 * len(TARGETS)), sharex=True)
axes = np.atleast_1d(axes)

for ax, target in zip(axes, TARGETS):
    label = TARGET_LABELS[target]
    c_act, _ = COLORS[target]

    grp = df.groupby("step").apply(
        lambda g: mean_absolute_error(g[f"actual_{target}"], g[f"pred_{target}"]),
        include_groups=False,
    ).reset_index(name="mae")

    ax.plot(grp["step"], grp["mae"], color=c_act, lw=1.5, marker="o", markersize=3)
    avg = grp["mae"].mean()
    ax.axhline(avg, color="white", lw=0.8, linestyle=":", alpha=0.6, label=f"Avg {avg:.3f}")
    ax.set_ylabel(label.split("(")[0].strip(), fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True)

axes[-1].set_xlabel("Walk-Forward Step")
fig.suptitle("MAE Evolution Across Walk-Forward Steps", fontsize=13, y=1.01)

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "mae_evolution.png")
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# 5. Score summary banner
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print(f"  FINAL SCORE : {score:.4f}")
print(f"  Global MAE  : {mae_global:.4f}")
print("=" * 55)
print("\nAll plots saved to:", PLOTS_DIR)

# ---------------------------------------------------------------------------
# 6. Write comprehensive txt report
# ---------------------------------------------------------------------------
rain_actual   = df["actual_target_rain"].to_numpy()
rain_pred     = df["pred_target_rain"].to_numpy()
rain_non_zero = rain_actual > 0

lines = []
lines.append("METEOROLOGY FORECASTING - MAIN METRICS")
lines.append("=" * 52)
lines.append(f"Rows        : {len(df):,}")
lines.append(f"Steps       : {df['step'].nunique()} (from {df['step'].min()} to {df['step'].max()})")
lines.append("")

lines.append("PER-TARGET METRICS")
lines.append("-" * 52)
for target in TARGETS:
    m = metrics_dict[target]
    lines.append(f"{TARGET_LABELS[target]}:")
    lines.append(f"  MAE                : {m['mae']:.6f}")
    lines.append(f"  RMSE               : {m['rmse']:.6f}")
    lines.append(f"  R2                 : {m['r2']:.6f}")
    lines.append(f"  Bias (pred-real)   : {m['bias']:.6f}")
    lines.append(f"  Corr (Pearson)     : {m['corr']:.6f}")
    lines.append(f"  Step MAE mean/std  : {m['step_mae_mean']:.6f} / {m['step_mae_std']:.6f}")
    lines.append(f"  Step MAE min/max   : {m['step_mae_min']:.6f} / {m['step_mae_max']:.6f}")
    lines.append("")

lines.append("GLOBAL METRICS")
lines.append("-" * 52)
lines.append(f"Global MAE                      : {mae_global:.6f}")
lines.append(f"Score ({len(TARGETS)} targets)             : {score:.6f}")
lines.append(f"Score (competition-equivalent)  : {score_competition_equivalent:.6f}")
lines.append("")

lines.append("RAIN DIAGNOSTICS")
lines.append("-" * 52)
lines.append(f"Actual non-zero rate            : {np.mean(rain_non_zero):.6f}")
lines.append(f"Pred non-zero rate              : {np.mean(rain_pred > 0):.6f}")
lines.append(f"Pred mean when actual=0         : {rain_pred[~rain_non_zero].mean():.6f}")
lines.append(f"Pred mean when actual>0         : {rain_pred[rain_non_zero].mean():.6f}")
lines.append(f"Max actual rain                 : {rain_actual.max():.6f}")
lines.append(f"Max predicted rain              : {rain_pred.max():.6f}")
lines.append("")

lines.append("NOTES")
lines.append("-" * 52)
lines.append("- Bias > 0 means overestimation on average.")
lines.append("- For rain, lower dry-hour prediction mean indicates fewer false drizzle events.")
lines.append("- Initial train window: 6 months (covers wet + dry season before first validation).")
lines.append("- Rain rolling features use .sum() (total accumulation, not mean).")

with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nMetrics report saved to: {REPORT_TXT}")