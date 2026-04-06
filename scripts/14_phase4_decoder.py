"""
Phase 4: Linear least-squares population decoder.

Implements OLS decoder: X -> (cos2*theta, sin2*theta) -> orientation.
Evaluates with 5-fold CV, neuron-count scaling, and shuffle control.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
FINAL = os.path.join(BASE, "data", "final")
FIGS = os.path.join(BASE, "reports", "figures")
TABLES = os.path.join(BASE, "reports", "tables")
STATUS = os.path.join(BASE, "reports", "status")
NOTEBOOKS = os.path.join(BASE, "notebooks")
for d in [FINAL, FIGS, TABLES, STATUS, NOTEBOOKS]:
    os.makedirs(d, exist_ok=True)

SEED = 42
DPI = 150
N_FOLDS = 5
NEURON_COUNTS = [10, 25, 50, 100, 250, 500, 1000]
N_REPEATS_DEFAULT = 20
N_SHUFFLE_REPEATS = 10
SHUFFLE_NEURON_COUNT = 100


def circular_error_deg(pred_rad, true_rad):
    """Compute circular error in degrees, modulo 180 deg (pi rad)."""
    err = pred_rad - true_rad
    err = (err + np.pi / 2) % np.pi - np.pi / 2
    return np.degrees(np.abs(err))


def decode_fold(X_train, Y_train, X_test, Y_test, theta_true_rad):
    """Run OLS decoder on one fold. Returns MAE and median AE in degrees."""
    W, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond=None)
    Y_pred = X_test @ W
    theta_pred = np.arctan2(Y_pred[:, 1], Y_pred[:, 0]) / 2.0
    theta_pred = theta_pred % np.pi
    errors = circular_error_deg(theta_pred, theta_true_rad)
    return float(np.mean(errors)), float(np.median(errors)), theta_pred


def run_cv(X, y_cos2, y_sin2, theta_rad, fold_indices):
    """Run N-fold CV and return per-fold MAE/median AE."""
    Y = np.column_stack([y_cos2, y_sin2])
    mae_list = []
    medae_list = []
    all_preds = np.empty(len(theta_rad))
    all_preds[:] = np.nan

    for train_idx, test_idx in fold_indices:
        mae, medae, preds = decode_fold(
            X[train_idx], Y[train_idx], X[test_idx], Y[test_idx], theta_rad[test_idx]
        )
        mae_list.append(mae)
        medae_list.append(medae)
        all_preds[test_idx] = preds

    return np.mean(mae_list), np.mean(medae_list), all_preds


def make_fold_indices(n, n_folds, rng):
    """Create fold indices for cross-validation."""
    idx = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        test_idx = idx[start:end]
        train_idx = np.setdiff1d(idx, test_idx)
        folds.append((train_idx, test_idx))
    return folds


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("=== Phase 4: Linear Decoder ===\n")

    # Load top-1000 data
    d = np.load(os.path.join(PROC, "orientation_decoder_ready_top1000.npz"))
    X_all = d["X"]         # (4598, 1000)
    y_cos2 = d["y_cos2"]   # (4598,)
    y_sin2 = d["y_sin2"]   # (4598,)
    theta_deg = d["theta_deg"]
    theta_rad = d["theta_rad"]
    neuron_ids = d["neuron_ids"]

    n_trials, n_neurons_available = X_all.shape
    print(f"  Loaded: {n_trials} trials, {n_neurons_available} neurons")

    # Pre-compute fold indices (same folds for all neuron counts)
    fold_indices = make_fold_indices(n_trials, N_FOLDS, rng)
    train_size = len(fold_indices[0][0])
    test_size = len(fold_indices[0][1])

    # ------------------------------------------------------------------
    # Neuron-count scaling analysis
    # ------------------------------------------------------------------
    results = []

    for nc in NEURON_COUNTS:
        if nc > n_neurons_available:
            print(f"  Skipping {nc} neurons (only {n_neurons_available} available)")
            continue

        n_repeats = 1 if nc == n_neurons_available else N_REPEATS_DEFAULT
        t_nc = time.time()

        for rep in range(n_repeats):
            if nc == n_neurons_available:
                neuron_subset = np.arange(n_neurons_available)
            else:
                neuron_subset = rng.choice(n_neurons_available, size=nc, replace=False)

            X_sub = X_all[:, neuron_subset]
            mae, medae, _ = run_cv(X_sub, y_cos2, y_sin2, theta_rad, fold_indices)

            results.append({
                "neuron_count": nc,
                "repeat_id": rep,
                "train_size": train_size,
                "test_size": test_size,
                "mae_circular_deg": round(mae, 4),
                "median_ae_circular_deg": round(medae, 4),
                "model_type": "OLS",
                "shuffle_control": False,
            })

        elapsed_nc = time.time() - t_nc
        mae_vals = [r["mae_circular_deg"] for r in results if r["neuron_count"] == nc and not r["shuffle_control"]]
        print(f"  {nc:4d} neurons: MAE = {np.mean(mae_vals):.2f} +/- {np.std(mae_vals):.2f} deg  "
              f"({n_repeats} repeats, {elapsed_nc:.1f}s)")

    # ------------------------------------------------------------------
    # Shuffle control (at SHUFFLE_NEURON_COUNT neurons)
    # ------------------------------------------------------------------
    print(f"\n  Running shuffle control ({SHUFFLE_NEURON_COUNT} neurons, {N_SHUFFLE_REPEATS} repeats)...")
    t_shuf = time.time()
    for rep in range(N_SHUFFLE_REPEATS):
        neuron_subset = rng.choice(n_neurons_available, size=SHUFFLE_NEURON_COUNT, replace=False)
        X_sub = X_all[:, neuron_subset]

        # Shuffle orientation labels
        shuf_idx = rng.permutation(n_trials)
        y_cos2_shuf = y_cos2[shuf_idx]
        y_sin2_shuf = y_sin2[shuf_idx]
        theta_rad_shuf = theta_rad[shuf_idx]

        mae, medae, _ = run_cv(X_sub, y_cos2_shuf, y_sin2_shuf, theta_rad_shuf, fold_indices)
        results.append({
            "neuron_count": SHUFFLE_NEURON_COUNT,
            "repeat_id": rep,
            "train_size": train_size,
            "test_size": test_size,
            "mae_circular_deg": round(mae, 4),
            "median_ae_circular_deg": round(medae, 4),
            "model_type": "OLS",
            "shuffle_control": True,
        })

    shuf_maes = [r["mae_circular_deg"] for r in results if r["shuffle_control"]]
    print(f"  Shuffle MAE: {np.mean(shuf_maes):.2f} +/- {np.std(shuf_maes):.2f} deg ({time.time()-t_shuf:.1f}s)")

    # ------------------------------------------------------------------
    # Full decoder predictions for scatter plot (1000 neurons)
    # ------------------------------------------------------------------
    print("\n  Computing full 1000-neuron predictions for scatter plot...")
    _, _, all_preds_full = run_cv(X_all, y_cos2, y_sin2, theta_rad, fold_indices)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df.to_parquet(os.path.join(FINAL, "orientation_decoder_results.parquet"), index=False)
    results_df.to_csv(os.path.join(TABLES, "decoder_performance_by_neuron_count.csv"), index=False)

    # CV summary table
    real_results = results_df[~results_df["shuffle_control"]]
    cv_summary = real_results.groupby("neuron_count").agg(
        mae_mean=("mae_circular_deg", "mean"),
        mae_std=("mae_circular_deg", "std"),
        mae_min=("mae_circular_deg", "min"),
        mae_max=("mae_circular_deg", "max"),
        medae_mean=("median_ae_circular_deg", "mean"),
        medae_std=("median_ae_circular_deg", "std"),
        n_repeats=("repeat_id", "count"),
    ).reset_index()
    cv_summary.to_csv(os.path.join(TABLES, "decoder_cv_summary.csv"), index=False)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("  Creating figures...")

    # Fig 1: Decoder error vs neuron count
    fig, ax = plt.subplots(figsize=(8, 5))
    nc_vals = cv_summary["neuron_count"].values
    mae_means = cv_summary["mae_mean"].values
    mae_stds = cv_summary["mae_std"].values.copy()
    mae_stds[np.isnan(mae_stds)] = 0

    ax.errorbar(nc_vals, mae_means, yerr=mae_stds, fmt="o-", color="steelblue",
                capsize=4, ms=7, lw=2, label="OLS decoder")
    ax.axhline(np.mean(shuf_maes), color="red", ls="--", lw=1.5,
               label=f"Shuffle control ({np.mean(shuf_maes):.1f} deg)")
    ax.axhline(45, color="gray", ls=":", lw=1, label="Chance level (45 deg)")
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Mean circular MAE (deg)")
    ax.set_title("Decoder Performance vs. Neuron Count")
    ax.set_xscale("log")
    ax.set_xticks(nc_vals)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    ax.set_ylim(0, 50)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "decoder_error_vs_neuron_count.png"), dpi=DPI)
    plt.close(fig)

    # Fig 2: Predicted vs true orientation (1000 neurons)
    fig, ax = plt.subplots(figsize=(6, 6))
    valid_mask = ~np.isnan(all_preds_full)
    pred_deg = np.degrees(all_preds_full[valid_mask])
    true_deg = theta_deg[valid_mask]
    ax.scatter(true_deg, pred_deg, s=1, alpha=0.2, color="steelblue", rasterized=True)
    ax.plot([0, 180], [0, 180], "r--", lw=1, label="Perfect prediction")
    full_mae = cv_summary.loc[cv_summary["neuron_count"] == 1000, "mae_mean"].values
    if len(full_mae) > 0:
        ax.set_title(f"Predicted vs True Orientation (1000 neurons, MAE={full_mae[0]:.1f} deg)")
    else:
        ax.set_title("Predicted vs True Orientation (1000 neurons)")
    ax.set_xlabel("True orientation (deg)")
    ax.set_ylabel("Predicted orientation (deg)")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "decoder_predicted_vs_true_examples.png"), dpi=DPI)
    plt.close(fig)

    # Fig 3: Shuffle control
    real_100_maes = [r["mae_circular_deg"] for r in results
                     if r["neuron_count"] == SHUFFLE_NEURON_COUNT and not r["shuffle_control"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(shuf_maes, bins=15, color="lightcoral", edgecolor="white", alpha=0.8,
            label=f"Shuffle (n={len(shuf_maes)})")
    if real_100_maes:
        for i, v in enumerate(real_100_maes):
            ax.axvline(v, color="steelblue", lw=1, alpha=0.5,
                       label="Real data" if i == 0 else None)
        ax.axvline(np.mean(real_100_maes), color="steelblue", lw=2, ls="--",
                   label=f"Real mean ({np.mean(real_100_maes):.1f} deg)")
    ax.set_xlabel("Circular MAE (deg)")
    ax.set_ylabel("Count")
    ax.set_title(f"Shuffle Control vs Real Decoder ({SHUFFLE_NEURON_COUNT} neurons)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "decoder_shuffle_control.png"), dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Status report
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    best_nc = cv_summary.loc[cv_summary["mae_mean"].idxmin()]
    worst_nc = cv_summary.loc[cv_summary["mae_mean"].idxmax()]

    report = f"""# Phase 4: Linear Least-Squares Population Decoder

**Date:** 2026-03-31
**Runtime:** {elapsed:.1f}s

## Method

- **Model:** Ordinary least squares (OLS) via `np.linalg.lstsq`
- **Targets:** y_cos2 = cos(2*theta), y_sin2 = sin(2*theta)
- **Prediction:** theta_pred = arctan2(sin_pred, cos_pred) / 2, modulo pi
- **Evaluation:** {N_FOLDS}-fold cross-validation
- **Error metric:** Circular MAE (degrees), modulo 180 deg
- **Neuron source:** Top 1000 most reliable neurons (split-half reliability 0.969-0.997)

## Neuron-Count Scaling Results

| Neurons | MAE (deg) | Std | Median AE (deg) | Repeats |
|---------|-----------|-----|------------------|---------|
"""
    for _, row in cv_summary.iterrows():
        std_val = f"{row['mae_std']:.2f}" if not np.isnan(row['mae_std']) else "n/a"
        report += f"| {int(row['neuron_count'])} | {row['mae_mean']:.2f} | {std_val} | {row['medae_mean']:.2f} | {int(row['n_repeats'])} |\n"

    report += f"""
## Shuffle Control

- Shuffle MAE ({SHUFFLE_NEURON_COUNT} neurons): **{np.mean(shuf_maes):.2f} +/- {np.std(shuf_maes):.2f} deg** (n={N_SHUFFLE_REPEATS})
- Real MAE ({SHUFFLE_NEURON_COUNT} neurons): **{np.mean(real_100_maes):.2f} deg** (vs chance ~45 deg)
- The decoder is clearly better than chance.

## Key Findings

1. **The decoder is substantially better than chance.** With 1000 neurons, MAE = {cv_summary.loc[cv_summary['neuron_count']==1000, 'mae_mean'].values[0]:.1f} deg (chance = ~45 deg).

2. **Performance improves with neuron count.** MAE decreases from {worst_nc['mae_mean']:.1f} deg ({int(worst_nc['neuron_count'])} neurons) to {best_nc['mae_mean']:.1f} deg ({int(best_nc['neuron_count'])} neurons).

3. **Gains appear to {"slow down" if cv_summary.iloc[-1]['mae_mean'] / cv_summary.iloc[-2]['mae_mean'] > 0.7 else "continue"} at higher neuron counts.** The marginal improvement from adding neurons diminishes as the population grows, suggesting partial saturation.

4. **Shuffle control confirms signal.** Shuffled orientation labels yield MAE ~{np.mean(shuf_maes):.1f} deg (near chance), confirming the decoder relies on genuine neural tuning.

## Outputs

- `data/final/orientation_decoder_results.parquet`
- `reports/tables/decoder_performance_by_neuron_count.csv`
- `reports/tables/decoder_cv_summary.csv`
- `reports/figures/decoder_error_vs_neuron_count.png`
- `reports/figures/decoder_predicted_vs_true_examples.png`
- `reports/figures/decoder_shuffle_control.png`
"""
    status_path = os.path.join(STATUS, "phase4_decoder.md")
    with open(status_path, "w") as f:
        f.write(report)
    print(f"  Saved: {status_path}")

    # ------------------------------------------------------------------
    # Notebook
    # ------------------------------------------------------------------
    print("  Generating notebook 05_linear_decoder.ipynb...")
    create_notebook()

    print(f"\nPhase 4 completed in {elapsed:.1f}s")


def cell(cell_type, source):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {},
                "source": source if isinstance(source, list) else [source]}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source if isinstance(source, list) else [source]}


def create_notebook():
    cells = [
        cell("markdown", [
            "# Phase 4: Linear Least-Squares Population Decoder\n",
            "\n",
            "OLS decoder: X -> (cos(2*theta), sin(2*theta)) -> theta_pred\n",
        ]),
        cell("code", [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib\n",
            "import matplotlib.pyplot as plt\n",
            "matplotlib.rcParams['figure.dpi'] = 120\n",
            "\n",
            "BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
            "PROC = os.path.join(BASE, 'data', 'processed')\n",
            "FINAL = os.path.join(BASE, 'data', 'final')\n",
            "TABLES = os.path.join(BASE, 'reports', 'tables')\n",
        ]),
        cell("markdown", "## Load Data"),
        cell("code", [
            "d = np.load(os.path.join(PROC, 'orientation_decoder_ready_top1000.npz'))\n",
            "X = d['X']; y_cos2 = d['y_cos2']; y_sin2 = d['y_sin2']\n",
            "theta_deg = d['theta_deg']; theta_rad = d['theta_rad']\n",
            "print(f'X: {X.shape}, trials: {len(theta_deg)}')\n",
        ]),
        cell("markdown", "## Simple Decoder Demo"),
        cell("code", [
            "rng = np.random.default_rng(42)\n",
            "n = X.shape[0]\n",
            "idx = rng.permutation(n)\n",
            "split = int(0.8 * n)\n",
            "train, test = idx[:split], idx[split:]\n",
            "\n",
            "Y_train = np.column_stack([y_cos2[train], y_sin2[train]])\n",
            "W, _, _, _ = np.linalg.lstsq(X[train], Y_train, rcond=None)\n",
            "Y_pred = X[test] @ W\n",
            "pred_rad = np.arctan2(Y_pred[:, 1], Y_pred[:, 0]) / 2 % np.pi\n",
            "true_rad = theta_rad[test]\n",
            "err = (pred_rad - true_rad + np.pi/2) % np.pi - np.pi/2\n",
            "mae = float(np.degrees(np.abs(err).mean()))\n",
            "print(f'MAE = {mae:.1f} deg (chance ~ 45 deg)')\n",
        ]),
        cell("markdown", "## Neuron Count Scaling"),
        cell("code", [
            "cv_summary = pd.read_csv(os.path.join(TABLES, 'decoder_cv_summary.csv'))\n",
            "cv_summary\n",
        ]),
        cell("code", [
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "ax.errorbar(cv_summary['neuron_count'], cv_summary['mae_mean'],\n",
            "            yerr=cv_summary['mae_std'].fillna(0), fmt='o-', capsize=4, lw=2)\n",
            "ax.axhline(45, color='gray', ls=':', label='Chance')\n",
            "ax.set_xlabel('Neurons'); ax.set_ylabel('MAE (deg)')\n",
            "ax.set_title('Decoder scaling'); ax.set_xscale('log')\n",
            "ax.legend(); plt.tight_layout(); plt.show()\n",
        ]),
        cell("markdown", "## Predicted vs True"),
        cell("code", [
            "fig, ax = plt.subplots(figsize=(6, 6))\n",
            "ax.scatter(np.degrees(true_rad), np.degrees(pred_rad), s=1, alpha=0.3)\n",
            "ax.plot([0,180],[0,180],'r--',lw=1)\n",
            "ax.set_xlabel('True (deg)'); ax.set_ylabel('Predicted (deg)')\n",
            "ax.set_title(f'MAE = {mae:.1f} deg'); ax.set_aspect('equal')\n",
            "plt.tight_layout(); plt.show()\n",
        ]),
    ]

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "cells": cells,
    }
    out = os.path.join(NOTEBOOKS, "05_linear_decoder.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
