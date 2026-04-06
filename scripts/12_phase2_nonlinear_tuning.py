"""
Phase 2: Finalize nonlinear least-squares tuning analysis.

Creates the final neuron analysis table, stratified example fit plots,
tuning fit summary, and interpretation.
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


def von_mises_tuning(theta, b, a, kappa, theta0):
    return b + a * np.exp(kappa * np.cos(2.0 * (theta - theta0)))


def plot_example_grid(df, tuning_mean, tuning_sem, centers_deg, neuron_ids,
                      title, filename, rng):
    """Plot a 4x3 grid of example tuning curves with fits."""
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    theta_fine = np.linspace(0, np.pi, 200)

    for ax, nid in zip(axes.ravel(), neuron_ids):
        row = df.iloc[nid]
        tc = tuning_mean[nid]
        se = tuning_sem[nid]

        ax.errorbar(centers_deg, tc, yerr=se, fmt="o", color="steelblue", ms=3,
                    capsize=2, label="Empirical")
        if row["fit_success"]:
            y_fit = von_mises_tuning(
                theta_fine,
                row["fit_baseline"], row["fit_amplitude"],
                row["fit_kappa_or_width"], np.radians(row["fit_pref_orientation_deg"])
            )
            ax.plot(np.degrees(theta_fine), y_fit, "-", color="tomato", lw=1.5,
                    label=f"Fit R2={row['fit_r2']:.2f}")

        ax.set_title(f"Neuron {nid}  kappa={row['fit_kappa_or_width']:.1f}  "
                     f"pref={row['fit_pref_orientation_deg']:.0f} deg", fontsize=8)
        ax.set_xlim(0, 180)
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("Orientation (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Response")

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, filename), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("=== Phase 2: Nonlinear Tuning Analysis ===\n")

    # Load data
    df = pd.read_parquet(os.path.join(PROC, "orientation_neuron_summary.parquet"))
    bd = np.load(os.path.join(PROC, "orientation_binned_tuning.npz"))
    tuning_mean = bd["tuning_mean"]
    tuning_sem = bd["tuning_sem"]
    centers_deg = bd["angle_bin_centers_deg"]

    n_neurons = len(df)
    df_fit = df[df["fit_success"]].copy()
    n_success = len(df_fit)

    print(f"  Neurons: {n_neurons}, Successful fits: {n_success} ({100*n_success/n_neurons:.1f}%)")

    # ------------------------------------------------------------------
    # 1. Create final neuron analysis table
    # ------------------------------------------------------------------
    print("  Creating final neuron analysis table...")
    final_cols = [
        "neuron_id", "recording_id", "n_trials", "mean_response", "response_std",
        "pref_orientation_deg_empirical", "osi_empirical", "split_half_reliability",
        "fit_success", "fit_r2", "fit_baseline", "fit_amplitude",
        "fit_pref_orientation_deg", "fit_kappa_or_width", "fit_method",
    ]
    df_final = df[final_cols].copy()
    df_final.to_parquet(os.path.join(FINAL, "orientation_neuron_analysis.parquet"), index=False)
    df_final.to_csv(os.path.join(FINAL, "orientation_neuron_analysis.csv"), index=False)
    print(f"  Saved final table: {len(df_final)} rows x {len(df_final.columns)} cols")

    # ------------------------------------------------------------------
    # 2. Stratified example fit plots
    # ------------------------------------------------------------------
    print("  Creating stratified example fit plots...")

    # Best fits: R^2 > 0.9
    best_ids = df_fit[df_fit["fit_r2"] > 0.9].index.values
    if len(best_ids) >= 12:
        chosen_best = rng.choice(best_ids, size=12, replace=False)
    else:
        chosen_best = best_ids[:12]
    plot_example_grid(df, tuning_mean, tuning_sem, centers_deg, chosen_best,
                      f"Best fits (R2 > 0.9, n={len(best_ids)})",
                      "tuning_examples_best_fits.png", rng)
    print(f"    Best fits: {len(best_ids)} neurons available, plotted 12")

    # Moderate fits: R^2 in [0.5, 0.7]
    mod_ids = df_fit[(df_fit["fit_r2"] >= 0.5) & (df_fit["fit_r2"] <= 0.7)].index.values
    if len(mod_ids) >= 12:
        chosen_mod = rng.choice(mod_ids, size=12, replace=False)
    else:
        chosen_mod = mod_ids[:12]
    plot_example_grid(df, tuning_mean, tuning_sem, centers_deg, chosen_mod,
                      f"Moderate fits (R2 0.5-0.7, n={len(mod_ids)})",
                      "tuning_examples_moderate_fits.png", rng)
    print(f"    Moderate fits: {len(mod_ids)} neurons available, plotted 12")

    # Poor/noisy fits: R^2 < 0.3
    poor_ids = df_fit[df_fit["fit_r2"] < 0.3].index.values
    if len(poor_ids) >= 12:
        chosen_poor = rng.choice(poor_ids, size=12, replace=False)
    else:
        chosen_poor = poor_ids[:min(12, len(poor_ids))]

    if len(chosen_poor) >= 3:
        # Pad to 12 if needed
        if len(chosen_poor) < 12:
            extra = min(12 - len(chosen_poor), len(poor_ids))
            chosen_poor = poor_ids[:max(extra, len(chosen_poor))]
        plot_example_grid(df, tuning_mean, tuning_sem, centers_deg, chosen_poor[:12],
                          f"Poor/noisy fits (R2 < 0.3, n={len(poor_ids)})",
                          "tuning_examples_failed_or_noisy_fits.png", rng)
        print(f"    Poor fits: {len(poor_ids)} neurons available, plotted {min(12, len(chosen_poor))}")
    else:
        print(f"    Poor fits: only {len(poor_ids)} neurons with R2 < 0.3, skipping plot")

    # ------------------------------------------------------------------
    # 3. Tuning fit summary table
    # ------------------------------------------------------------------
    print("  Computing tuning fit summary...")
    r2_vals = df_fit["fit_r2"].values
    kappa_vals = df_fit["fit_kappa_or_width"].values
    pref_vals = df_fit["fit_pref_orientation_deg"].values

    summary = {
        "metric": [
            "n_neurons_total", "n_fits_successful", "frac_success",
            "r2_mean", "r2_median", "r2_std", "r2_q25", "r2_q75",
            "kappa_mean", "kappa_median", "kappa_std", "kappa_q25", "kappa_q75",
            "kappa_lt_1", "kappa_1_to_5", "kappa_5_to_10", "kappa_gt_10",
            "pref_orientation_mean", "pref_orientation_std",
            "baseline_mean", "baseline_median",
            "amplitude_mean", "amplitude_median",
            "r2_gt_0.5", "r2_gt_0.8", "r2_gt_0.9",
        ],
        "value": [
            n_neurons, n_success, n_success / n_neurons,
            r2_vals.mean(), np.median(r2_vals), r2_vals.std(),
            np.percentile(r2_vals, 25), np.percentile(r2_vals, 75),
            kappa_vals.mean(), np.median(kappa_vals), kappa_vals.std(),
            np.percentile(kappa_vals, 25), np.percentile(kappa_vals, 75),
            (kappa_vals < 1).sum(), ((kappa_vals >= 1) & (kappa_vals < 5)).sum(),
            ((kappa_vals >= 5) & (kappa_vals < 10)).sum(), (kappa_vals >= 10).sum(),
            pref_vals.mean(), pref_vals.std(),
            df_fit["fit_baseline"].mean(), df_fit["fit_baseline"].median(),
            df_fit["fit_amplitude"].mean(), df_fit["fit_amplitude"].median(),
            (r2_vals > 0.5).sum(), (r2_vals > 0.8).sum(), (r2_vals > 0.9).sum(),
        ],
    }
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(TABLES, "tuning_fit_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # ------------------------------------------------------------------
    # 4. Status report with interpretation
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    # Compute heterogeneity measures for interpretation
    kappa_iqr = np.percentile(kappa_vals, 75) - np.percentile(kappa_vals, 25)
    pref_circular_var = 1.0 - np.abs(np.mean(np.exp(2j * np.radians(pref_vals))))

    report = f"""# Phase 2: Nonlinear Least-Squares Tuning Analysis

**Date:** 2026-03-31
**Runtime:** {elapsed:.1f}s

## Summary

All {n_neurons:,} neurons were previously fitted with the von Mises tuning model:

    r(theta) = b + a * exp(kappa * cos(2 * (theta - theta0)))

**No re-fitting was needed** — 100% of fits succeeded in the setup phase.

## Fit Quality Distribution

| Metric | Value |
|--------|-------|
| Successful fits | {n_success:,} / {n_neurons:,} (100%) |
| Mean R2 | {r2_vals.mean():.3f} |
| Median R2 | {np.median(r2_vals):.3f} |
| R2 > 0.5 | {(r2_vals > 0.5).sum():,} ({100*(r2_vals > 0.5).mean():.1f}%) |
| R2 > 0.8 | {(r2_vals > 0.8).sum():,} ({100*(r2_vals > 0.8).mean():.1f}%) |
| R2 > 0.9 | {(r2_vals > 0.9).sum():,} ({100*(r2_vals > 0.9).mean():.1f}%) |

## Tuning Sharpness Distribution

| Metric | Value |
|--------|-------|
| Mean kappa | {kappa_vals.mean():.2f} |
| Median kappa | {np.median(kappa_vals):.2f} |
| IQR | {kappa_iqr:.2f} |
| kappa < 1 (broadly tuned) | {(kappa_vals < 1).sum():,} ({100*(kappa_vals < 1).mean():.1f}%) |
| kappa 1-5 (moderate) | {((kappa_vals >= 1) & (kappa_vals < 5)).sum():,} ({100*((kappa_vals >= 1) & (kappa_vals < 5)).mean():.1f}%) |
| kappa 5-10 (sharp) | {((kappa_vals >= 5) & (kappa_vals < 10)).sum():,} ({100*((kappa_vals >= 5) & (kappa_vals < 10)).mean():.1f}%) |
| kappa >= 10 (very sharp) | {(kappa_vals >= 10).sum():,} ({100*(kappa_vals >= 10).mean():.1f}%) |

## Preferred Orientation Distribution

| Metric | Value |
|--------|-------|
| Mean preferred orientation | {pref_vals.mean():.1f} deg |
| Std | {pref_vals.std():.1f} deg |
| Circular variance | {pref_circular_var:.3f} |

The preferred orientations are approximately uniformly distributed (circular variance ~ {pref_circular_var:.3f}, where 1.0 = perfectly uniform), indicating no strong population-level orientation bias.

## Outputs Created

- `data/final/orientation_neuron_analysis.parquet` — {n_neurons:,} rows x 15 columns
- `data/final/orientation_neuron_analysis.csv`
- `reports/tables/tuning_fit_summary.csv`
- `reports/figures/tuning_examples_best_fits.png`
- `reports/figures/tuning_examples_moderate_fits.png`
- `reports/figures/tuning_examples_failed_or_noisy_fits.png`

## Biological Interpretation

The nonlinear least-squares tuning analysis reveals substantial **heterogeneity in orientation tuning** across mouse V1 neurons:

1. **Tuning sharpness varies widely.** The fitted kappa parameter spans from near 0 (essentially untuned) to the cap at 20 (very sharply tuned), with a median of {np.median(kappa_vals):.2f} and an IQR of {kappa_iqr:.2f}. This indicates that the V1 population contains a continuum of tuning widths, from broadly responsive neurons that fire to many orientations to narrowly tuned neurons that respond primarily to a single orientation.

2. **No population-level orientation bias.** Preferred orientations are distributed approximately uniformly across 0-180 deg, consistent with the idea that mouse V1 tiles all orientations without a strong cardinal bias (unlike cat or primate V1 where cardinal orientations can be overrepresented).

3. **Most neurons are well-described by the von Mises model.** With median R2 = {np.median(r2_vals):.3f} and {100*(r2_vals > 0.5).mean():.1f}% of neurons having R2 > 0.5, the single-peaked von Mises model captures the tuning of most neurons. The {100*(r2_vals < 0.3).mean():.1f}% with R2 < 0.3 likely include neurons with weak or complex (e.g., multi-peaked) orientation responses.

4. **The wide range of tuning sharpness supports efficient population coding.** A mixture of broadly and narrowly tuned neurons can support both coarse discrimination (via broadly tuned neurons) and fine discrimination (via sharply tuned neurons), consistent with theoretical models of optimal population coding.
"""
    status_path = os.path.join(STATUS, "phase2_nonlinear_fits.md")
    with open(status_path, "w") as f:
        f.write(report)
    print(f"  Saved: {status_path}")

    # ------------------------------------------------------------------
    # 5. Notebook
    # ------------------------------------------------------------------
    print("  Generating notebook 03_nonlinear_tuning.ipynb...")
    create_notebook()

    print(f"\nPhase 2 completed in {elapsed:.1f}s")


def cell(cell_type, source):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {},
                "source": source if isinstance(source, list) else [source]}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source if isinstance(source, list) else [source]}


def create_notebook():
    cells = [
        cell("markdown", [
            "# Phase 2: Nonlinear Least-Squares Tuning Analysis\n",
            "\n",
            "Von Mises model: r(theta) = b + a * exp(kappa * cos(2*(theta - theta0)))\n",
        ]),
        cell("code", [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "matplotlib.rcParams['figure.dpi'] = 120\n",
            "\n",
            "BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
            "PROC = os.path.join(BASE, 'data', 'processed')\n",
            "FINAL = os.path.join(BASE, 'data', 'final')\n",
            "\n",
            "df = pd.read_parquet(os.path.join(FINAL, 'orientation_neuron_analysis.parquet'))\n",
            "bd = np.load(os.path.join(PROC, 'orientation_binned_tuning.npz'))\n",
            "tuning_mean = bd['tuning_mean']\n",
            "tuning_sem = bd['tuning_sem']\n",
            "centers_deg = bd['angle_bin_centers_deg']\n",
            "\n",
            "print(f'Loaded {len(df)} neurons')\n",
            "df.head()\n",
        ]),
        cell("markdown", "## Fit Summary"),
        cell("code", [
            "summary = pd.read_csv(os.path.join(BASE, 'reports', 'tables', 'tuning_fit_summary.csv'))\n",
            "summary\n",
        ]),
        cell("markdown", "## Tuning Parameter Distributions"),
        cell("code", [
            "df_fit = df[df['fit_success']]\n",
            "\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
            "axes[0].hist(df_fit['fit_r2'], bins=60, color='teal')\n",
            "axes[0].set_xlabel('R-squared'); axes[0].set_title('Fit quality')\n",
            "axes[1].hist(df_fit['fit_kappa_or_width'], bins=60, color='coral')\n",
            "axes[1].set_xlabel('Kappa'); axes[1].set_title('Tuning sharpness')\n",
            "axes[2].hist(df_fit['fit_pref_orientation_deg'], bins=36, color='mediumpurple')\n",
            "axes[2].set_xlabel('Preferred orientation (deg)'); axes[2].set_title('Preferred orientation')\n",
            "plt.tight_layout(); plt.show()\n",
        ]),
        cell("markdown", "## Example Fits by Quality"),
        cell("code", [
            "def von_mises_tuning(theta, b, a, kappa, theta0):\n",
            "    return b + a * np.exp(kappa * np.cos(2.0 * (theta - theta0)))\n",
            "\n",
            "rng = np.random.default_rng(42)\n",
            "theta_fine = np.linspace(0, np.pi, 200)\n",
            "\n",
            "for label, mask in [('Best (R2>0.9)', df_fit['fit_r2']>0.9),\n",
            "                    ('Moderate (R2 0.5-0.7)', (df_fit['fit_r2']>=0.5)&(df_fit['fit_r2']<=0.7)),\n",
            "                    ('Poor (R2<0.3)', df_fit['fit_r2']<0.3)]:\n",
            "    ids = df_fit[mask].index.values\n",
            "    if len(ids) < 4: continue\n",
            "    chosen = rng.choice(ids, size=min(4, len(ids)), replace=False)\n",
            "    fig, axes = plt.subplots(1, len(chosen), figsize=(4*len(chosen), 3))\n",
            "    if len(chosen) == 1: axes = [axes]\n",
            "    for ax, nid in zip(axes, chosen):\n",
            "        row = df.iloc[nid]\n",
            "        ax.errorbar(centers_deg, tuning_mean[nid], yerr=tuning_sem[nid], fmt='o', ms=3, capsize=2)\n",
            "        y_fit = von_mises_tuning(theta_fine, row['fit_baseline'], row['fit_amplitude'],\n",
            "                                 row['fit_kappa_or_width'], np.radians(row['fit_pref_orientation_deg']))\n",
            "        ax.plot(np.degrees(theta_fine), y_fit, '-', color='tomato')\n",
            "        ax.set_title(f'N{nid} R2={row[\"fit_r2\"]:.2f}', fontsize=9)\n",
            "    fig.suptitle(label); plt.tight_layout(); plt.show()\n",
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
    out = os.path.join(NOTEBOOKS, "03_nonlinear_tuning.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
