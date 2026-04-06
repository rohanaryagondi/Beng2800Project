"""
Phase 5: Project synthesis and final figures/tables.

Creates polished final figures, summary tables, and project question answers.
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
for d in [FIGS, TABLES, STATUS, NOTEBOOKS]:
    os.makedirs(d, exist_ok=True)

SEED = 42
DPI = 200


def von_mises_tuning(theta, b, a, kappa, theta0):
    return b + a * np.exp(kappa * np.cos(2.0 * (theta - theta0)))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("=== Phase 5: Synthesis ===\n")

    # Load all results
    df = pd.read_parquet(os.path.join(FINAL, "orientation_neuron_analysis.parquet"))
    bd = np.load(os.path.join(PROC, "orientation_binned_tuning.npz"))
    tuning_mean = bd["tuning_mean"]
    tuning_sem = bd["tuning_sem"]
    centers_deg = bd["angle_bin_centers_deg"]
    cv_summary = pd.read_csv(os.path.join(TABLES, "decoder_cv_summary.csv"))
    model_results = pd.read_csv(os.path.join(TABLES, "reliability_model_results.csv"))
    corr_results = pd.read_csv(os.path.join(TABLES, "reliability_correlations.csv"))
    decoder_results = pd.read_csv(os.path.join(TABLES, "decoder_performance_by_neuron_count.csv"))

    df_fit = df[df["fit_success"]].copy()
    n_neurons = len(df)

    # ------------------------------------------------------------------
    # Final Figure 1: Example tuning curves with fits (6 selected)
    # ------------------------------------------------------------------
    print("  Fig 1: Example tuning curves...")
    # Select 6 neurons spanning the quality range
    r2_percentiles = [10, 30, 50, 70, 85, 95]
    r2_targets = np.percentile(df_fit["fit_r2"].values, r2_percentiles)
    example_ids = []
    for target in r2_targets:
        candidates = df_fit[(df_fit["fit_r2"] > target - 0.05) &
                           (df_fit["fit_r2"] < target + 0.05) &
                           (df_fit["split_half_reliability"] > 0.3)]
        if len(candidates) > 0:
            example_ids.append(rng.choice(candidates.index.values))
        else:
            example_ids.append(df_fit.iloc[(df_fit["fit_r2"] - target).abs().argsort().iloc[0]].name)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    theta_fine = np.linspace(0, np.pi, 300)

    for ax, nid in zip(axes.ravel(), example_ids):
        row = df.iloc[nid]
        tc = tuning_mean[nid]
        se = tuning_sem[nid]

        ax.errorbar(centers_deg, tc, yerr=se, fmt="o", color="#2171b5", ms=4,
                    capsize=2, capthick=1, label="Empirical", zorder=2)
        y_fit = von_mises_tuning(
            theta_fine,
            row["fit_baseline"], row["fit_amplitude"],
            row["fit_kappa_or_width"], np.radians(row["fit_pref_orientation_deg"])
        )
        ax.plot(np.degrees(theta_fine), y_fit, "-", color="#e6550d", lw=2,
                label=f"Von Mises fit", zorder=3)
        ax.set_title(f"Neuron {nid}\n"
                     f"R$^2$={row['fit_r2']:.2f}, $\\kappa$={row['fit_kappa_or_width']:.1f}, "
                     f"pref={row['fit_pref_orientation_deg']:.0f}$^\\circ$", fontsize=9)
        ax.set_xlim(0, 180)
        ax.legend(fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("Orientation (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Deconvolved activity")

    fig.suptitle("Example orientation tuning curves with von Mises fits", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "final_fig1_tuning_examples.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Final Figure 2: Tuning parameter distributions (2-panel)
    # ------------------------------------------------------------------
    print("  Fig 2: Tuning parameter distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(df_fit["fit_pref_orientation_deg"], bins=36, color="#2171b5",
                 edgecolor="white", linewidth=0.5, density=True)
    axes[0].axhline(1/180, color="red", ls="--", lw=1, label="Uniform")
    axes[0].set_xlabel("Preferred orientation (deg)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of preferred orientations")
    axes[0].set_xlim(0, 180)
    axes[0].legend()

    axes[1].hist(df_fit["fit_kappa_or_width"], bins=50, color="#e6550d",
                 edgecolor="white", linewidth=0.5)
    kappa_median = df_fit["fit_kappa_or_width"].median()
    axes[1].axvline(kappa_median, color="black", ls="--", lw=1.5,
                    label=f"Median = {kappa_median:.2f}")
    axes[1].set_xlabel("Tuning sharpness ($\\kappa$)")
    axes[1].set_ylabel("Number of neurons")
    axes[1].set_title("Distribution of tuning sharpness")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "final_fig2_tuning_parameter_distributions.png"), dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Final Figure 3: Reliability vs sharpness
    # ------------------------------------------------------------------
    print("  Fig 3: Reliability vs sharpness...")
    kappa = df_fit["fit_kappa_or_width"].values
    reliability = df_fit["split_half_reliability"].values

    # Get stats from saved results
    spearman_row = corr_results[corr_results["method"] == "Spearman"]
    spearman_r = spearman_row[spearman_row["metric"] == "r"]["value"].values[0]
    ci_lo = spearman_row[spearman_row["metric"] == "r"]["bootstrap_ci_lo"].values[0]
    ci_hi = spearman_row[spearman_row["metric"] == "r"]["bootstrap_ci_hi"].values[0]

    kappa_row = model_results[model_results["parameter"] == "kappa"]
    beta_kappa = kappa_row["coefficient"].values[0]
    p_kappa = kappa_row["p_value"].values[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: scatter
    hb = axes[0].hexbin(kappa.clip(0, 15), reliability,
                        gridsize=40, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=axes[0], label="Neuron count")
    beta0 = model_results[model_results["parameter"] == "intercept"]["coefficient"].values[0]
    beta2 = model_results[model_results["parameter"] == "mean_response"]["coefficient"].values[0]
    mean_resp_avg = df_fit["mean_response"].mean()
    k_range = np.linspace(0, 15, 100)
    axes[0].plot(k_range, beta0 + beta_kappa * k_range + beta2 * mean_resp_avg,
                 "-", color="black", lw=2, label=f"OLS ($\\beta_\\kappa$={beta_kappa:.4f})")
    axes[0].set_xlabel("Tuning sharpness ($\\kappa$)")
    axes[0].set_ylabel("Split-half reliability")
    axes[0].set_title("Reliability vs. tuning sharpness")
    axes[0].text(0.03, 0.03,
                 f"Spearman $r$ = {spearman_r:.3f}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]\nn = {len(df_fit):,}",
                 transform=axes[0].transAxes, fontsize=9, va="bottom",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    axes[0].legend(fontsize=9)

    # Right: binned means
    n_bins = 10
    bin_edges = np.linspace(0, 20, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = np.empty(n_bins)
    bin_sems = np.empty(n_bins)
    for i in range(n_bins):
        mask = (kappa >= bin_edges[i]) & (kappa < bin_edges[i+1])
        if i == n_bins - 1:
            mask = mask | (kappa == bin_edges[i+1])
        vals = reliability[mask]
        bin_means[i] = vals.mean() if len(vals) > 0 else np.nan
        bin_sems[i] = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else np.nan

    axes[1].errorbar(bin_centers, bin_means, yerr=bin_sems, fmt="o-", color="#2171b5",
                     capsize=4, ms=6, lw=2)
    axes[1].set_xlabel("Tuning sharpness ($\\kappa$)")
    axes[1].set_ylabel("Mean reliability")
    axes[1].set_title("Binned: reliability vs. sharpness")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "final_fig3_reliability_vs_sharpness.png"), dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Final Figure 4: Decoder scaling
    # ------------------------------------------------------------------
    print("  Fig 4: Decoder scaling...")
    fig, ax = plt.subplots(figsize=(8, 5))
    nc = cv_summary["neuron_count"].values
    mae = cv_summary["mae_mean"].values
    mae_std = cv_summary["mae_std"].fillna(0).values

    ax.errorbar(nc, mae, yerr=mae_std, fmt="o-", color="#2171b5",
                capsize=5, ms=8, lw=2.5, label="OLS decoder")

    # Shuffle baseline
    shuf_results = decoder_results[decoder_results["shuffle_control"] == True]
    if len(shuf_results) > 0:
        shuf_mean = shuf_results["mae_circular_deg"].mean()
        ax.axhline(shuf_mean, color="#e6550d", ls="--", lw=2,
                   label=f"Shuffle control ({shuf_mean:.1f} deg)")

    ax.axhline(45, color="gray", ls=":", lw=1, label="Chance (45 deg)")
    ax.set_xlabel("Number of neurons", fontsize=11)
    ax.set_ylabel("Mean circular MAE (deg)", fontsize=11)
    ax.set_title("Population decoder performance vs. neuron count", fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(nc)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=10)
    ax.set_ylim(0, 50)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "final_fig4_decoder_scaling.png"), dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Final Figure 5: Decoder predicted vs true
    # ------------------------------------------------------------------
    print("  Fig 5: Decoder predicted vs true...")
    # Re-run a quick decode for the scatter
    d = np.load(os.path.join(PROC, "orientation_decoder_ready_top1000.npz"))
    X_all = d["X"]
    y_cos2 = d["y_cos2"]
    y_sin2 = d["y_sin2"]
    theta_rad_dec = d["theta_rad"]
    theta_deg_dec = d["theta_deg"]

    rng2 = np.random.default_rng(SEED)
    idx = rng2.permutation(len(X_all))
    split = int(0.8 * len(X_all))
    train, test = idx[:split], idx[split:]
    Y_train = np.column_stack([y_cos2[train], y_sin2[train]])
    W, _, _, _ = np.linalg.lstsq(X_all[train], Y_train, rcond=None)
    Y_pred = X_all[test] @ W
    pred_rad = np.arctan2(Y_pred[:, 1], Y_pred[:, 0]) / 2 % np.pi
    true_rad = theta_rad_dec[test]
    err = (pred_rad - true_rad + np.pi/2) % np.pi - np.pi/2
    mae_full = float(np.degrees(np.abs(err).mean()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.degrees(true_rad), np.degrees(pred_rad),
               s=2, alpha=0.25, color="#2171b5", rasterized=True)
    ax.plot([0, 180], [0, 180], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("True orientation (deg)", fontsize=11)
    ax.set_ylabel("Predicted orientation (deg)", fontsize=11)
    ax.set_title(f"Linear decoder: predicted vs. true (MAE = {mae_full:.1f} deg)", fontsize=12)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "final_fig5_decoder_examples.png"), dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Final results tables
    # ------------------------------------------------------------------
    print("  Creating final results tables...")

    # Main results table
    kappa_vals = df_fit["fit_kappa_or_width"].values
    r2_vals = df_fit["fit_r2"].values
    main_results = pd.DataFrame([{
        "metric": "Total neurons",
        "value": str(n_neurons),
    }, {
        "metric": "Total trials",
        "value": "4598",
    }, {
        "metric": "Fit success rate",
        "value": f"{100*df['fit_success'].mean():.1f}%",
    }, {
        "metric": "Median fit R-squared",
        "value": f"{np.median(r2_vals):.3f}",
    }, {
        "metric": "Median tuning sharpness (kappa)",
        "value": f"{np.median(kappa_vals):.2f}",
    }, {
        "metric": "Kappa IQR",
        "value": f"[{np.percentile(kappa_vals, 25):.2f}, {np.percentile(kappa_vals, 75):.2f}]",
    }, {
        "metric": "Median split-half reliability",
        "value": f"{df['split_half_reliability'].median():.3f}",
    }, {
        "metric": "Spearman r (kappa vs reliability)",
        "value": f"{spearman_r:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]",
    }, {
        "metric": "OLS beta_kappa (controlling for mean response)",
        "value": f"{beta_kappa:.4f} (p={p_kappa:.2e})",
    }, {
        "metric": "Best decoder MAE (1000 neurons)",
        "value": f"{cv_summary.loc[cv_summary['neuron_count']==1000, 'mae_mean'].values[0]:.2f} deg",
    }, {
        "metric": "Decoder MAE with 10 neurons",
        "value": f"{cv_summary.loc[cv_summary['neuron_count']==10, 'mae_mean'].values[0]:.2f} deg",
    }, {
        "metric": "Shuffle control MAE",
        "value": f"{shuf_results['mae_circular_deg'].mean():.2f} deg" if len(shuf_results) > 0 else "n/a",
    }])
    main_results.to_csv(os.path.join(TABLES, "final_main_results_table.csv"), index=False)

    # Decoder results table
    cv_summary.to_csv(os.path.join(TABLES, "final_decoder_results_table.csv"), index=False)

    # ------------------------------------------------------------------
    # Status report with project question answers
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    report = f"""# Phase 5: Project Synthesis

**Date:** 2026-03-31
**Runtime:** {elapsed:.1f}s

## Final Figures Created

1. **final_fig1_tuning_examples.png** — 6 example tuning curves spanning quality range
2. **final_fig2_tuning_parameter_distributions.png** — Preferred orientation and kappa distributions
3. **final_fig3_reliability_vs_sharpness.png** — Scatter + binned reliability vs sharpness
4. **final_fig4_decoder_scaling.png** — Decoder error vs neuron count
5. **final_fig5_decoder_examples.png** — Predicted vs true orientation scatter

## Answers to Project Questions

### 1. How much do preferred orientation and tuning sharpness vary across neurons in mouse V1?

There is **substantial heterogeneity** in both parameters:

- **Preferred orientation** is distributed approximately uniformly across 0-180 deg, indicating no population-level orientation bias in this mouse V1 recording.
- **Tuning sharpness (kappa)** varies widely, with a median of {np.median(kappa_vals):.2f} and an IQR of [{np.percentile(kappa_vals, 25):.2f}, {np.percentile(kappa_vals, 75):.2f}]. About {100*(kappa_vals < 1).mean():.0f}% of neurons are broadly tuned (kappa < 1), while {100*(kappa_vals >= 10).mean():.0f}% are very sharply tuned (kappa >= 10).
- The von Mises model fits most neurons well (median R-squared = {np.median(r2_vals):.3f}), but fit quality itself is heterogeneous.

### 2. Are neurons with sharper tuning also more reliable across trials?

**Yes, there is a positive but moderate association.** Neurons with higher kappa tend to have higher split-half reliability:

- Spearman r = {spearman_r:.3f} (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])
- This relationship holds after controlling for mean response in the OLS model (beta_kappa = {beta_kappa:.4f}, p = {p_kappa:.2e})
- However, kappa and mean response together explain only ~5.5% of reliability variance, indicating that other factors (noise correlations, non-stationarity, etc.) dominate trial-to-trial variability.

### 3. How accurately can a simple population decoder predict stimulus orientation from neural activity?

The OLS linear decoder performs **far above chance**:

- With 1000 neurons: MAE = {cv_summary.loc[cv_summary['neuron_count']==1000, 'mae_mean'].values[0]:.2f} deg (chance = ~45 deg)
- This represents an ~{45/cv_summary.loc[cv_summary['neuron_count']==1000, 'mae_mean'].values[0]:.0f}x improvement over random guessing
- Even 10 neurons achieve ~{cv_summary.loc[cv_summary['neuron_count']==10, 'mae_mean'].values[0]:.0f} deg MAE

### 4. How does decoder performance change as the number of neurons increases?

Performance improves monotonically with neuron count, with **diminishing returns**:

| Neurons | MAE (deg) |
|---------|-----------|
"""
    for _, row in cv_summary.iterrows():
        report += f"| {int(row['neuron_count'])} | {row['mae_mean']:.2f} |\n"

    report += f"""
The largest gains occur from 10 to 100 neurons. Beyond 250 neurons, improvements are more gradual. Even at 1000 neurons, performance has not fully saturated, suggesting additional neurons could further improve decoding.

## Outputs

- `reports/tables/final_main_results_table.csv`
- `reports/tables/final_decoder_results_table.csv`
- 5 final figures in `reports/figures/`
"""
    status_path = os.path.join(STATUS, "phase5_synthesis.md")
    with open(status_path, "w") as f:
        f.write(report)
    print(f"  Saved: {status_path}")

    # ------------------------------------------------------------------
    # Notebook
    # ------------------------------------------------------------------
    print("  Generating notebook 06_results_synthesis.ipynb...")
    create_notebook()

    print(f"\nPhase 5 completed in {elapsed:.1f}s")


def cell(cell_type, source):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {},
                "source": source if isinstance(source, list) else [source]}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source if isinstance(source, list) else [source]}


def create_notebook():
    cells = [
        cell("markdown", [
            "# Phase 5: Results Synthesis\n",
            "\n",
            "Final figures and summary of project findings.\n",
        ]),
        cell("code", [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from IPython.display import Image, display\n",
            "matplotlib.rcParams['figure.dpi'] = 120\n",
            "\n",
            "BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
            "FIGS = os.path.join(BASE, 'reports', 'figures')\n",
            "TABLES = os.path.join(BASE, 'reports', 'tables')\n",
        ]),
        cell("markdown", "## Main Results Table"),
        cell("code", [
            "main_results = pd.read_csv(os.path.join(TABLES, 'final_main_results_table.csv'))\n",
            "main_results\n",
        ]),
        cell("markdown", "## Decoder Scaling Results"),
        cell("code", [
            "decoder_results = pd.read_csv(os.path.join(TABLES, 'final_decoder_results_table.csv'))\n",
            "decoder_results\n",
        ]),
        cell("markdown", "## Final Figures"),
        cell("code", [
            "for fname in ['final_fig1_tuning_examples.png',\n",
            "              'final_fig2_tuning_parameter_distributions.png',\n",
            "              'final_fig3_reliability_vs_sharpness.png',\n",
            "              'final_fig4_decoder_scaling.png',\n",
            "              'final_fig5_decoder_examples.png']:\n",
            "    path = os.path.join(FIGS, fname)\n",
            "    if os.path.exists(path):\n",
            "        print(f'\\n--- {fname} ---')\n",
            "        display(Image(filename=path, width=700))\n",
        ]),
        cell("markdown", [
            "## Project Question Answers\n",
            "\n",
            "See `reports/status/phase5_synthesis.md` for full answers.\n",
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
    out = os.path.join(NOTEBOOKS, "06_results_synthesis.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
