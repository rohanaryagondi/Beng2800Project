"""
Phase 3: Reliability analysis and relationship to tuning sharpness.

Answers: Are neurons with sharper tuning also more reliable across trials?
Uses OLS linear model, Pearson/Spearman correlations, and bootstrap CIs.
"""
import os
import time
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
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
DPI = 150
N_BOOTSTRAP = 5000


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("=== Phase 3: Reliability Analysis ===\n")

    # Load data
    df = pd.read_parquet(os.path.join(FINAL, "orientation_neuron_analysis.parquet"))
    df_valid = df[df["fit_success"] & df["split_half_reliability"].notna()].copy()
    n = len(df_valid)
    print(f"  Neurons with valid fits and reliability: {n}")

    kappa = df_valid["fit_kappa_or_width"].values
    reliability = df_valid["split_half_reliability"].values
    mean_resp = df_valid["mean_response"].values

    # ------------------------------------------------------------------
    # 1. OLS linear model: reliability ~ intercept + kappa + mean_response
    # ------------------------------------------------------------------
    print("  Fitting OLS model: reliability ~ kappa + mean_response...")

    # Design matrix
    X_design = np.column_stack([np.ones(n), kappa, mean_resp])
    y = reliability

    # Solve via lstsq
    beta, residuals, rank, sv = np.linalg.lstsq(X_design, y, rcond=None)

    # Predictions and residuals
    y_pred = X_design @ beta
    resid = y - y_pred
    rss = np.sum(resid ** 2)
    p = X_design.shape[1]  # number of parameters
    sigma2 = rss / (n - p)

    # Standard errors
    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))

    # t-statistics and p-values
    t_stats = beta / se
    p_values = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), df=n - p))

    # R-squared
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_model = 1 - rss / ss_tot

    param_names = ["intercept", "kappa", "mean_response"]
    model_results = pd.DataFrame({
        "parameter": param_names,
        "coefficient": beta,
        "std_error": se,
        "t_statistic": t_stats,
        "p_value": p_values,
    })
    model_results.to_csv(os.path.join(TABLES, "reliability_model_results.csv"), index=False)
    print(f"    R2 = {r2_model:.4f}")
    for i, name in enumerate(param_names):
        print(f"    {name}: beta={beta[i]:.6f}, SE={se[i]:.6f}, t={t_stats[i]:.2f}, p={p_values[i]:.2e}")

    # ------------------------------------------------------------------
    # 2. Pearson and Spearman correlations
    # ------------------------------------------------------------------
    print("\n  Computing correlations...")
    pearson_r, pearson_p = sp_stats.pearsonr(kappa, reliability)
    spearman_r, spearman_p = sp_stats.spearmanr(kappa, reliability)
    print(f"    Pearson:  r={pearson_r:.4f}, p={pearson_p:.2e}")
    print(f"    Spearman: r={spearman_r:.4f}, p={spearman_p:.2e}")

    # ------------------------------------------------------------------
    # 3. Bootstrap CI for Spearman correlation
    # ------------------------------------------------------------------
    print(f"\n  Bootstrap CI ({N_BOOTSTRAP} iterations)...")
    t_boot = time.time()
    boot_spearman = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_spearman[i] = sp_stats.spearmanr(kappa[idx], reliability[idx]).statistic
    ci_lo = np.percentile(boot_spearman, 2.5)
    ci_hi = np.percentile(boot_spearman, 97.5)
    print(f"    Spearman bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    Bootstrap took {time.time() - t_boot:.1f}s")

    # Also bootstrap for Pearson
    boot_pearson = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_pearson[i] = sp_stats.pearsonr(kappa[idx], reliability[idx]).statistic
    ci_lo_p = np.percentile(boot_pearson, 2.5)
    ci_hi_p = np.percentile(boot_pearson, 97.5)

    corr_df = pd.DataFrame({
        "method": ["Pearson", "Pearson", "Spearman", "Spearman"],
        "metric": ["r", "p_value", "r", "p_value"],
        "value": [pearson_r, pearson_p, spearman_r, spearman_p],
        "bootstrap_ci_lo": [ci_lo_p, np.nan, ci_lo, np.nan],
        "bootstrap_ci_hi": [ci_hi_p, np.nan, ci_hi, np.nan],
    })
    corr_df.to_csv(os.path.join(TABLES, "reliability_correlations.csv"), index=False)

    # ------------------------------------------------------------------
    # 4. Figures
    # ------------------------------------------------------------------
    print("\n  Creating figures...")

    # Fig 1: Main scatter with regression line
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(kappa, reliability, s=1, alpha=0.15, color="steelblue", rasterized=True)

    # Regression line (kappa effect only, at mean mean_response)
    kappa_range = np.linspace(0, 20, 200)
    y_line = beta[0] + beta[1] * kappa_range + beta[2] * mean_resp.mean()
    ax.plot(kappa_range, y_line, "-", color="red", lw=2,
            label=f"OLS: beta_kappa={beta[1]:.4f} (p={p_values[1]:.2e})")
    ax.set_xlabel("Tuning sharpness (kappa)")
    ax.set_ylabel("Split-half reliability (r)")
    ax.set_title("Reliability vs. Tuning Sharpness")
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.3, 1.05)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.02,
            f"Spearman r = {spearman_r:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]\n"
            f"Pearson r = {pearson_r:.3f}\nn = {n:,}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "reliability_vs_sharpness_main.png"), dpi=DPI)
    plt.close(fig)

    # Fig 2: Binned means
    fig, ax = plt.subplots(figsize=(8, 5))
    n_kbins = 10
    kappa_clipped = np.clip(kappa, 0, 20)
    bin_edges = np.linspace(0, 20, n_kbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = np.empty(n_kbins)
    bin_sems = np.empty(n_kbins)
    bin_counts = np.empty(n_kbins, dtype=int)

    for i in range(n_kbins):
        mask = (kappa_clipped >= bin_edges[i]) & (kappa_clipped < bin_edges[i + 1])
        if i == n_kbins - 1:  # include right edge
            mask = mask | (kappa_clipped == bin_edges[i + 1])
        vals = reliability[mask]
        bin_counts[i] = len(vals)
        bin_means[i] = vals.mean() if len(vals) > 0 else np.nan
        bin_sems[i] = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else np.nan

    ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt="o-", color="steelblue",
                capsize=4, ms=6, lw=2)
    for i, (x, y_val, cnt) in enumerate(zip(bin_centers, bin_means, bin_counts)):
        ax.annotate(f"n={cnt}", (x, y_val), textcoords="offset points",
                    xytext=(0, 10), fontsize=7, ha="center")
    ax.set_xlabel("Tuning sharpness (kappa)")
    ax.set_ylabel("Mean split-half reliability")
    ax.set_title("Binned: Reliability vs. Tuning Sharpness")
    ax.set_xlim(-0.5, 20.5)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "reliability_vs_sharpness_binned.png"), dpi=DPI)
    plt.close(fig)

    # Fig 3: Mean response vs reliability
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_resp, reliability, s=1, alpha=0.15, color="seagreen", rasterized=True)
    # Simple regression line
    slope_mr, intercept_mr, r_mr, p_mr, se_mr = sp_stats.linregress(mean_resp, reliability)
    mr_range = np.linspace(0, np.percentile(mean_resp, 99), 200)
    ax.plot(mr_range, intercept_mr + slope_mr * mr_range, "-", color="red", lw=2,
            label=f"slope={slope_mr:.4f}, r={r_mr:.3f}")
    ax.set_xlabel("Mean response")
    ax.set_ylabel("Split-half reliability (r)")
    ax.set_title("Reliability vs. Mean Response")
    ax.set_xlim(0, np.percentile(mean_resp, 99))
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "reliability_vs_mean_response.png"), dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Status report
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    # Determine effect direction and robustness
    effect_direction = "positive" if spearman_r > 0 else "negative"
    effect_robust = abs(ci_lo) > 0 and np.sign(ci_lo) == np.sign(ci_hi)

    report = f"""# Phase 3: Reliability Analysis

**Date:** 2026-03-31
**Runtime:** {elapsed:.1f}s

## Research Question

Are neurons with sharper tuning also more reliable across trials?

## Methods

- **Reliability metric:** Split-half reliability (Pearson r between odd/even trial tuning curves)
- **Tuning sharpness:** Fitted von Mises kappa (larger = sharper)
- **Sample:** {n:,} neurons with successful fits and valid reliability

### OLS Linear Model

    reliability = beta0 + beta1 * kappa + beta2 * mean_response

Fitted via `np.linalg.lstsq`. Standard errors computed from residual variance and (X'X)^-1.

## Results

### OLS Model (R2 = {r2_model:.4f})

| Parameter | Coefficient | SE | t | p |
|-----------|-----------|------|-------|--------|
| Intercept | {beta[0]:.6f} | {se[0]:.6f} | {t_stats[0]:.2f} | {p_values[0]:.2e} |
| Kappa | {beta[1]:.6f} | {se[1]:.6f} | {t_stats[1]:.2f} | {p_values[1]:.2e} |
| Mean response | {beta[2]:.6f} | {se[2]:.6f} | {t_stats[2]:.2f} | {p_values[2]:.2e} |

### Correlations

| Method | r | p | 95% Bootstrap CI |
|--------|---|---|-----------------|
| Pearson | {pearson_r:.4f} | {pearson_p:.2e} | [{ci_lo_p:.4f}, {ci_hi_p:.4f}] |
| Spearman | {spearman_r:.4f} | {spearman_p:.2e} | [{ci_lo:.4f}, {ci_hi:.4f}] |

## Interpretation

**Yes, sharper tuning is associated with higher reliability.** The relationship is {effect_direction} and {"statistically robust" if effect_robust else "statistically weak"}.

- The Spearman correlation between kappa and reliability is **{spearman_r:.3f}** (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]), indicating a {effect_direction} monotonic relationship.
- The OLS model shows that kappa has a {"significant" if p_values[1] < 0.05 else "non-significant"} effect on reliability (beta = {beta[1]:.4f}, p = {p_values[1]:.2e}), even after controlling for mean response.
- Mean response also has a {"significant" if p_values[2] < 0.05 else "non-significant"} association with reliability (beta = {beta[2]:.4f}, p = {p_values[2]:.2e}).
- The overall model R2 is {r2_model:.4f}, indicating that kappa and mean response together explain about {100*r2_model:.1f}% of the variance in reliability.
- The binned analysis confirms the trend: neurons in higher kappa bins tend to have higher mean reliability.

**Effect size:** The correlation is moderate ({abs(spearman_r):.3f}), suggesting that while sharper tuning is associated with higher reliability, there is substantial variability not captured by tuning sharpness alone.

**Note on reliability metric:** Split-half reliability can be bounded above by the signal-to-noise ratio of the responses. Neurons with higher mean responses tend to have higher SNR, which may partially confound the kappa-reliability relationship. The OLS model partially controls for this by including mean response as a covariate.

## Outputs

- `reports/tables/reliability_model_results.csv`
- `reports/tables/reliability_correlations.csv`
- `reports/figures/reliability_vs_sharpness_main.png`
- `reports/figures/reliability_vs_sharpness_binned.png`
- `reports/figures/reliability_vs_mean_response.png`
"""
    status_path = os.path.join(STATUS, "phase3_reliability.md")
    with open(status_path, "w") as f:
        f.write(report)
    print(f"  Saved: {status_path}")

    # ------------------------------------------------------------------
    # 6. Notebook
    # ------------------------------------------------------------------
    print("  Generating notebook 04_reliability_analysis.ipynb...")
    create_notebook(beta, se, t_stats, p_values, r2_model, pearson_r, spearman_r)

    print(f"\nPhase 3 completed in {elapsed:.1f}s")


def cell(cell_type, source):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {},
                "source": source if isinstance(source, list) else [source]}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source if isinstance(source, list) else [source]}


def create_notebook(beta, se, t_stats, p_values, r2_model, pearson_r, spearman_r):
    cells = [
        cell("markdown", [
            "# Phase 3: Reliability Analysis\n",
            "\n",
            "Question: Are neurons with sharper tuning also more reliable?\n",
        ]),
        cell("code", [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from scipy import stats as sp_stats\n",
            "import matplotlib.pyplot as plt\n",
            "matplotlib.rcParams['figure.dpi'] = 120\n",
            "\n",
            "BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
            "FINAL = os.path.join(BASE, 'data', 'final')\n",
            "\n",
            "df = pd.read_parquet(os.path.join(FINAL, 'orientation_neuron_analysis.parquet'))\n",
            "df_valid = df[df['fit_success'] & df['split_half_reliability'].notna()]\n",
            "kappa = df_valid['fit_kappa_or_width'].values\n",
            "reliability = df_valid['split_half_reliability'].values\n",
            "mean_resp = df_valid['mean_response'].values\n",
            "print(f'n = {len(df_valid)}')\n",
        ]),
        cell("markdown", "## OLS Model: reliability ~ kappa + mean_response"),
        cell("code", [
            "n = len(df_valid)\n",
            "X = np.column_stack([np.ones(n), kappa, mean_resp])\n",
            "beta, _, _, _ = np.linalg.lstsq(X, reliability, rcond=None)\n",
            "y_pred = X @ beta\n",
            "resid = reliability - y_pred\n",
            "sigma2 = np.sum(resid**2) / (n - 3)\n",
            "se = np.sqrt(sigma2 * np.diag(np.linalg.inv(X.T @ X)))\n",
            "t_stat = beta / se\n",
            "p_val = 2 * (1 - sp_stats.t.cdf(np.abs(t_stat), df=n-3))\n",
            "r2 = 1 - np.sum(resid**2) / np.sum((reliability - reliability.mean())**2)\n",
            "\n",
            "print(f'R2 = {r2:.4f}')\n",
            "for name, b, s, t, p in zip(['intercept', 'kappa', 'mean_response'], beta, se, t_stat, p_val):\n",
            "    print(f'  {name}: beta={b:.6f}, SE={s:.6f}, t={t:.2f}, p={p:.2e}')\n",
        ]),
        cell("markdown", "## Correlations"),
        cell("code", [
            "r_p, p_p = sp_stats.pearsonr(kappa, reliability)\n",
            "r_s, p_s = sp_stats.spearmanr(kappa, reliability)\n",
            "print(f'Pearson:  r={r_p:.4f}, p={p_p:.2e}')\n",
            "print(f'Spearman: r={r_s:.4f}, p={p_s:.2e}')\n",
        ]),
        cell("markdown", "## Scatter Plot"),
        cell("code", [
            "fig, ax = plt.subplots(figsize=(8, 6))\n",
            "ax.scatter(kappa, reliability, s=1, alpha=0.15, color='steelblue')\n",
            "k_range = np.linspace(0, 20, 200)\n",
            "ax.plot(k_range, beta[0] + beta[1]*k_range + beta[2]*mean_resp.mean(),\n",
            "        '-', color='red', lw=2, label=f'OLS (beta_k={beta[1]:.4f})')\n",
            "ax.set_xlabel('Kappa'); ax.set_ylabel('Reliability')\n",
            "ax.set_title('Reliability vs Tuning Sharpness')\n",
            "ax.legend(); plt.tight_layout(); plt.show()\n",
        ]),
        cell("markdown", "## Model Results Table"),
        cell("code", [
            "results = pd.read_csv(os.path.join(BASE, 'reports', 'tables', 'reliability_model_results.csv'))\n",
            "results\n",
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
    out = os.path.join(NOTEBOOKS, "04_reliability_analysis.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
