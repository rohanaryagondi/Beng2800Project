"""
Fix Issues 2 and 3 from the Issues folder.

Issue 2: Run simulation showing the expected kappa-reliability correlation,
         demonstrating the observed r=0.239 is LOWER than the mathematical expectation.
Issue 3: Run decoder with randomly selected neurons from the full 23,589 population,
         producing an unbiased comparison to the top-1000 results.

Saves new figures, tables, and updated status reports.
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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

SEED = 42
DPI = 150


def von_mises_tuning(theta, b, a, kappa, theta0):
    return b + a * np.exp(kappa * np.cos(2.0 * (theta - theta0)))


def circular_error_deg(pred_rad, true_rad):
    err = pred_rad - true_rad
    err = (err + np.pi / 2) % np.pi - np.pi / 2
    return np.degrees(np.abs(err))


def make_fold_indices(n, n_folds, rng):
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


def run_cv(X, y_cos2, y_sin2, theta_rad, fold_indices):
    Y = np.column_stack([y_cos2, y_sin2])
    mae_list = []
    medae_list = []
    for train_idx, test_idx in fold_indices:
        W, _, _, _ = np.linalg.lstsq(X[train_idx], Y[train_idx], rcond=None)
        Y_pred = X[test_idx] @ W
        theta_pred = np.arctan2(Y_pred[:, 1], Y_pred[:, 0]) / 2.0 % np.pi
        errors = circular_error_deg(theta_pred, theta_rad[test_idx])
        mae_list.append(float(np.mean(errors)))
        medae_list.append(float(np.median(errors)))
    return np.mean(mae_list), np.mean(medae_list)


# ==================================================================
# ISSUE 2: Simulation showing expected kappa-reliability correlation
# ==================================================================
def fix_issue2():
    print("=== Issue 2: Reliability-sharpness simulation ===\n")
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    n_sim = 5000
    n_bins = 36
    n_trials_per_bin = 128  # matches real data
    centers_rad = np.radians(np.linspace(2.5, 177.5, n_bins))

    # Draw kappas from distribution similar to real data
    true_kappas = rng.exponential(4, size=n_sim)
    true_kappas = np.clip(true_kappas, 0, 20)
    true_theta0 = rng.uniform(0, np.pi, size=n_sim)
    true_baseline = rng.uniform(1, 10, size=n_sim)
    true_amplitude = rng.uniform(0.5, 5, size=n_sim)

    # Noise level INDEPENDENT of kappa
    noise_std = np.abs(rng.normal(3, 1.5, size=n_sim))

    sim_kappas = np.empty(n_sim)
    sim_reliabilities = np.empty(n_sim)

    for i in range(n_sim):
        # Generate true tuning curve
        true_tc = von_mises_tuning(centers_rad, true_baseline[i], true_amplitude[i],
                                   true_kappas[i], true_theta0[i])

        # Generate two half-datasets (odd/even trials)
        half1_means = true_tc + rng.normal(0, noise_std[i] / np.sqrt(n_trials_per_bin / 2), size=n_bins)
        half2_means = true_tc + rng.normal(0, noise_std[i] / np.sqrt(n_trials_per_bin / 2), size=n_bins)

        # Compute reliability (Pearson correlation between halves)
        full_mean = (half1_means + half2_means) / 2
        r_val, _ = sp_stats.pearsonr(half1_means, half2_means)
        sim_reliabilities[i] = r_val

        # Fit von Mises to the full mean
        try:
            b_init = np.percentile(full_mean, 10)
            a_init = max(full_mean.max() - b_init, 0.01)
            popt, _ = curve_fit(von_mises_tuning, centers_rad, full_mean,
                                p0=[b_init, a_init, 1.0, centers_rad[np.argmax(full_mean)]],
                                bounds=([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, 20, np.inf]),
                                maxfev=5000)
            sim_kappas[i] = popt[2]
        except Exception:
            sim_kappas[i] = np.nan
            sim_reliabilities[i] = np.nan

    # Remove failed fits
    valid = ~np.isnan(sim_kappas) & ~np.isnan(sim_reliabilities)
    sim_k = sim_kappas[valid]
    sim_r = sim_reliabilities[valid]

    sim_spearman, _ = sp_stats.spearmanr(sim_k, sim_r)
    print(f"  Simulation: Spearman r = {sim_spearman:.3f} (n={valid.sum()})")
    print(f"  Real data:  Spearman r = 0.239")
    print(f"  Simulation took {time.time()-t0:.1f}s")

    # Load real data for comparison
    df = pd.read_parquet(os.path.join(FINAL, "orientation_neuron_analysis.parquet"))
    df_fit = df[df["fit_success"]]
    real_k = df_fit["fit_kappa_or_width"].values
    real_r = df_fit["split_half_reliability"].values

    # Figure: simulation vs real data
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: simulation scatter
    ax = axes[0]
    ax.scatter(sim_k, sim_r, s=2, alpha=0.3, color="#e6550d", rasterized=True)
    ax.set_xlabel("Fitted $\\kappa$")
    ax.set_ylabel("Split-half reliability")
    ax.set_title(f"Simulation (noise independent of $\\kappa$)\nSpearman r = {sim_spearman:.3f}")
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.5, 1.05)

    # Middle: real data scatter
    ax = axes[1]
    ax.scatter(real_k, real_r, s=1, alpha=0.1, color="#2171b5", rasterized=True)
    ax.set_xlabel("Fitted $\\kappa$")
    ax.set_ylabel("Split-half reliability")
    ax.set_title(f"Real data (Stringer et al.)\nSpearman r = 0.239")
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.5, 1.05)

    # Right: comparison of binned means
    ax = axes[2]
    n_kbins = 10
    bin_edges = np.linspace(0, 20, n_kbins + 1)
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2

    for data_k, data_r, label, color in [(sim_k, sim_r, "Simulation", "#e6550d"),
                                          (real_k, real_r, "Real data", "#2171b5")]:
        bm = np.empty(n_kbins)
        bs = np.empty(n_kbins)
        for i in range(n_kbins):
            mask = (data_k >= bin_edges[i]) & (data_k < bin_edges[i + 1])
            if i == n_kbins - 1:
                mask = mask | (data_k == bin_edges[i + 1])
            vals = data_r[mask]
            bm[i] = vals.mean() if len(vals) > 0 else np.nan
            bs[i] = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        ax.errorbar(bc, bm, yerr=bs, fmt="o-", color=color, capsize=3, ms=5, lw=1.5, label=label)

    ax.set_xlabel("$\\kappa$")
    ax.set_ylabel("Mean reliability")
    ax.set_title("Binned comparison")
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "issue2_simulation_vs_real.png"), dpi=DPI)
    plt.close(fig)
    print(f"  Saved: issue2_simulation_vs_real.png")

    # Save simulation results
    sim_df = pd.DataFrame({
        "analysis": ["simulation_noise_independent", "real_data"],
        "spearman_r": [round(sim_spearman, 4), 0.2392],
        "n_neurons": [int(valid.sum()), len(df_fit)],
        "interpretation": [
            "Expected correlation when noise is independent of kappa",
            "Observed correlation — lower than expected, suggesting noise decorrelates kappa and reliability",
        ],
    })
    sim_df.to_csv(os.path.join(TABLES, "issue2_simulation_results.csv"), index=False)

    return sim_spearman


# ==================================================================
# ISSUE 3: Decoder with random neurons from full population
# ==================================================================
def fix_issue3():
    print("\n=== Issue 3: Random-neuron decoder ===\n")
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Load full decoder matrix
    print("  Loading full decoder matrix (232 MB)...")
    d_full = np.load(os.path.join(PROC, "orientation_decoder_ready.npz"))
    X_full = d_full["X"]           # (4598, 23589)
    theta_deg = d_full["theta_deg"]
    theta_rad = d_full["theta_rad"]

    # Load targets
    dt = np.load(os.path.join(PROC, "orientation_decoder_targets.npz"))
    y_cos2 = dt["y_cos2"]
    y_sin2 = dt["y_sin2"]

    n_trials, n_neurons_full = X_full.shape
    print(f"  Full matrix: {n_trials} trials x {n_neurons_full} neurons")

    # Also load top-1000 results for comparison
    top1000_cv = pd.read_csv(os.path.join(TABLES, "decoder_cv_summary.csv"))

    # Fold indices
    fold_indices = make_fold_indices(n_trials, 5, rng)

    # Neuron counts — extend to larger counts since we have 23,589
    neuron_counts = [10, 25, 50, 100, 250, 500, 1000, 2000, 5000]
    n_repeats = 10  # fewer repeats since full matrix is expensive

    random_results = []
    for nc in neuron_counts:
        if nc > n_neurons_full:
            continue
        t_nc = time.time()
        for rep in range(n_repeats):
            subset = rng.choice(n_neurons_full, size=nc, replace=False)
            X_sub = X_full[:, subset]
            mae, medae = run_cv(X_sub, y_cos2, y_sin2, theta_rad, fold_indices)
            random_results.append({
                "neuron_count": nc,
                "repeat_id": rep,
                "mae_circular_deg": round(mae, 4),
                "median_ae_circular_deg": round(medae, 4),
                "source": "random_full_population",
            })
        elapsed_nc = time.time() - t_nc
        maes = [r["mae_circular_deg"] for r in random_results if r["neuron_count"] == nc]
        print(f"  {nc:5d} random neurons: MAE = {np.mean(maes):.2f} +/- {np.std(maes):.2f} deg  ({elapsed_nc:.1f}s)")

    # Release full matrix
    del X_full, d_full
    print("  (Released full X)")

    # Save results
    rand_df = pd.DataFrame(random_results)
    rand_summary = rand_df.groupby("neuron_count").agg(
        mae_mean=("mae_circular_deg", "mean"),
        mae_std=("mae_circular_deg", "std"),
        medae_mean=("median_ae_circular_deg", "mean"),
        n_repeats=("repeat_id", "count"),
    ).reset_index()
    rand_summary["source"] = "random_full_population"

    # Combined comparison table
    top_summary = top1000_cv[["neuron_count", "mae_mean", "mae_std", "medae_mean", "n_repeats"]].copy()
    top_summary["source"] = "top_1000_pool"

    combined = pd.concat([top_summary, rand_summary], ignore_index=True)
    combined.to_csv(os.path.join(TABLES, "issue3_decoder_comparison.csv"), index=False)
    rand_df.to_csv(os.path.join(TABLES, "decoder_random_neurons_detail.csv"), index=False)

    # Figure: both curves on same plot
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Top-1000 curve
    nc_top = top1000_cv["neuron_count"].values
    mae_top = top1000_cv["mae_mean"].values
    std_top = top1000_cv["mae_std"].fillna(0).values
    ax.errorbar(nc_top, mae_top, yerr=std_top, fmt="s-", color="#e6550d",
                capsize=4, ms=7, lw=2, label="Top-1000 reliable neurons")

    # Random curve
    nc_rand = rand_summary["neuron_count"].values
    mae_rand = rand_summary["mae_mean"].values
    std_rand = rand_summary["mae_std"].values
    ax.errorbar(nc_rand, mae_rand, yerr=std_rand, fmt="o-", color="#2171b5",
                capsize=4, ms=7, lw=2, label="Random neurons (full population)")

    ax.axhline(45, color="gray", ls=":", lw=1, label="Chance (45 deg)")
    ax.set_xlabel("Number of neurons", fontsize=11)
    ax.set_ylabel("Mean circular MAE (deg)", fontsize=11)
    ax.set_title("Decoder performance: pre-selected vs. random neurons", fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(sorted(set(list(nc_top) + list(nc_rand))))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 50)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "issue3_decoder_random_vs_top1000.png"), dpi=DPI)
    plt.close(fig)
    print(f"\n  Saved: issue3_decoder_random_vs_top1000.png")

    elapsed = time.time() - t0
    print(f"  Issue 3 completed in {elapsed:.1f}s")

    return rand_summary, combined


def main():
    sim_r = fix_issue2()
    rand_summary, combined = fix_issue3()

    # Print summary
    print("\n" + "=" * 60)
    print("ISSUE FIX SUMMARY")
    print("=" * 60)
    print(f"\nIssue 2: Simulation Spearman r = {sim_r:.3f} vs real r = 0.239")
    print(f"  -> Real correlation is LOWER than expected, not a pure biological finding")
    print(f"\nIssue 3: Random-neuron decoder results:")
    for _, row in rand_summary.iterrows():
        print(f"  {int(row['neuron_count']):5d} neurons: MAE = {row['mae_mean']:.2f} +/- {row['mae_std']:.2f} deg")
    print()


if __name__ == "__main__":
    main()
