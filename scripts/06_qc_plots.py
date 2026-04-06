"""
Generate QC figures:
  1. Histogram of mean neuron responses
  2. Histogram of split-half reliability
  3. Empirical tuning curves for 12 random neurons
  4. Fitted tuning curves for the same 12 neurons
  5. Distribution of preferred orientations

Saves to reports/figures/.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
FIG_DIR = os.path.join(BASE, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
N_EXAMPLE = 12


def load_data():
    df = pd.read_parquet(os.path.join(PROC, "orientation_neuron_summary.parquet"))
    bd = np.load(os.path.join(PROC, "orientation_binned_tuning.npz"))
    return df, bd


def plot_mean_response_hist(df, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["mean_response"], bins=100, color="steelblue", edgecolor="none")
    ax.set_xlabel("Mean response (deconvolved activity)")
    ax.set_ylabel("Number of neurons")
    ax.set_title(f"Mean Neuron Response Distribution\n(n={len(df):,} neurons)")
    ax.axvline(df["mean_response"].median(), color="red", lw=1.5, linestyle="--",
               label=f"Median = {df['mean_response'].median():.3f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_reliability_hist(df, save_path):
    rel = df["split_half_reliability"].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rel, bins=80, color="darkorange", edgecolor="none")
    ax.set_xlabel("Split-half reliability (Pearson r)")
    ax.set_ylabel("Number of neurons")
    ax.set_title(f"Neuron Reliability Distribution\n(n={len(rel):,} neurons)")
    ax.axvline(rel.median(), color="red", lw=1.5, linestyle="--",
               label=f"Median = {rel.median():.3f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_example_tuning_curves(df, bd, example_ids, save_path, show_fit=False):
    tuning_mean = bd["tuning_mean"]   # (n_neurons, n_bins)
    tuning_sem = bd["tuning_sem"]
    centers_deg = bd["angle_bin_centers_deg"]

    n = len(example_ids)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for ax_i, nid in enumerate(example_ids):
        ax = axes[ax_i]
        tc = tuning_mean[nid]
        se = tuning_sem[nid]
        ax.plot(centers_deg, tc, "o-", color="steelblue", ms=4, lw=1.5, label="Empirical")
        ax.fill_between(centers_deg, tc - se, tc + se, alpha=0.2, color="steelblue")

        if show_fit:
            row = df.iloc[nid]
            if row["fit_success"]:
                from scripts_helper import von_mises_tuning  # inline below
                pass

        ax.set_title(f"Neuron {nid}\npref={df.iloc[nid]['pref_orientation_deg_empirical']:.0f}°  "
                     f"OSI={df.iloc[nid]['osi_empirical']:.2f}", fontsize=8)
        ax.set_xlabel("Orientation (°)", fontsize=7)
        ax.set_ylabel("Response", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax_i in range(n, len(axes)):
        axes[ax_i].set_visible(False)

    title = "Fitted Tuning Curves" if show_fit else "Empirical Tuning Curves"
    fig.suptitle(f"{title} — 12 Example Neurons", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def von_mises_tuning(theta, b, a, kappa, theta0):
    return b + a * np.exp(kappa * np.cos(2.0 * (theta - theta0)))


def plot_empirical_and_fitted(df, bd, example_ids, save_path):
    tuning_mean = bd["tuning_mean"]
    tuning_sem = bd["tuning_sem"]
    centers_deg = bd["angle_bin_centers_deg"]
    centers_rad = np.radians(centers_deg)

    n = len(example_ids)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    theta_fine = np.linspace(0, np.pi, 200)

    for ax_i, nid in enumerate(example_ids):
        ax = axes[ax_i]
        tc = tuning_mean[nid]
        se = tuning_sem[nid]

        ax.plot(centers_deg, tc, "o", color="steelblue", ms=4, label="Empirical")
        ax.fill_between(centers_deg, tc - se, tc + se, alpha=0.2, color="steelblue")

        row = df.iloc[nid]
        if row["fit_success"]:
            b = float(row["fit_baseline"])
            a = float(row["fit_amplitude"])
            kappa = float(row["fit_kappa_or_width"])
            theta0 = np.radians(float(row["fit_pref_orientation_deg"]))
            y_fit = von_mises_tuning(theta_fine, b, a, kappa, theta0)
            ax.plot(np.degrees(theta_fine), y_fit, "-", color="tomato", lw=1.5,
                    label=f"Fit R²={row['fit_r2']:.2f}")
            ax.legend(fontsize=6)

        ax.set_title(
            f"Neuron {nid}\npref={row['pref_orientation_deg_empirical']:.0f}°  "
            f"κ={row['fit_kappa_or_width']:.1f}" if row["fit_success"] else f"Neuron {nid}\n(fit failed)",
            fontsize=8
        )
        ax.set_xlabel("Orientation (°)", fontsize=7)
        ax.set_ylabel("Response", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax_i in range(n, len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("Empirical + Fitted Tuning Curves — 12 Example Neurons", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_pref_orientation_dist(df, save_path):
    pref = df.loc[df["fit_success"], "fit_pref_orientation_deg"].dropna()
    pref_emp = df["pref_orientation_deg_empirical"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(pref_emp, bins=36, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Preferred orientation (°) — empirical")
    axes[0].set_ylabel("Number of neurons")
    axes[0].set_title(f"Empirical Pref Orientation\n(n={len(pref_emp):,})")

    axes[1].hist(pref, bins=36, color="darkorange", edgecolor="none")
    axes[1].set_xlabel("Preferred orientation (°) — fitted")
    axes[1].set_ylabel("Number of neurons")
    axes[1].set_title(f"Fitted Pref Orientation\n(n={len(pref):,})")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")


def main():
    print("Loading processed data...")
    df, bd = load_data()
    n_neurons = len(df)

    # Choose 12 example neurons: use reliable neurons with successful fits
    reliable = df[df["split_half_reliability"] > 0.3]
    if len(reliable) >= N_EXAMPLE:
        example_ids = rng.choice(reliable.index, size=N_EXAMPLE, replace=False)
    else:
        example_ids = rng.choice(n_neurons, size=N_EXAMPLE, replace=False)
    example_ids = np.sort(example_ids)
    print(f"Example neuron IDs: {example_ids}")

    print("\nGenerating plots...")
    plot_mean_response_hist(df, os.path.join(FIG_DIR, "01_mean_response_histogram.png"))
    plot_reliability_hist(df, os.path.join(FIG_DIR, "02_reliability_histogram.png"))
    plot_example_tuning_curves(df, bd, example_ids,
                               os.path.join(FIG_DIR, "03_empirical_tuning_curves.png"))
    plot_empirical_and_fitted(df, bd, example_ids,
                              os.path.join(FIG_DIR, "04_fitted_tuning_curves.png"))
    plot_pref_orientation_dist(df, os.path.join(FIG_DIR, "05_preferred_orientation_distribution.png"))

    print("\nAll figures saved to reports/figures/")


if __name__ == "__main__":
    main()
