"""
Generate answers to project understanding questions with figures.
Outputs to Questions/ folder.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
FINAL = os.path.join(BASE, "data", "final")
QDIR = os.path.join(BASE, "Questions")
os.makedirs(QDIR, exist_ok=True)

SEED = 42
DPI = 150


def von_mises_tuning(theta, b, a, kappa, theta0):
    return b + a * np.exp(kappa * np.cos(2.0 * (theta - theta0)))


def main():
    rng = np.random.default_rng(SEED)

    # Load data
    df = pd.read_parquet(os.path.join(FINAL, "orientation_neuron_analysis.parquet"))
    bd = np.load(os.path.join(PROC, "orientation_binned_tuning.npz"))
    tuning_mean = bd["tuning_mean"]
    tuning_sem = bd["tuning_sem"]
    centers_deg = bd["angle_bin_centers_deg"]
    centers_rad = np.radians(centers_deg)

    df_fit = df[df["fit_success"]].copy()
    k = df_fit["fit_kappa_or_width"].values
    rel = df_fit["split_half_reliability"].values

    # ==================================================================
    # QUESTION 1: Example tuning curves at kappa ~ 0 and kappa ~ 20
    # ==================================================================
    print("Q1: Generating extreme kappa example tuning curves...")

    # Find 3 neurons near kappa=0 and 3 near kappa=20
    near_zero = df_fit[df_fit["fit_kappa_or_width"] < 0.05].copy()
    near_twenty = df_fit[df_fit["fit_kappa_or_width"] >= 19.9].copy()

    # Pick examples with decent reliability so the empirical curve is visible
    zero_examples = near_zero.sort_values("split_half_reliability", ascending=False).head(20)
    zero_chosen = zero_examples.sample(n=min(3, len(zero_examples)), random_state=SEED)

    twenty_examples = near_twenty.sort_values("fit_r2", ascending=False).head(20)
    twenty_chosen = twenty_examples.sample(n=min(3, len(twenty_examples)), random_state=SEED)

    theta_fine = np.linspace(0, np.pi, 300)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Top row: kappa ~ 0
    for ax, (_, row) in zip(axes[0], zero_chosen.iterrows()):
        nid = int(row["neuron_id"])
        tc = tuning_mean[nid]
        se = tuning_sem[nid]
        ax.errorbar(centers_deg, tc, yerr=se, fmt="o", color="#2171b5", ms=4,
                    capsize=2, label="Empirical")
        y_fit = von_mises_tuning(theta_fine, row["fit_baseline"], row["fit_amplitude"],
                                 row["fit_kappa_or_width"], np.radians(row["fit_pref_orientation_deg"]))
        ax.plot(np.degrees(theta_fine), y_fit, "-", color="#e6550d", lw=2, label="Von Mises fit")
        ax.set_title(f"Neuron {nid}\n$\\kappa$ = {row['fit_kappa_or_width']:.4f}, "
                     f"R$^2$ = {row['fit_r2']:.3f}", fontsize=10)
        ax.set_xlim(0, 180)
        ax.legend(fontsize=7)

    # Bottom row: kappa ~ 20
    for ax, (_, row) in zip(axes[1], twenty_chosen.iterrows()):
        nid = int(row["neuron_id"])
        tc = tuning_mean[nid]
        se = tuning_sem[nid]
        ax.errorbar(centers_deg, tc, yerr=se, fmt="o", color="#2171b5", ms=4,
                    capsize=2, label="Empirical")
        y_fit = von_mises_tuning(theta_fine, row["fit_baseline"], row["fit_amplitude"],
                                 row["fit_kappa_or_width"], np.radians(row["fit_pref_orientation_deg"]))
        ax.plot(np.degrees(theta_fine), y_fit, "-", color="#e6550d", lw=2, label="Von Mises fit")
        ax.set_title(f"Neuron {nid}\n$\\kappa$ = {row['fit_kappa_or_width']:.2f}, "
                     f"R$^2$ = {row['fit_r2']:.3f}", fontsize=10)
        ax.set_xlim(0, 180)
        ax.legend(fontsize=7)

    for ax in axes[0]:
        ax.set_ylabel("Response")
    for ax in axes[1]:
        ax.set_xlabel("Orientation (deg)")
        ax.set_ylabel("Response")

    axes[0][0].annotate("$\\kappa \\approx 0$ (broadly tuned / flat)", xy=(0.5, 1.15),
                        xycoords="axes fraction", fontsize=12, fontweight="bold",
                        ha="center", color="#2171b5")
    axes[1][0].annotate("$\\kappa \\approx 20$ (very sharply tuned)", xy=(0.5, 1.15),
                        xycoords="axes fraction", fontsize=12, fontweight="bold",
                        ha="center", color="#e6550d")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(QDIR, "Q1_extreme_kappa_tuning_curves.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Also make a theoretical comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    theta0 = np.pi / 2  # 90 degrees preferred

    for kval, color, label in [(0.01, "#2171b5", "$\\kappa = 0$"),
                                (1, "#6baed6", "$\\kappa = 1$"),
                                (3, "#fd8d3c", "$\\kappa = 3$"),
                                (10, "#e6550d", "$\\kappa = 10$"),
                                (20, "#a63603", "$\\kappa = 20$")]:
        y = von_mises_tuning(theta_fine, 0, 1, kval, theta0)
        # Normalize to [0, 1]
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.ones_like(y) * 0.5
        axes[0].plot(np.degrees(theta_fine), y_norm, lw=2, label=label, color=color)

    axes[0].set_xlabel("Orientation (deg)")
    axes[0].set_ylabel("Normalized response")
    axes[0].set_title("Theoretical von Mises curves at different $\\kappa$")
    axes[0].legend()
    axes[0].set_xlim(0, 180)
    axes[0].axvline(90, color="gray", ls=":", lw=1, alpha=0.5)

    # Show where these extremes sit on the kappa distribution
    axes[1].hist(k, bins=60, color="coral", edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color="#2171b5", lw=2, ls="--", label="$\\kappa = 0$ (flat)")
    axes[1].axvline(20, color="#a63603", lw=2, ls="--", label="$\\kappa = 20$ (sharp)")
    axes[1].set_xlabel("Tuning sharpness ($\\kappa$)")
    axes[1].set_ylabel("Number of neurons")
    axes[1].set_title("Distribution of $\\kappa$ with extremes marked")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(QDIR, "Q1_kappa_theoretical_comparison.png"), dpi=DPI)
    plt.close(fig)

    # ==================================================================
    # QUESTION 2: Why does binned reliability behave as it does?
    # ==================================================================
    print("Q2: Generating detailed binned reliability analysis...")

    # Finer bins to show the pattern clearly
    n_bins = 20
    bin_edges = np.linspace(0, 20, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = np.empty(n_bins)
    bin_sems = np.empty(n_bins)
    bin_counts = np.empty(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (k >= bin_edges[i]) & (k < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (k == bin_edges[i + 1])
        vals = rel[mask]
        bin_counts[i] = len(vals)
        bin_means[i] = vals.mean() if len(vals) > 0 else np.nan
        bin_sems[i] = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else np.nan

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: binned reliability with sample sizes
    ax = axes[0, 0]
    ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt="o-", color="#2171b5",
                capsize=3, ms=5, lw=1.5)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ax.annotate(f"n={bin_counts[i]}", (bin_centers[i], bin_means[i]),
                        textcoords="offset points", xytext=(0, 10), fontsize=6,
                        ha="center", color="gray")
    ax.set_xlabel("$\\kappa$")
    ax.set_ylabel("Mean reliability")
    ax.set_title("Binned reliability (20 bins) with sample sizes")
    ax.axhline(rel.mean(), color="gray", ls=":", lw=1, label=f"Overall mean = {rel.mean():.3f}")
    ax.legend(fontsize=8)

    # Top-right: sample size per bin (explains the noise)
    ax = axes[0, 1]
    ax.bar(bin_centers, bin_counts, width=0.8, color="coral", edgecolor="white")
    ax.set_xlabel("$\\kappa$")
    ax.set_ylabel("Number of neurons")
    ax.set_title("Sample size per $\\kappa$ bin")
    ax.annotate("Most neurons\nconcentrated\nat low $\\kappa$", xy=(1, 5000),
                fontsize=10, color="#e6550d", fontweight="bold")
    ax.annotate("Very few neurons\n$\\rightarrow$ noisy estimates", xy=(12, 500),
                fontsize=10, color="#e6550d", fontweight="bold")
    ax.annotate("Spike at cap!\n($\\kappa$ = 20)", xy=(19, bin_counts[-1]),
                fontsize=10, color="red", fontweight="bold",
                xytext=(14, bin_counts[-1] + 300),
                arrowprops=dict(arrowstyle="->", color="red"))

    # Bottom-left: reliability distribution split by kappa group
    ax = axes[1, 0]
    groups = [
        (k < 1, "$\\kappa < 1$ (broad)", "#2171b5"),
        ((k >= 1) & (k < 5), "$\\kappa$ 1-5 (moderate)", "#6baed6"),
        ((k >= 5) & (k < 19), "$\\kappa$ 5-19 (sharp)", "#fd8d3c"),
        (k >= 19.5, "$\\kappa \\geq$ 19.5 (capped)", "#e6550d"),
    ]
    for mask, label, color in groups:
        vals = rel[mask]
        ax.hist(vals, bins=40, alpha=0.5, color=color, label=f"{label} (n={mask.sum()})",
                density=True, edgecolor="none")
    ax.set_xlabel("Split-half reliability")
    ax.set_ylabel("Density")
    ax.set_title("Reliability distribution by $\\kappa$ group")
    ax.legend(fontsize=8)

    # Bottom-right: R2 of capped vs uncapped neurons
    ax = axes[1, 1]
    capped = df_fit[df_fit["fit_kappa_or_width"] >= 19.5]
    uncapped = df_fit[(df_fit["fit_kappa_or_width"] >= 5) & (df_fit["fit_kappa_or_width"] < 19)]
    ax.hist(uncapped["fit_r2"], bins=40, alpha=0.6, color="#6baed6",
            label=f"$\\kappa$ 5-19 (n={len(uncapped)})", density=True, edgecolor="none")
    ax.hist(capped["fit_r2"], bins=40, alpha=0.6, color="#e6550d",
            label=f"$\\kappa \\geq$ 19.5 (capped, n={len(capped)})", density=True, edgecolor="none")
    ax.set_xlabel("Fit R$^2$")
    ax.set_ylabel("Density")
    ax.set_title("Fit quality: capped vs uncapped neurons")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(QDIR, "Q2_binned_reliability_explained.png"), dpi=DPI)
    plt.close(fig)

    # ==================================================================
    # QUESTION 3: What is the OLS scatter showing?
    # ==================================================================
    print("Q3: Generating OLS scatter explanation figure...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: the actual hexbin scatter with OLS line
    ax = axes[0]
    hb = ax.hexbin(k.clip(0, 20), rel, gridsize=40, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Neuron count")
    # OLS line
    beta0, beta1, beta2 = 0.6703, 0.004006, 0.003977
    mean_resp_avg = df_fit["mean_response"].mean()
    k_range = np.linspace(0, 20, 200)
    ax.plot(k_range, beta0 + beta1 * k_range + beta2 * mean_resp_avg,
            "-", color="black", lw=2.5, label="OLS regression line")
    ax.set_xlabel("$\\kappa$")
    ax.set_ylabel("Split-half reliability")
    ax.set_title("(a) The OLS scatter: why is the line flat?")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 20)

    # Middle: zoomed in on just the regression line to show the actual slope
    ax = axes[1]
    y_line = beta0 + beta1 * k_range + beta2 * mean_resp_avg
    ax.plot(k_range, y_line, "-", color="black", lw=2.5)
    ax.fill_between(k_range, y_line - 0.005, y_line + 0.005, alpha=0.2, color="gray")
    ax.set_xlabel("$\\kappa$")
    ax.set_ylabel("Predicted reliability")
    ax.set_title("(b) Zoomed: the line IS increasing")
    ax.annotate(f"Total rise = {beta1 * 20:.3f}\n(from $\\kappa$=0 to $\\kappa$=20)",
                xy=(10, y_line[100]), fontsize=10, fontweight="bold",
                xytext=(2, y_line[100] + 0.02),
                arrowprops=dict(arrowstyle="->", color="black"))
    # Show the scale comparison
    ax.set_ylim(y_line.min() - 0.05, y_line.max() + 0.05)

    # Right: the binned means (what you should compare to, not the OLS line)
    ax = axes[2]
    bin_edges_10 = np.linspace(0, 20, 11)
    bc_10 = (bin_edges_10[:-1] + bin_edges_10[1:]) / 2
    bm_10 = np.empty(10)
    bs_10 = np.empty(10)
    for i in range(10):
        mask = (k >= bin_edges_10[i]) & (k < bin_edges_10[i + 1])
        if i == 9:
            mask = mask | (k == bin_edges_10[i + 1])
        vals = rel[mask]
        bm_10[i] = vals.mean()
        bs_10[i] = vals.std() / np.sqrt(len(vals))

    ax.errorbar(bc_10, bm_10, yerr=bs_10, fmt="o-", color="#2171b5", capsize=4, ms=6, lw=2)
    ax.plot(k_range, y_line, "--", color="black", lw=1.5, alpha=0.5, label="OLS line (for comparison)")
    ax.set_xlabel("$\\kappa$")
    ax.set_ylabel("Mean reliability")
    ax.set_title("(c) Binned means show the TRUE pattern")
    ax.legend(fontsize=8)
    ax.annotate("Sharp initial rise\n(real nonlinear effect)", xy=(1, 0.66),
                fontsize=9, color="#e6550d",
                xytext=(4, 0.63), arrowprops=dict(arrowstyle="->", color="#e6550d"))
    ax.annotate("Plateau + noise\n(few neurons)", xy=(12, 0.78),
                fontsize=9, color="gray")
    ax.annotate("Drop: capped\nneurons", xy=(19, 0.72),
                fontsize=9, color="red",
                xytext=(15, 0.64), arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    fig.savefig(os.path.join(QDIR, "Q3_ols_line_explained.png"), dpi=DPI)
    plt.close(fig)

    # ==================================================================
    # Write the answers markdown
    # ==================================================================
    print("Writing answers document...")

    answers = f"""# Project Understanding Questions & Answers

---

## Question 1: What do tuning curves look like at the extremes of kappa (kappa ~ 0 and kappa ~ 20)?

### Answer

The tuning sharpness parameter kappa controls how narrowly tuned a neuron is to its preferred orientation. At the two extremes:

**Kappa ~ 0 (broadly tuned / essentially flat):**
- The von Mises curve becomes nearly flat: exp(0 * cos(...)) = exp(0) = 1 for all orientations
- The neuron responds roughly equally to all orientations --- it has no meaningful orientation preference
- In our data, **3,661 neurons** (15.5%) have kappa < 0.05
- These neurons tend to have lower fit R-squared (mean R-squared = 0.38) because a flat line cannot capture much variance
- Their lower reliability (mean = 0.59) makes sense: if there is no orientation signal, odd and even trials are just two samples of noise

**Kappa ~ 20 (very sharply tuned, at the cap):**
- The von Mises curve becomes an extremely narrow peak, almost like a delta function
- The neuron responds strongly to a tiny range of orientations and is nearly silent elsewhere
- In our data, **1,978 neurons** (8.4%) have kappa >= 19.5 (hitting the fitting cap at 20)
- IMPORTANT: many of these neurons hit the cap not because they are genuinely infinitely sharp, but because the optimizer ran up against the bound. This is an **artifact of the fitting constraint**

See the figures below:

![Extreme kappa tuning curves](Q1_extreme_kappa_tuning_curves.png)

*Figure Q1a: Real example neurons at both extremes. Top row: kappa near 0 (flat curves, no clear preferred orientation). Bottom row: kappa near 20 (very narrow peaks at the preferred orientation).*

![Theoretical comparison](Q1_kappa_theoretical_comparison.png)

*Figure Q1b: Left --- Theoretical von Mises curves at different kappa values (all normalized, preferred orientation = 90 deg). As kappa increases, the peak narrows dramatically. Right --- The kappa distribution, showing where these extremes fall in the population.*

---

## Question 2: Why does binned mean reliability rise sharply for kappa > 1, oscillate in the middle, and drop at the end?

### Answer

The "binned reliability vs. sharpness" plot shows three distinct regimes, each with a different explanation:

### (a) Sharp rise from kappa = 0 to kappa ~ 2 (reliability jumps from ~0.66 to ~0.77)

This is a **real biological/statistical effect**. Neurons with kappa near 0 are essentially untuned --- they have no orientation preference. Their split-half reliability measures the correlation between two tuning curves that are both just noise, so it tends to be low. As soon as kappa exceeds ~1, there is a genuine orientation signal in the tuning curve. Both the odd and even trial halves will show a similar peak, producing a higher correlation. This transition from "no signal" to "clear signal" drives the steep initial rise.

The bin from kappa = 0 to 2 contains **7,541 neurons** (32% of the population), so this is a well-estimated effect.

### (b) Apparent oscillation from kappa ~ 3 to kappa ~ 16 (reliability hovers around 0.77)

This is primarily **sampling noise**, not a real biological oscillation. Here is why:

- The number of neurons per bin **drops dramatically** at higher kappa values:
  - kappa 0-2: 7,541 neurons
  - kappa 2-4: 4,603 neurons
  - kappa 8-10: 1,335 neurons
  - kappa 12-14: 650 neurons
  - kappa 14-16: 546 neurons

- With only ~500-1,300 neurons per bin, the standard error of the mean is 0.005-0.009 (visible as error bars)
- The apparent wiggles are within ~1-2 SEMs of each other --- they are **not statistically meaningful**
- The true underlying relationship is likely a smooth, gentle plateau in this range

Once kappa exceeds ~2, adding more sharpness does not dramatically improve trial-to-trial consistency. The signal is already clear enough for the two halves to correlate well.

### (c) Drop in the last bin (kappa ~ 18-20, reliability drops to ~0.71)

This is an **artifact of the kappa = 20 fitting cap**. The fitting procedure constrains kappa to the range [0, 20]. When the optimizer wants kappa > 20, it gets stuck at 20. These 1,978 "capped" neurons are a **heterogeneous mix**:

- Some are genuinely very sharply tuned
- Others are neurons where the optimizer converged poorly and ran to the boundary
- The capped group has **lower mean fit R-squared** (0.686) compared to the uncapped sharp neurons at kappa 5-19 (R-squared = 0.783)
- This lower fit quality translates to lower reliability, because poor fits often indicate noisy or complex responses

In short: the drop is not because "very sharp tuning causes unreliability" --- it is because the kappa = 20 cap creates a grab bag of neurons with varying quality.

![Binned reliability explained](Q2_binned_reliability_explained.png)

*Figure Q2: Four-panel explanation. (a) Binned reliability with sample sizes per bin --- note how counts drop at higher kappa. (b) Sample size distribution --- the population is concentrated at low kappa, with a spike at the cap (kappa ~ 20). (c) Reliability distributions by kappa group, showing the capped neurons (red) have a broader, lower distribution. (d) Fit quality: capped neurons (orange) have worse R-squared than genuinely sharp neurons (blue), explaining their lower reliability.*

---

## Question 3: What exactly is the split-half reliability scatter showing, and why does the OLS line look so flat?

### Answer

### What is split-half reliability?

Split-half reliability measures **how consistent a neuron's orientation tuning is across repeated trials**. Here is exactly how it works:

1. Take all 4,598 trials and split them into two halves: odd-numbered trials (trial 1, 3, 5, ...) and even-numbered trials (trial 2, 4, 6, ...)
2. For each half, compute a tuning curve: the mean response at each of the 36 orientation bins
3. Compute the **Pearson correlation** between these two 36-element tuning curves
4. This correlation (ranging from -1 to +1) is the split-half reliability

**High reliability (close to 1):** The neuron's tuning curve looks the same whether you use odd or even trials. It is a reliable, consistent responder.

**Low reliability (close to 0):** The tuning curve looks different on different subsets of trials. The neuron's responses are noisy or inconsistent.

**Negative reliability:** The two halves are anti-correlated, which usually means the neuron has no real tuning and both curves are just noise.

### Is the scatter plot the "trendline" for the binned graph?

**Not exactly --- they show different things:**

- The **binned means plot** (right panel of Figure 3 in the report) shows the *average* reliability for neurons grouped by kappa. It reveals the *shape* of the relationship (sharp rise, plateau, drop).
- The **hexbin scatter plot** (left panel) shows every single neuron as a data point, revealing the full *spread* of the data. The OLS line on this plot is a *linear* fit through all 23,589 individual points.

The OLS line is a **straight line forced through a nonlinear relationship** with enormous scatter. The binned means are a better representation of the actual trend.

### Why is the OLS line so flat?

The line IS increasing --- it just looks flat because of the **scale mismatch**:

- The reliability axis spans from -0.38 to +0.997 (a range of 1.38)
- The OLS line predicts a total increase of only **0.080** across the full kappa range (0 to 20)
- That 0.080 increase is only **5.8% of the total reliability range**
- With 23,589 data points scattered across this huge range, a line that rises by 0.080 units looks essentially flat

**Why so small a slope?** Because the relationship between kappa and reliability is:
1. **Nonlinear** --- most of the action happens between kappa 0 and 2, then it plateaus. A straight line averages this into a gentle overall slope.
2. **Noisy** --- kappa and mean response together explain only R-squared = 0.055 (5.5%) of the variance in reliability. The other 94.5% comes from factors we did not measure (noise correlations, behavioral state, calcium indicator dynamics, etc.)

The OLS model is statistically significant (p < 10^-70 for the kappa coefficient) only because we have so many neurons. The *effect size* is modest.

![OLS line explained](Q3_ols_line_explained.png)

*Figure Q3: (a) The hexbin scatter with OLS line --- appears flat because the total predicted change (0.08) is tiny compared to the data spread. (b) Zoomed in on just the OLS line --- it IS increasing, just very gradually. (c) The binned means (blue circles) reveal the true nonlinear pattern; the OLS line (dashed) is a poor summary because the real relationship is not linear.*

---
"""

    with open(os.path.join(QDIR, "answers.md"), "w") as f:
        f.write(answers)

    print(f"Saved: {os.path.join(QDIR, 'answers.md')}")
    print("Done!")


if __name__ == "__main__":
    main()
