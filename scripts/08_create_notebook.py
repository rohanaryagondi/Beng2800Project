"""
Create the starter Jupyter notebook: notebooks/01_orientation_project_start.ipynb
"""
import os
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "notebooks", "01_orientation_project_start.ipynb")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def cell(cell_type, source, **kwargs):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [source],
        }
    else:  # code
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else [source],
        }


cells = [
    # ---------------------------------------------------------------
    cell("markdown", [
        "# Orientation Tuning in Mouse V1 — BENG 2800 Project\n",
        "\n",
        "**Dataset:** Stringer, Michaelos, Pachitariu (2021) *High-precision coding in visual cortex*. Cell.\n",
        "DOI: [10.1016/j.cell.2021.03.042](https://doi.org/10.1016/j.cell.2021.03.042)\n",
        "\n",
        "**Recording:** ~23,589 neurons simultaneously recorded from mouse V1 during presentation of oriented gratings.\n",
        "\n",
        "---\n",
        "\n",
        "## Project Questions\n",
        "1. How heterogeneous are orientation tuning properties across neurons in mouse V1?\n",
        "2. Are neurons with sharper tuning also more reliable across trials?\n",
        "3. How well can a simple linear population decoder recover stimulus orientation?\n",
    ]),

    # ---------------------------------------------------------------
    cell("markdown", "## Setup"),

    cell("code", [
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib\n",
        "matplotlib.rcParams['figure.dpi'] = 120\n",
        "\n",
        "# Base path — adjust if running from a different directory\n",
        "BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
        "PROC = os.path.join(BASE, 'data', 'processed')\n",
        "\n",
        "print('Base:', BASE)\n",
        "print('Processed files:')\n",
        "for f in sorted(os.listdir(PROC)):\n",
        "    p = os.path.join(PROC, f)\n",
        "    print(f'  {f}: {os.path.getsize(p)/1e6:.1f} MB')\n",
    ]),

    # ---------------------------------------------------------------
    cell("markdown", "## Introduction / Dataset"),

    cell("code", [
        "import json\n",
        "\n",
        "with open(os.path.join(PROC, 'orientation_project_metadata.json')) as f:\n",
        "    meta = json.load(f)\n",
        "\n",
        "print('Recording ID(s):', meta['selected_recording_ids'])\n",
        "print('N neurons:      ', meta['n_neurons'])\n",
        "print('N trials:       ', meta['n_trials'])\n",
        "print('Orientation bins:', meta['n_orientation_bins'], 'x', meta['angle_bin_width_deg'], '°')\n",
        "print('Response summary:', meta['response_summary'][:120], '...')\n",
    ]),

    cell("code", [
        "# Load the neuron summary table\n",
        "df = pd.read_parquet(os.path.join(PROC, 'orientation_neuron_summary.parquet'))\n",
        "print('Neuron summary shape:', df.shape)\n",
        "print('\\nColumns:', list(df.columns))\n",
        "df.head()\n",
    ]),

    # ---------------------------------------------------------------
    cell("markdown", "## EDA"),

    cell("code", [
        "# Load decoder-ready matrix\n",
        "d = np.load(os.path.join(PROC, 'orientation_decoder_ready.npz'))\n",
        "X = d['X']                # (n_trials, n_neurons)\n",
        "theta_deg = d['theta_deg'] # (n_trials,)\n",
        "theta_rad = d['theta_rad'] # (n_trials,)\n",
        "\n",
        "print('X (trials x neurons):', X.shape)\n",
        "print('theta_deg range:     ', theta_deg.min(), '-', theta_deg.max(), '°')\n",
        "print('Unique orientations: ', len(np.unique(np.round(theta_deg, 1))))\n",
    ]),

    cell("code", [
        "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
        "\n",
        "axes[0].hist(df['mean_response'], bins=80, color='steelblue', edgecolor='none')\n",
        "axes[0].set_xlabel('Mean response')\n",
        "axes[0].set_ylabel('Neurons')\n",
        "axes[0].set_title('Mean Response Distribution')\n",
        "\n",
        "axes[1].hist(df['split_half_reliability'], bins=60, color='darkorange', edgecolor='none')\n",
        "axes[1].set_xlabel('Split-half reliability (r)')\n",
        "axes[1].set_title('Reliability Distribution')\n",
        "\n",
        "axes[2].hist(theta_deg, bins=36, color='seagreen', edgecolor='none')\n",
        "axes[2].set_xlabel('Orientation (°)')\n",
        "axes[2].set_ylabel('Trials')\n",
        "axes[2].set_title('Trial Orientation Distribution')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
    ]),

    # ---------------------------------------------------------------
    cell("markdown", "## Nonlinear Least-Squares Tuning Fits"),

    cell("code", [
        "from scipy.optimize import curve_fit\n",
        "\n",
        "def von_mises_tuning(theta, b, a, kappa, theta0):\n",
        "    \"\"\"r(theta) = b + a * exp(kappa * cos(2*(theta - theta0)))\"\"\"\n",
        "    return b + a * np.exp(kappa * np.cos(2.0 * (theta - theta0)))\n",
    ]),

    cell("code", [
        "# Load binned tuning data\n",
        "bd = np.load(os.path.join(PROC, 'orientation_binned_tuning.npz'))\n",
        "tuning_mean = bd['tuning_mean']         # (n_neurons, n_bins)\n",
        "tuning_sem  = bd['tuning_sem']           # (n_neurons, n_bins)\n",
        "centers_deg = bd['angle_bin_centers_deg'] # (n_bins,)\n",
        "centers_rad = np.radians(centers_deg)\n",
        "\n",
        "print('Tuning mean shape:', tuning_mean.shape)\n",
        "print('Bin centers (deg):', centers_deg[[0, -1]])\n",
    ]),

    cell("code", [
        "# Show 1 example fit\n",
        "rng = np.random.default_rng(42)\n",
        "# Pick a reliable neuron with a successful fit\n",
        "good_neurons = df[(df['split_half_reliability'] > 0.4) & df['fit_success']].index\n",
        "ex_id = int(rng.choice(good_neurons)) if len(good_neurons) > 0 else 0\n",
        "\n",
        "tc = tuning_mean[ex_id]\n",
        "se = tuning_sem[ex_id]\n",
        "row = df.iloc[ex_id]\n",
        "\n",
        "theta_fine = np.linspace(0, np.pi, 300)\n",
        "y_fit = von_mises_tuning(\n",
        "    theta_fine,\n",
        "    row['fit_baseline'], row['fit_amplitude'],\n",
        "    row['fit_kappa_or_width'], np.radians(row['fit_pref_orientation_deg'])\n",
        ")\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(6, 4))\n",
        "ax.errorbar(centers_deg, tc, yerr=se, fmt='o', color='steelblue', ms=5, label='Empirical')\n",
        "ax.plot(np.degrees(theta_fine), y_fit, '-', color='tomato', lw=2,\n",
        "        label=f'Von Mises fit (R²={row[\"fit_r2\"]:.2f}, κ={row[\"fit_kappa_or_width\"]:.2f})')\n",
        "ax.axvline(row['fit_pref_orientation_deg'], color='gray', lw=1, linestyle='--',\n",
        "           label=f'Pref = {row[\"fit_pref_orientation_deg\"]:.0f}°')\n",
        "ax.set_xlabel('Orientation (°)')\n",
        "ax.set_ylabel('Deconvolved activity')\n",
        "ax.set_title(f'Neuron {ex_id} — Orientation Tuning Curve')\n",
        "ax.legend(fontsize=9)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "print(f'OSI = {row[\"osi_empirical\"]:.3f}  |  Reliability = {row[\"split_half_reliability\"]:.3f}')\n",
    ]),

    cell("code", [
        "# Summary of fit quality\n",
        "n_success = df['fit_success'].sum()\n",
        "print(f'Successful fits: {n_success}/{len(df)} ({100*n_success/len(df):.1f}%)')\n",
        "print(f'Median R²:       {df.loc[df[\"fit_success\"], \"fit_r2\"].median():.3f}')\n",
        "print(f'Median κ:        {df.loc[df[\"fit_success\"], \"fit_kappa_or_width\"].median():.3f}')\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
        "df.loc[df['fit_success'], 'fit_r2'].hist(bins=50, ax=axes[0], color='steelblue')\n",
        "axes[0].set_xlabel('Fit R²')\n",
        "axes[0].set_title('Distribution of Fit Quality')\n",
        "\n",
        "df.loc[df['fit_success'], 'fit_kappa_or_width'].hist(bins=50, ax=axes[1], color='darkorange')\n",
        "axes[1].set_xlabel('Tuning sharpness κ')\n",
        "axes[1].set_title('Distribution of Tuning Sharpness')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
    ]),

    # ---------------------------------------------------------------
    cell("markdown", "## Reliability Analysis"),

    cell("code", [
        "# Tuning sharpness vs reliability scatter\n",
        "df_fit = df[df['fit_success']].copy()\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(6, 5))\n",
        "sc = ax.scatter(\n",
        "    df_fit['fit_kappa_or_width'].clip(0, 10),\n",
        "    df_fit['split_half_reliability'],\n",
        "    c=df_fit['fit_r2'], cmap='viridis',\n",
        "    s=1, alpha=0.3\n",
        ")\n",
        "plt.colorbar(sc, ax=ax, label='Fit R²')\n",
        "ax.set_xlabel('Tuning sharpness κ (clipped at 10)')\n",
        "ax.set_ylabel('Split-half reliability (r)')\n",
        "ax.set_title('Tuning Sharpness vs. Reliability')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "from scipy.stats import spearmanr\n",
        "r, p = spearmanr(df_fit['fit_kappa_or_width'], df_fit['split_half_reliability'])\n",
        "print(f'Spearman r = {r:.3f}, p = {p:.2e}')\n",
    ]),

    # ---------------------------------------------------------------
    cell("markdown", "## Linear Decoder"),

    cell("code", [
        "# Load decoder targets (y vectors only — X is in orientation_decoder_ready.npz)\n",
        "dt = np.load(os.path.join(PROC, 'orientation_decoder_targets.npz'))\n",
        "y_cos2 = dt['y_cos2']     # (n_trials,)  cos(2*theta)\n",
        "y_sin2 = dt['y_sin2']     # (n_trials,)  sin(2*theta)\n",
        "theta_deg_dec = dt['theta_deg']\n",
        "print('y_cos2:', y_cos2.shape)\n",
        "\n",
        "# Load top-1000 most reliable neurons (recommended for laptop decoding)\n",
        "top1000_path = os.path.join(PROC, 'orientation_decoder_ready_top1000.npz')\n",
        "d1000 = np.load(top1000_path)\n",
        "X_top = d1000['X']   # (n_trials, 1000)\n",
        "print(f'Top-1000 subset: {X_top.shape}')\n",
        "\n",
        "# Full matrix if needed (221 MB in memory)\n",
        "# d_full = np.load(os.path.join(PROC, 'orientation_decoder_ready.npz'))\n",
        "# X_full = d_full['X']   # (4598, 23589)\n",
    ]),

    cell("code", [
        "# Simple least-squares linear decoder demonstration\n",
        "# Split into train/test (80/20)\n",
        "rng2 = np.random.default_rng(0)\n",
        "n = X_top.shape[0]\n",
        "idx = rng2.permutation(n)\n",
        "train_idx, test_idx = idx[:int(0.8*n)], idx[int(0.8*n):]\n",
        "\n",
        "X_train, X_test = X_top[train_idx], X_top[test_idx]\n",
        "yc_train, yc_test = y_cos2[train_idx], y_cos2[test_idx]\n",
        "ys_train, ys_test = y_sin2[train_idx], y_sin2[test_idx]\n",
        "\n",
        "# Fit with numpy lstsq\n",
        "Y_train = np.stack([yc_train, ys_train], axis=1)  # (n_train, 2)\n",
        "W, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond=None)  # (n_neurons, 2)\n",
        "\n",
        "# Predict on test set\n",
        "Y_pred = X_test @ W          # (n_test, 2)\n",
        "theta_pred = np.arctan2(Y_pred[:, 1], Y_pred[:, 0]) / 2.0  # radians\n",
        "theta_pred = theta_pred % np.pi                              # 0..pi\n",
        "theta_true = np.radians(theta_deg_dec[test_idx])\n",
        "\n",
        "# Circular error\n",
        "err = theta_pred - theta_true\n",
        "err = (err + np.pi/2) % np.pi - np.pi/2  # wrap to [-pi/2, pi/2]\n",
        "mae_deg = float(np.degrees(np.abs(err).mean()))\n",
        "print(f'Linear decoder MAE: {mae_deg:.1f}°  (chance ≈ 45°)')\n",
    ]),

    cell("code", [
        "# Scatter plot: true vs predicted orientation\n",
        "fig, ax = plt.subplots(figsize=(5, 5))\n",
        "ax.scatter(np.degrees(theta_true), np.degrees(theta_pred),\n",
        "           s=1, alpha=0.3, color='steelblue')\n",
        "ax.plot([0, 180], [0, 180], 'r--', lw=1, label='Perfect')\n",
        "ax.set_xlabel('True orientation (°)')\n",
        "ax.set_ylabel('Predicted orientation (°)')\n",
        "ax.set_title(f'Linear Decoder — MAE = {mae_deg:.1f}°')\n",
        "ax.legend(fontsize=9)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
    ]),

    # ---------------------------------------------------------------
    cell("markdown", "## Results / Interpretation"),

    cell("markdown", [
        "*(Fill in your analysis results here.)*\n",
        "\n",
        "### Suggested questions to address:\n",
        "1. What fraction of neurons have a clear preferred orientation (OSI > 0.3)?\n",
        "2. Is there a relationship between tuning sharpness (κ) and reliability?\n",
        "3. How does decoder performance scale with the number of neurons?\n",
        "4. What is the distribution of preferred orientations — is it uniform or are some angles over-represented?\n",
    ]),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0",
        },
    },
    "cells": cells,
}

with open(OUT, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written: {OUT}")
print(f"Size: {os.path.getsize(OUT)/1e3:.1f} KB")
