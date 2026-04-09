"""
Fit von Mises orientation tuning curves to each neuron using nonlinear least squares.

Model: r(theta) = b + a * exp(kappa * cos(2*(theta - theta0)))
  b       : baseline firing
  a       : amplitude (>= 0)
  theta0  : preferred orientation in radians [0, pi)
  kappa   : tuning sharpness (>= 0; larger = sharper)

Uses scipy.optimize.curve_fit. Fits to binned tuning means.
Updates data/processed/orientation_neuron_summary.parquet/.csv with fit columns.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")

SEED = 42
rng = np.random.default_rng(SEED)
FIT_METHOD = "von_mises_nlls"


def von_mises_tuning(theta, b, a, kappa, theta0):
    """Orientation tuning: r(theta) = b + a * exp(kappa * cos(2*(theta-theta0)))."""
    return b + a * np.exp(kappa * np.cos(2.0 * (theta - theta0)))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def fit_one_neuron(theta_rad, response, pref_init_rad):
    """
    Fit von Mises tuning curve to one neuron's binned responses.

    theta_rad: (n_bins,) bin center angles in radians
    response:  (n_bins,) mean response per bin
    pref_init_rad: initial guess for preferred orientation

    Returns dict with fit results.
    """
    b_init = float(np.percentile(response, 10))
    a_init = float(max(response.max() - b_init, 0.01))
    kappa_init = 1.0

    p0 = [b_init, a_init, kappa_init, pref_init_rad]
    bounds = (
        [-np.inf, 0.0,   0.0, -np.inf],
        [ np.inf, np.inf, 50.0,  np.inf],
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                von_mises_tuning,
                theta_rad,
                response,
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )
        b, a, kappa, theta0 = popt
        # Normalize theta0 to [0, pi)
        theta0 = theta0 % np.pi
        y_pred = von_mises_tuning(theta_rad, *popt)
        r2 = r_squared(response, y_pred)

        if r2 < -1.0 or np.isnan(r2) or a < 0:
            return dict(fit_success=False, fit_r2=np.nan,
                        fit_baseline=np.nan, fit_amplitude=np.nan,
                        fit_pref_orientation_deg=np.nan, fit_kappa_or_width=np.nan)

        return dict(
            fit_success=True,
            fit_r2=float(r2),
            fit_baseline=float(b),
            fit_amplitude=float(a),
            fit_pref_orientation_deg=float(np.degrees(theta0)),
            fit_kappa_or_width=float(kappa),
        )
    except Exception:
        return dict(fit_success=False, fit_r2=np.nan,
                    fit_baseline=np.nan, fit_amplitude=np.nan,
                    fit_pref_orientation_deg=np.nan, fit_kappa_or_width=np.nan)


def main():
    # Load binned tuning
    binned_path = os.path.join(PROC, "orientation_binned_tuning.npz")
    if not os.path.exists(binned_path):
        print(f"ERROR: {binned_path} not found. Run 03_process.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading binned tuning data...")
    bd = np.load(binned_path)
    tuning_mean = bd["tuning_mean"]   # (n_neurons, n_bins)
    centers_deg = bd["angle_bin_centers_deg"]
    centers_rad = np.radians(centers_deg)
    n_neurons, n_bins = tuning_mean.shape
    print(f"  n_neurons={n_neurons}, n_bins={n_bins}")

    # Load neuron summary
    summary_path = os.path.join(PROC, "orientation_neuron_summary.parquet")
    df = pd.read_parquet(summary_path)
    pref_init_deg = df["pref_orientation_deg_empirical"].values

    # Fit all neurons
    print(f"Fitting von Mises tuning curves for {n_neurons} neurons...")
    t0 = time.time()

    results = []
    log_interval = max(1, n_neurons // 20)

    for i in range(n_neurons):
        if i % log_interval == 0:
            elapsed = time.time() - t0
            pct = 100 * i / n_neurons
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_neurons - i) / rate if rate > 0 else 0
            print(f"  {pct:4.0f}%  neuron {i:5d}/{n_neurons}  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s")
            sys.stdout.flush()

        response = tuning_mean[i].astype(np.float64)
        pref_init_rad = np.radians(float(pref_init_deg[i]))
        res = fit_one_neuron(centers_rad, response, pref_init_rad)
        results.append(res)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/n_neurons*1000:.1f} ms/neuron)")

    # Update dataframe
    for col in ["fit_success", "fit_r2", "fit_baseline", "fit_amplitude",
                "fit_pref_orientation_deg", "fit_kappa_or_width"]:
        df[col] = [r[col] for r in results]
    df["fit_method"] = FIT_METHOD

    n_success = df["fit_success"].sum()
    print(f"\nFit results: {n_success}/{n_neurons} successful "
          f"({100*n_success/n_neurons:.1f}%)")
    print(f"Median R²: {df.loc[df['fit_success'], 'fit_r2'].median():.3f}")
    print(f"Median kappa: {df.loc[df['fit_success'], 'fit_kappa_or_width'].median():.3f}")

    df.to_parquet(summary_path, index=False)
    df.to_csv(os.path.join(PROC, "orientation_neuron_summary.csv"), index=False)
    print(f"\nUpdated: {summary_path}")


if __name__ == "__main__":
    main()
