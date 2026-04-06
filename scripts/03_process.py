"""
Build the main analysis-ready processed outputs from the raw Stringer data.

Outputs:
  data/processed/orientation_decoder_ready.npz
  data/processed/orientation_neuron_summary.parquet
  data/processed/orientation_neuron_summary.csv
  data/processed/orientation_binned_tuning.npz
  data/processed/orientation_project_metadata.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw", "stringer_orientations.npy")
PROC = os.path.join(BASE, "data", "processed")
os.makedirs(PROC, exist_ok=True)

RECORDING_ID = "stringer2019_v1_mouse1"
N_BINS = 36  # 5° bins over 0–180°
SEED = 42

rng = np.random.default_rng(SEED)


def load_raw():
    print(f"Loading {RAW}...")
    t0 = time.time()
    dat = np.load(RAW, allow_pickle=True).item()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return dat


def make_orientation_bins(n_bins=N_BINS):
    """Return bin edges and centers in degrees over [0, 180)."""
    edges_deg = np.linspace(0, 180, n_bins + 1)
    centers_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    return edges_deg, centers_deg


def assign_bins(theta_orientation_deg, edges_deg):
    """Assign each trial to an orientation bin index (0-indexed)."""
    # clip to [0, 180) for safety
    clipped = np.clip(theta_orientation_deg, 0, 180 - 1e-9)
    bin_idx = np.digitize(clipped, edges_deg) - 1
    bin_idx = np.clip(bin_idx, 0, len(edges_deg) - 2)
    return bin_idx


def compute_binned_tuning(sresp, bin_idx, n_bins):
    """
    Compute mean and SEM of neuron responses per orientation bin.

    sresp: (n_neurons, n_trials)
    bin_idx: (n_trials,) integer bin assignments

    Returns:
        tuning_mean: (n_neurons, n_bins)
        tuning_sem:  (n_neurons, n_bins)
    """
    n_neurons, n_trials = sresp.shape
    tuning_mean = np.zeros((n_neurons, n_bins), dtype=np.float32)
    tuning_sem = np.zeros((n_neurons, n_bins), dtype=np.float32)

    for b in range(n_bins):
        mask = bin_idx == b
        n = mask.sum()
        if n == 0:
            continue
        vals = sresp[:, mask]  # (n_neurons, n_trials_in_bin)
        tuning_mean[:, b] = vals.mean(axis=1)
        if n > 1:
            tuning_sem[:, b] = vals.std(axis=1, ddof=1) / np.sqrt(n)
        else:
            tuning_sem[:, b] = 0.0

    return tuning_mean, tuning_sem


def compute_split_half_reliability(sresp, bin_idx, n_bins):
    """
    Compute split-half reliability for each neuron.

    Split trials into odd/even halves by trial index.
    For each half, compute mean response per orientation bin.
    Return Pearson correlation between the two halves (per neuron).

    sresp: (n_neurons, n_trials)
    Returns: (n_neurons,) reliability values in [-1, 1]
    """
    n_neurons, n_trials = sresp.shape
    odd_mask = np.arange(n_trials) % 2 == 0   # "even" index (0-based)
    even_mask = ~odd_mask

    # Compute per-bin means for each half
    tuning_h1 = np.zeros((n_neurons, n_bins), dtype=np.float32)
    tuning_h2 = np.zeros((n_neurons, n_bins), dtype=np.float32)

    for b in range(n_bins):
        b_mask = bin_idx == b
        mask1 = b_mask & odd_mask
        mask2 = b_mask & even_mask
        if mask1.sum() > 0:
            tuning_h1[:, b] = sresp[:, mask1].mean(axis=1)
        if mask2.sum() > 0:
            tuning_h2[:, b] = sresp[:, mask2].mean(axis=1)

    # Per-neuron Pearson correlation across bins
    # Vectorized: subtract mean across bins
    h1 = tuning_h1 - tuning_h1.mean(axis=1, keepdims=True)
    h2 = tuning_h2 - tuning_h2.mean(axis=1, keepdims=True)
    num = (h1 * h2).sum(axis=1)
    denom = np.sqrt((h1**2).sum(axis=1) * (h2**2).sum(axis=1))
    reliability = np.where(denom > 0, num / denom, 0.0)
    return reliability.astype(np.float32)


def compute_empirical_pref_osi(tuning_mean, centers_deg):
    """
    Compute empirical preferred orientation and OSI from binned tuning curves.

    tuning_mean: (n_neurons, n_bins)
    centers_deg: (n_bins,) bin centers in degrees

    Returns:
        pref_deg: (n_neurons,) preferred orientation in degrees
        osi: (n_neurons,) orientation selectivity index
    """
    pref_idx = np.argmax(tuning_mean, axis=1)  # (n_neurons,)
    pref_deg = centers_deg[pref_idx]

    R_pref = tuning_mean[np.arange(len(pref_idx)), pref_idx]

    # Orthogonal bin: pref + 90° (wrapped to 0-180°)
    orth_deg = (pref_deg + 90.0) % 180.0
    orth_idx = np.argmin(np.abs(centers_deg[None, :] - orth_deg[:, None]), axis=1)
    R_orth = tuning_mean[np.arange(len(orth_idx)), orth_idx]

    eps = 1e-8
    osi = (R_pref - R_orth) / (R_pref + R_orth + eps)

    return pref_deg.astype(np.float32), osi.astype(np.float32)


def main():
    dat = load_raw()

    sresp = dat["sresp"].astype(np.float32)   # (n_neurons, n_trials)
    istim = dat["istim"].astype(np.float64)   # (n_trials,) in radians 0–2π

    n_neurons, n_trials = sresp.shape
    print(f"sresp: {sresp.shape}, istim: {istim.shape}")
    print(f"istim range: {istim.min():.3f} – {istim.max():.3f} rad "
          f"({np.degrees(istim.min()):.1f}° – {np.degrees(istim.max()):.1f}°)")

    # ------------------------------------------------------------------
    # Map direction (0–2π) to orientation (0–π = 0–180°)
    # istim represents the direction of the stimulus; for static gratings
    # θ and θ+π produce identical images, so orientation = istim % π
    # ------------------------------------------------------------------
    theta_rad_orientation = istim % np.pi                     # 0–π
    theta_deg_orientation = np.degrees(theta_rad_orientation) # 0–180

    neuron_ids = np.arange(n_neurons, dtype=np.int32)

    # ------------------------------------------------------------------
    # Binned tuning
    # ------------------------------------------------------------------
    edges_deg, centers_deg = make_orientation_bins(N_BINS)
    bin_idx = assign_bins(theta_deg_orientation, edges_deg)

    print("Computing binned tuning curves...")
    tuning_mean, tuning_sem = compute_binned_tuning(sresp, bin_idx, N_BINS)

    # ------------------------------------------------------------------
    # Empirical preferred orientation + OSI
    # ------------------------------------------------------------------
    print("Computing empirical pref orientation and OSI...")
    pref_deg_emp, osi_emp = compute_empirical_pref_osi(tuning_mean, centers_deg)

    # ------------------------------------------------------------------
    # Split-half reliability
    # ------------------------------------------------------------------
    print("Computing split-half reliability...")
    reliability = compute_split_half_reliability(sresp, bin_idx, N_BINS)

    # ------------------------------------------------------------------
    # Basic per-neuron stats
    # ------------------------------------------------------------------
    mean_response = sresp.mean(axis=1).astype(np.float32)
    response_std = sresp.std(axis=1).astype(np.float32)
    n_trials_per_neuron = np.full(n_neurons, n_trials, dtype=np.int32)

    # ------------------------------------------------------------------
    # A. decoder-ready matrix (trials × neurons)
    # ------------------------------------------------------------------
    X = sresp.T  # (n_trials, n_neurons)
    print(f"Saving orientation_decoder_ready.npz  X={X.shape}")
    np.savez_compressed(
        os.path.join(PROC, "orientation_decoder_ready.npz"),
        X=X,
        theta_deg=theta_deg_orientation.astype(np.float32),
        theta_rad=theta_rad_orientation.astype(np.float32),
        neuron_ids=neuron_ids,
        recording_id=np.array([RECORDING_ID]),
    )

    # ------------------------------------------------------------------
    # B. Neuron summary (initial — fits added by 04_fit_tuning.py)
    # ------------------------------------------------------------------
    print("Building neuron summary table...")
    df = pd.DataFrame({
        "neuron_id": neuron_ids,
        "recording_id": RECORDING_ID,
        "n_trials": n_trials_per_neuron,
        "mean_response": mean_response,
        "response_std": response_std,
        "pref_orientation_deg_empirical": pref_deg_emp,
        "osi_empirical": osi_emp,
        "split_half_reliability": reliability,
        # Fit columns placeholder — filled by 04_fit_tuning.py
        "fit_success": False,
        "fit_r2": np.nan,
        "fit_baseline": np.nan,
        "fit_amplitude": np.nan,
        "fit_pref_orientation_deg": np.nan,
        "fit_kappa_or_width": np.nan,
        "fit_method": "",
    })
    df.to_parquet(os.path.join(PROC, "orientation_neuron_summary.parquet"), index=False)
    df.to_csv(os.path.join(PROC, "orientation_neuron_summary.csv"), index=False)
    print(f"Saved neuron summary: {df.shape}")

    # ------------------------------------------------------------------
    # C. Binned tuning matrix
    # ------------------------------------------------------------------
    print("Saving orientation_binned_tuning.npz...")
    np.savez_compressed(
        os.path.join(PROC, "orientation_binned_tuning.npz"),
        tuning_mean=tuning_mean,
        tuning_sem=tuning_sem,
        angle_bin_centers_deg=centers_deg.astype(np.float32),
        neuron_ids=neuron_ids,
    )

    # ------------------------------------------------------------------
    # D. Metadata JSON
    # ------------------------------------------------------------------
    def file_mb(fname):
        p = os.path.join(PROC, fname)
        return round(os.path.getsize(p) / 1e6, 2) if os.path.exists(p) else None

    raw_size_mb = round(os.path.getsize(RAW) / 1e6, 2)

    metadata = {
        "source_doi": "10.1016/j.cell.2021.03.042",
        "source_url": "https://osf.io/ny4ut/download",
        "osf_project": "https://osf.io/hygbm/",
        "github_reference": "https://github.com/MouseLand/stringer-et-al-2019",
        "selected_recording_ids": [RECORDING_ID],
        "raw_files": {
            "stringer_orientations.npy": raw_size_mb
        },
        "processed_files": {
            "orientation_decoder_ready.npz": file_mb("orientation_decoder_ready.npz"),
            "orientation_neuron_summary.parquet": file_mb("orientation_neuron_summary.parquet"),
            "orientation_neuron_summary.csv": file_mb("orientation_neuron_summary.csv"),
            "orientation_binned_tuning.npz": file_mb("orientation_binned_tuning.npz"),
        },
        "orientation_range_deg": [0.0, 180.0],
        "orientation_mapping": "istim % pi  (direction 0-2pi mapped to orientation 0-pi)",
        "n_trials": int(n_trials),
        "n_neurons": int(n_neurons),
        "n_orientation_bins": N_BINS,
        "angle_bin_width_deg": 180.0 / N_BINS,
        "response_summary": (
            "Raw deconvolved calcium activity from dat['sresp'] (neurons x trials). "
            "Each trial value is a single scalar per neuron (already trial-summarized in the raw data). "
            "No additional response-window extraction needed."
        ),
        "random_seed": SEED,
    }
    with open(os.path.join(PROC, "orientation_project_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone. Outputs in data/processed/:")
    for fname in os.listdir(PROC):
        p = os.path.join(PROC, fname)
        print(f"  {fname}: {os.path.getsize(p)/1e6:.1f} MB")

    total_proc_mb = sum(
        os.path.getsize(os.path.join(PROC, f)) for f in os.listdir(PROC)
    ) / 1e6
    print(f"\nTotal processed: {total_proc_mb:.1f} MB  (raw: {raw_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
