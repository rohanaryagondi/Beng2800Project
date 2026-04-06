"""
Prepare decoder-ready matrices for the linear orientation decoding method.

Orientation is periodic modulo 180°, so we represent it as:
  y_cos2 = cos(2 * theta_rad)   [in orientation space, not direction]
  y_sin2 = sin(2 * theta_rad)

A linear decoder X @ W ≈ [y_cos2, y_sin2] can then be used.

Outputs:
  data/processed/orientation_decoder_targets.npz
  data/processed/orientation_decoder_ready_top1000.npz   (top 1000 most reliable neurons)
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
TOP_N = 1000


def main():
    # Load decoder-ready matrix
    dec_path = os.path.join(PROC, "orientation_decoder_ready.npz")
    if not os.path.exists(dec_path):
        print(f"ERROR: {dec_path} not found. Run 03_process.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading decoder-ready matrix...")
    d = np.load(dec_path)
    X = d["X"]                         # (n_trials, n_neurons)
    theta_deg = d["theta_deg"]          # (n_trials,)
    theta_rad = d["theta_rad"]          # (n_trials,) in [0, pi]
    neuron_ids = d["neuron_ids"]
    recording_id = d["recording_id"]

    n_trials, n_neurons = X.shape
    print(f"  X: {X.shape}, theta_deg range: {theta_deg.min():.1f}–{theta_deg.max():.1f}°")

    # Compute circular targets
    y_cos2 = np.cos(2.0 * theta_rad).astype(np.float32)  # (n_trials,)
    y_sin2 = np.sin(2.0 * theta_rad).astype(np.float32)  # (n_trials,)

    # NOTE: X is NOT stored here to avoid duplication with orientation_decoder_ready.npz.
    # Load X from that file when using the decoder. This keeps total processed size < 400 MB.
    print("Saving orientation_decoder_targets.npz (targets only — X from orientation_decoder_ready.npz)...")
    np.savez_compressed(
        os.path.join(PROC, "orientation_decoder_targets.npz"),
        y_cos2=y_cos2,
        y_sin2=y_sin2,
        theta_deg=theta_deg,
        theta_rad=theta_rad,
    )

    # Top-1000 reliable neurons subset
    summary_path = os.path.join(PROC, "orientation_neuron_summary.parquet")
    if os.path.exists(summary_path):
        df = pd.read_parquet(summary_path)
        reliability = df["split_half_reliability"].values.astype(np.float32)

        # Rank by reliability, take top N
        top_idx = np.argsort(reliability)[::-1][:TOP_N]
        top_idx_sorted = np.sort(top_idx)  # keep original neuron order
        X_top = X[:, top_idx_sorted]
        top_neuron_ids = neuron_ids[top_idx_sorted]

        print(f"Top-{TOP_N} neurons: reliability range "
              f"{reliability[top_idx_sorted].min():.3f}–{reliability[top_idx_sorted].max():.3f}")

        print(f"Saving orientation_decoder_ready_top{TOP_N}.npz...")
        np.savez_compressed(
            os.path.join(PROC, f"orientation_decoder_ready_top{TOP_N}.npz"),
            X=X_top,
            y_cos2=y_cos2,
            y_sin2=y_sin2,
            theta_deg=theta_deg,
            theta_rad=theta_rad,
            neuron_ids=top_neuron_ids,
            recording_id=recording_id,
        )
        print(f"  X_top: {X_top.shape}")
    else:
        print("WARNING: neuron summary not found, skipping top-1000 subset.")

    print("\nDecoder target files:")
    for fname in ["orientation_decoder_targets.npz",
                  f"orientation_decoder_ready_top{TOP_N}.npz"]:
        p = os.path.join(PROC, fname)
        if os.path.exists(p):
            print(f"  {fname}: {os.path.getsize(p)/1e6:.1f} MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
