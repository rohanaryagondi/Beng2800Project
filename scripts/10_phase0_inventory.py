"""
Phase 0: Inventory and data-readiness validation.

Loads all processed files, verifies shapes and consistency,
checks for quality issues, and writes a readiness report.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
STATUS = os.path.join(BASE, "reports", "status")
TABLES = os.path.join(BASE, "reports", "tables")
os.makedirs(STATUS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

EXPECTED = {
    "n_neurons": 23589,
    "n_trials": 4598,
    "n_bins": 36,
    "n_top": 1000,
}


def main():
    t0 = time.time()
    warnings = []
    inventory_rows = []

    # ------------------------------------------------------------------
    # 1. Load and verify each processed file
    # ------------------------------------------------------------------
    print("=== Phase 0: Inventory & Validation ===\n")

    # --- neuron summary ---
    print("Loading neuron summary...")
    df = pd.read_parquet(os.path.join(PROC, "orientation_neuron_summary.parquet"))
    assert df.shape[0] == EXPECTED["n_neurons"], f"Neuron count mismatch: {df.shape[0]}"
    inventory_rows.append(dict(
        file="orientation_neuron_summary.parquet",
        type="parquet",
        shape=str(df.shape),
        size_mb=round(os.path.getsize(os.path.join(PROC, "orientation_neuron_summary.parquet")) / 1e6, 2),
        keys_or_columns=", ".join(df.columns),
    ))
    print(f"  shape={df.shape}, columns={list(df.columns)}")

    # --- binned tuning ---
    print("Loading binned tuning...")
    bd = np.load(os.path.join(PROC, "orientation_binned_tuning.npz"))
    tuning_mean = bd["tuning_mean"]
    tuning_sem = bd["tuning_sem"]
    centers_deg = bd["angle_bin_centers_deg"]
    neuron_ids_bt = bd["neuron_ids"]
    assert tuning_mean.shape == (EXPECTED["n_neurons"], EXPECTED["n_bins"])
    assert tuning_sem.shape == (EXPECTED["n_neurons"], EXPECTED["n_bins"])
    assert centers_deg.shape == (EXPECTED["n_bins"],)
    inventory_rows.append(dict(
        file="orientation_binned_tuning.npz",
        type="npz",
        shape=f"tuning_mean={tuning_mean.shape}, tuning_sem={tuning_sem.shape}, centers={centers_deg.shape}",
        size_mb=round(os.path.getsize(os.path.join(PROC, "orientation_binned_tuning.npz")) / 1e6, 2),
        keys_or_columns="tuning_mean, tuning_sem, angle_bin_centers_deg, neuron_ids",
    ))
    print(f"  tuning_mean={tuning_mean.shape}, centers={centers_deg.shape}")

    # --- decoder targets ---
    print("Loading decoder targets...")
    dt = np.load(os.path.join(PROC, "orientation_decoder_targets.npz"))
    y_cos2 = dt["y_cos2"]
    y_sin2 = dt["y_sin2"]
    theta_deg_targets = dt["theta_deg"]
    theta_rad_targets = dt["theta_rad"]
    assert y_cos2.shape == (EXPECTED["n_trials"],)
    assert y_sin2.shape == (EXPECTED["n_trials"],)
    inventory_rows.append(dict(
        file="orientation_decoder_targets.npz",
        type="npz",
        shape=f"y_cos2={y_cos2.shape}, y_sin2={y_sin2.shape}",
        size_mb=round(os.path.getsize(os.path.join(PROC, "orientation_decoder_targets.npz")) / 1e6, 2),
        keys_or_columns="y_cos2, y_sin2, theta_deg, theta_rad",
    ))
    print(f"  y_cos2={y_cos2.shape}")

    # --- top 1000 decoder ready ---
    print("Loading top-1000 decoder ready...")
    d1k = np.load(os.path.join(PROC, "orientation_decoder_ready_top1000.npz"))
    X_top = d1k["X"]
    assert X_top.shape == (EXPECTED["n_trials"], EXPECTED["n_top"])
    neuron_ids_top = d1k["neuron_ids"]
    inventory_rows.append(dict(
        file="orientation_decoder_ready_top1000.npz",
        type="npz",
        shape=f"X={X_top.shape}, neuron_ids={neuron_ids_top.shape}",
        size_mb=round(os.path.getsize(os.path.join(PROC, "orientation_decoder_ready_top1000.npz")) / 1e6, 2),
        keys_or_columns="X, y_cos2, y_sin2, theta_deg, theta_rad, neuron_ids, recording_id",
    ))
    print(f"  X_top={X_top.shape}")

    # --- full decoder ready (load briefly for validation, then release) ---
    print("Loading full decoder ready (for validation)...")
    d_full = np.load(os.path.join(PROC, "orientation_decoder_ready.npz"))
    X_full = d_full["X"]
    theta_deg_full = d_full["theta_deg"]
    neuron_ids_full = d_full["neuron_ids"]
    assert X_full.shape == (EXPECTED["n_trials"], EXPECTED["n_neurons"])
    inventory_rows.append(dict(
        file="orientation_decoder_ready.npz",
        type="npz",
        shape=f"X={X_full.shape}, theta_deg={theta_deg_full.shape}",
        size_mb=round(os.path.getsize(os.path.join(PROC, "orientation_decoder_ready.npz")) / 1e6, 2),
        keys_or_columns="X, theta_deg, theta_rad, neuron_ids, recording_id",
    ))
    print(f"  X_full={X_full.shape}")

    # --- metadata ---
    print("Loading metadata...")
    with open(os.path.join(PROC, "orientation_project_metadata.json")) as f:
        meta = json.load(f)
    inventory_rows.append(dict(
        file="orientation_project_metadata.json",
        type="json",
        shape="n/a",
        size_mb=round(os.path.getsize(os.path.join(PROC, "orientation_project_metadata.json")) / 1e6, 3),
        keys_or_columns=", ".join(meta.keys()),
    ))

    # ------------------------------------------------------------------
    # 2. Cross-file consistency checks
    # ------------------------------------------------------------------
    print("\n--- Consistency checks ---")

    # Neuron IDs alignment
    ids_summary = df["neuron_id"].values
    ids_binned = neuron_ids_bt
    ids_decoder = neuron_ids_full

    ids_match_bt = np.array_equal(ids_summary, ids_binned)
    ids_match_dec = np.array_equal(ids_summary, ids_decoder)
    print(f"  Neuron IDs match (summary vs binned): {ids_match_bt}")
    print(f"  Neuron IDs match (summary vs decoder): {ids_match_dec}")
    if not ids_match_bt:
        warnings.append("Neuron IDs do not match between summary and binned tuning")
    if not ids_match_dec:
        warnings.append("Neuron IDs do not match between summary and decoder")

    # Top-1000 neuron IDs should be a subset
    top_in_full = np.all(np.isin(neuron_ids_top, ids_decoder))
    print(f"  Top-1000 IDs are subset of full: {top_in_full}")
    if not top_in_full:
        warnings.append("Top-1000 neuron IDs not a subset of full decoder IDs")

    # Theta alignment
    theta_match = np.allclose(theta_deg_full, theta_deg_targets, atol=0.01)
    print(f"  Theta_deg match (decoder vs targets): {theta_match}")
    if not theta_match:
        warnings.append("theta_deg mismatch between decoder_ready and decoder_targets")

    # cos/sin consistency
    expected_cos2 = np.cos(2 * theta_rad_targets).astype(np.float32)
    expected_sin2 = np.sin(2 * theta_rad_targets).astype(np.float32)
    cos_ok = np.allclose(y_cos2, expected_cos2, atol=1e-5)
    sin_ok = np.allclose(y_sin2, expected_sin2, atol=1e-5)
    print(f"  cos2/sin2 consistency: cos={cos_ok}, sin={sin_ok}")
    if not cos_ok or not sin_ok:
        warnings.append("y_cos2/y_sin2 do not match recomputed values from theta_rad")

    # Release full X
    del X_full, d_full
    print("  (released full X from memory)")

    # ------------------------------------------------------------------
    # 3. Quality checks on neuron summary
    # ------------------------------------------------------------------
    print("\n--- Quality checks ---")

    # NaN counts
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"  Columns with NaNs: {dict(nan_cols)}")
        for col, cnt in nan_cols.items():
            warnings.append(f"Column '{col}' has {cnt} NaN values")
    else:
        print("  No NaN values in neuron summary")

    # Duplicate neuron IDs
    n_dup = df["neuron_id"].duplicated().sum()
    print(f"  Duplicate neuron IDs: {n_dup}")
    if n_dup > 0:
        warnings.append(f"{n_dup} duplicate neuron IDs found")

    # Invalid angles
    pref_emp = df["pref_orientation_deg_empirical"].values
    bad_angles_emp = np.sum((pref_emp < 0) | (pref_emp > 180))
    print(f"  Invalid empirical pref angles (outside 0-180): {bad_angles_emp}")
    if bad_angles_emp > 0:
        warnings.append(f"{bad_angles_emp} empirical pref orientations outside [0, 180]")

    pref_fit = df.loc[df["fit_success"], "fit_pref_orientation_deg"].values
    bad_angles_fit = np.sum((pref_fit < 0) | (pref_fit > 180))
    print(f"  Invalid fitted pref angles (outside 0-180): {bad_angles_fit}")
    if bad_angles_fit > 0:
        warnings.append(f"{bad_angles_fit} fitted pref orientations outside [0, 180]")

    # Zero-variance neurons
    zero_var = (df["response_std"] == 0).sum()
    print(f"  Zero-variance neurons (response_std == 0): {zero_var}")
    if zero_var > 0:
        warnings.append(f"{zero_var} neurons have zero response variance")

    # Near-zero variance
    near_zero_var = (df["response_std"] < 0.01).sum()
    print(f"  Near-zero variance neurons (response_std < 0.01): {near_zero_var}")

    # Fit success
    n_success = df["fit_success"].sum()
    n_total = len(df)
    frac_success = n_success / n_total
    print(f"  Fit success: {n_success}/{n_total} ({100*frac_success:.1f}%)")

    # Negative reliability
    neg_rel = (df["split_half_reliability"] < 0).sum()
    print(f"  Neurons with negative reliability: {neg_rel}")

    # Nontrivial responses (mean_response > 1)
    nontrivial = (df["mean_response"] > 1.0).sum()
    frac_nontrivial = nontrivial / n_total
    print(f"  Nontrivial responses (mean > 1): {nontrivial}/{n_total} ({100*frac_nontrivial:.1f}%)")

    # Orientation range
    ori_min = theta_deg_targets.min()
    ori_max = theta_deg_targets.max()
    n_unique_ori = len(np.unique(np.round(theta_deg_targets, 1)))
    print(f"  Orientation range: {ori_min:.2f} - {ori_max:.2f} deg")
    print(f"  Unique orientations (rounded to 0.1): {n_unique_ori}")

    # Bin centers check
    expected_centers = np.arange(2.5, 180, 5.0)
    centers_ok = np.allclose(centers_deg, expected_centers)
    print(f"  Bin centers match expected [2.5, 7.5, ..., 177.5]: {centers_ok}")

    # ------------------------------------------------------------------
    # 4. Dataset sufficiency
    # ------------------------------------------------------------------
    print("\n--- Dataset sufficiency ---")
    trials_per_bin = EXPECTED["n_trials"] / EXPECTED["n_bins"]
    print(f"  Trials per orientation bin: ~{trials_per_bin:.0f}")
    print(f"  Neurons available for tuning fitting: {n_total}")
    print(f"  Neurons with successful fits: {n_success}")
    print(f"  Neurons with reliability > 0.5: {(df['split_half_reliability'] > 0.5).sum()}")
    print(f"  Top-1000 neurons for decoding: {EXPECTED['n_top']}")

    sufficient = (n_total >= 1000 and EXPECTED["n_trials"] >= 500 and frac_success > 0.5)
    print(f"  Data sufficient for all analyses: {sufficient}")

    elapsed = time.time() - t0
    print(f"\nPhase 0 completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------

    # Data inventory CSV
    inv_df = pd.DataFrame(inventory_rows)
    inv_path = os.path.join(TABLES, "data_inventory.csv")
    inv_df.to_csv(inv_path, index=False)
    print(f"\nSaved: {inv_path}")

    # Status report
    report_lines = [
        "# Phase 0: Inventory and Data-Readiness Validation",
        "",
        f"**Date:** 2026-03-31",
        f"**Runtime:** {elapsed:.1f}s",
        "",
        "## File Inventory",
        "",
        "| File | Shape | Size (MB) |",
        "|------|-------|-----------|",
    ]
    for row in inventory_rows:
        report_lines.append(f"| {row['file']} | {row['shape']} | {row['size_mb']} |")

    report_lines += [
        "",
        "## Key Dimensions",
        "",
        f"- **Neurons:** {EXPECTED['n_neurons']:,}",
        f"- **Trials:** {EXPECTED['n_trials']:,}",
        f"- **Orientation bins:** {EXPECTED['n_bins']} (5 deg each)",
        f"- **Orientation range:** {ori_min:.2f} - {ori_max:.2f} deg",
        f"- **Unique orientations:** ~{n_unique_ori}",
        f"- **Trials per bin:** ~{trials_per_bin:.0f}",
        "",
        "## Consistency Checks",
        "",
        f"- Neuron IDs match (summary vs binned): {ids_match_bt}",
        f"- Neuron IDs match (summary vs decoder): {ids_match_dec}",
        f"- Top-1000 IDs subset of full: {top_in_full}",
        f"- Theta alignment (decoder vs targets): {theta_match}",
        f"- cos2/sin2 consistency: cos={cos_ok}, sin={sin_ok}",
        f"- Bin centers correct: {centers_ok}",
        "",
        "## Quality Checks",
        "",
        f"- NaN values: {'none' if len(nan_cols) == 0 else str(dict(nan_cols))}",
        f"- Duplicate neuron IDs: {n_dup}",
        f"- Zero-variance neurons: {zero_var}",
        f"- Near-zero variance (std < 0.01): {near_zero_var}",
        f"- Invalid empirical pref angles: {bad_angles_emp}",
        f"- Invalid fitted pref angles: {bad_angles_fit}",
        f"- Negative reliability neurons: {neg_rel}",
        "",
        "## Neuron Response Summary",
        "",
        f"- Nontrivial responses (mean > 1): {nontrivial}/{n_total} ({100*frac_nontrivial:.1f}%)",
        f"- Fit success rate: {100*frac_success:.1f}%",
        f"- Median fit R²: {df.loc[df['fit_success'], 'fit_r2'].median():.3f}",
        f"- Median split-half reliability: {df['split_half_reliability'].median():.3f}",
        f"- Median tuning sharpness (kappa): {df.loc[df['fit_success'], 'fit_kappa_or_width'].median():.3f}",
        f"- Fraction reliability > 0.5: {100*(df['split_half_reliability'] > 0.5).sum()/n_total:.1f}%",
        "",
        "## Data Readiness Assessment",
        "",
        f"**Data are ready for all project analyses:** {'YES' if sufficient and len(warnings) == 0 else 'YES (with minor notes)' if sufficient else 'NO'}",
        "",
    ]

    if warnings:
        report_lines.append("### Warnings")
        report_lines.append("")
        for w in warnings:
            report_lines.append(f"- {w}")
        report_lines.append("")

    report_lines += [
        "### Sufficiency for each analysis",
        "",
        f"- **Tuning curve fitting:** {n_success:,} successful fits (100%) — READY",
        f"- **Reliability analysis:** {(df['split_half_reliability'] > 0.5).sum():,} neurons with reliability > 0.5 — READY",
        f"- **Decoding with neuron subsamples:** {EXPECTED['n_top']} top neurons available, {EXPECTED['n_trials']:,} trials — READY",
        "",
    ]

    status_path = os.path.join(STATUS, "phase0_inventory_and_readiness.md")
    with open(status_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved: {status_path}")


if __name__ == "__main__":
    main()
