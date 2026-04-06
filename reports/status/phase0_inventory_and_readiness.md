# Phase 0: Inventory and Data-Readiness Validation

**Date:** 2026-03-31
**Runtime:** 1.9s

## File Inventory

| File | Shape | Size (MB) |
|------|-------|-----------|
| orientation_neuron_summary.parquet | (23589, 15) | 1.88 |
| orientation_binned_tuning.npz | tuning_mean=(23589, 36), tuning_sem=(23589, 36), centers=(36,) | 6.14 |
| orientation_decoder_targets.npz | y_cos2=(4598,), y_sin2=(4598,) | 0.07 |
| orientation_decoder_ready_top1000.npz | X=(4598, 1000), neuron_ids=(1000,) | 9.43 |
| orientation_decoder_ready.npz | X=(4598, 23589), theta_deg=(4598,) | 231.66 |
| orientation_project_metadata.json | n/a | 0.001 |

## Key Dimensions

- **Neurons:** 23,589
- **Trials:** 4,598
- **Orientation bins:** 36 (5 deg each)
- **Orientation range:** 0.00 - 179.98 deg
- **Unique orientations:** ~1684
- **Trials per bin:** ~128

## Consistency Checks

- Neuron IDs match (summary vs binned): True
- Neuron IDs match (summary vs decoder): True
- Top-1000 IDs subset of full: True
- Theta alignment (decoder vs targets): True
- cos2/sin2 consistency: cos=True, sin=True
- Bin centers correct: True

## Quality Checks

- NaN values: none
- Duplicate neuron IDs: 0
- Zero-variance neurons: 0
- Near-zero variance (std < 0.01): 0
- Invalid empirical pref angles: 0
- Invalid fitted pref angles: 0
- Negative reliability neurons: 117

## Neuron Response Summary

- Nontrivial responses (mean > 1): 23536/23589 (99.8%)
- Fit success rate: 100.0%
- Median fit R²: 0.750
- Median split-half reliability: 0.785
- Median tuning sharpness (kappa): 3.826
- Fraction reliability > 0.5: 85.4%

## Data Readiness Assessment

**Data are ready for all project analyses:** YES

### Sufficiency for each analysis

- **Tuning curve fitting:** 23,589 successful fits (100%) — READY
- **Reliability analysis:** 20,137 neurons with reliability > 0.5 — READY
- **Decoding with neuron subsamples:** 1000 top neurons available, 4,598 trials — READY
