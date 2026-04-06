# Orientation Data Summary — QA Report

Generated: 2026-03-30 22:54:53

## Data Source

- **DOI:** 10.1016/j.cell.2021.03.042
- **URL:** https://osf.io/ny4ut/download
- **OSF project:** https://osf.io/hygbm/
- **GitHub reference:** https://github.com/MouseLand/stringer-et-al-2019
- **Recording ID(s):** stringer2019_v1_mouse1

## Files Downloaded

| File | Size |
|------|------|
| `data/raw/stringer_orientations.npy` | 936.83 MB |

**Total raw download size:** 936.8 MB

## Processed Files

| File | Size |
|------|------|
| `data/processed/orientation_binned_tuning.npz` | 6.1 MB |
| `data/processed/orientation_decoder_ready.npz` | 231.7 MB |
| `data/processed/orientation_decoder_ready_top1000.npz` | 9.4 MB |
| `data/processed/orientation_decoder_targets.npz` | 0.1 MB |
| `data/processed/orientation_neuron_summary.csv` | 4.6 MB |
| `data/processed/orientation_neuron_summary.parquet` | 1.9 MB |
| `data/processed/orientation_project_metadata.json` | 0.0 MB |

**Total processed size:** 253.8 MB

## Dataset Dimensions

- **Number of neurons:** 23,589
- **Number of trials:** 4,598
- **Orientation range:** 0.0° – 180.0°
- **Number of unique orientations (unique binned angles):** 1684
- **Orientation bins:** 36 bins × 5.0° each
- **Response summary:** Raw deconvolved calcium activity from dat['sresp'] (neurons x trials). Each trial value is a single scalar per neuron (already trial-summarized in the raw data). No additional response-window extraction needed.

## Orientation Distribution

Approximate trial counts per 10° orientation bin:
  - 0°–10°: 263 trials
  - 10°–20°: 253 trials
  - 20°–30°: 264 trials
  - 30°–40°: 259 trials
  - 40°–50°: 249 trials
  - 50°–60°: 243 trials
  - 60°–70°: 282 trials
  - 70°–80°: 244 trials
  - 80°–90°: 261 trials
  - 90°–100°: 275 trials
  - 100°–110°: 264 trials
  - 110°–120°: 254 trials
  - 120°–130°: 238 trials
  - 130°–140°: 272 trials
  - 140°–150°: 240 trials
  - 150°–160°: 253 trials
  - 160°–170°: 235 trials
  - 170°–180°: 249 trials

## Reliability Summary

- **Mean reliability:** 0.730
- **Median reliability:** 0.785
- **Std reliability:** 0.211
- **Fraction r > 0.5:** 85.37%
- **Fraction r > 0.2:** 97.61%
- **Fraction r < 0:** 0.50%
- **Missing/NaN:** 0

## Tuning Fit Summary

- **Fit method:** von_mises_nlls
- **Successful fits:** 23,589 / 23,589 (100.0%)
- **Median R² (successful fits):** 0.750
- **Mean R² (successful fits):** 0.688
- **Median κ (tuning sharpness):** 3.826

## Missing Data / Quality Issues

- No missing data in trial-level matrices (X shape=(4598, 23589))

## Suspicious Observations

- No major issues found. Dataset appears ready for analysis.

## Analysis Readiness

| Task | Status |
|------|--------|
| Nonlinear tuning curve fitting | ✅ Ready — `data/processed/orientation_binned_tuning.npz` + `orientation_neuron_summary.parquet` |
| Reliability analysis | ✅ Ready — `split_half_reliability` column in neuron summary |
| Linear population decoder | ✅ Ready — `orientation_decoder_targets.npz` with X, y_cos2, y_sin2 |
| Top-1000 neuron subset | ✅ Ready — `orientation_decoder_ready_top1000.npz` |