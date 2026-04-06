# Issue Fix Plan — What Needs to Run on HPC

**Date:** 2026-04-05
**Branch:** `Apr5Issuesv1`

---

## Status of Each Issue

| Issue | What's Needed | Already Done? | Compute? |
|-------|--------------|---------------|----------|
| 1 (Untuned neurons) | Text/reporting fix in report | No | None — text only |
| 2 (Reliability tautology) | Simulation + reframe | **YES** — simulation ran (r=0.906), figure + CSV saved | None remaining |
| 3 (Decoder bias) | Random-neuron decoder on full 23,589 matrix | **NO** — killed on laptop (OOM loading 232 MB matrix) | **HPC needed** |
| 4 (Binned R² inflation) | Text clarification in report | No | None — text only |
| 5 (Minor language) | Text edits in report | No | None — text only |

---

## What Must Run on HPC (Issue 3)

### Task: Random-neuron population decoder

Load the full decoder matrix `orientation_decoder_ready.npz` (232 MB, shape 4598 × 23589) and run the OLS decoder with **randomly selected neurons from the full population** (not the pre-selected top-1000).

### Neuron counts to test
`[10, 25, 50, 100, 250, 500, 1000, 2000, 5000]`

### Protocol per neuron count
- **10 random subsets** (drawn without replacement from all 23,589 neurons)
- **5-fold CV** on each subset
- Record: MAE (circular, modulo 180°) and median AE per repeat
- Use `seed = 42`

### Decoder method
```
Y_train = [y_cos2, y_sin2]   (from decoder_targets.npz)
W = np.linalg.lstsq(X_train, Y_train)
Y_pred = X_test @ W
theta_pred = arctan2(Y_pred[:,1], Y_pred[:,0]) / 2  % pi
circular_error = (pred - true + pi/2) % pi - pi/2
MAE = mean(abs(circular_error)) in degrees
```

### Expected runtime
- 10 neurons: seconds
- 100 neurons: seconds
- 1000 neurons: ~30s
- 5000 neurons: ~2-5 min (lstsq on 3678 × 5000 matrix)
- Total: ~10-15 minutes on HPC

### Expected peak memory
- Full X matrix: ~430 MB (float32: 4598 × 23589 × 4 bytes)
- Plus targets and working memory: ~600 MB total
- Well within any HPC node

---

## What Needs to Change in the Report After HPC Results

### Issue 1 fix (text only)
- Change "successfully fit to all 23,589 neurons (100% convergence rate)" →
  "The fitting algorithm converged for all 23,589 neurons; however, 2,218 (9.4%) produced R² < 0.3 and are better characterized as non-orientation-selective."
- Separate "broadly tuned" (kappa < 1 AND R² > 0.3) from "untuned" (R² < 0.3)

### Issue 2 fix (reframe + add simulation figure)
- Change "sharper tuning is associated with higher reliability" →
  "Kappa and reliability are positively correlated (Spearman r = 0.239), but this is substantially weaker than the r ≈ 0.91 expected from signal structure alone (simulation). The attenuation suggests biological noise sources partially decouple tuning strength from measurement reliability."
- Add simulation figure to report
- Move from "biological finding" framing to "expected relationship, interesting deviation"

### Issue 3 fix (add random-neuron results) — NEEDS HPC DATA
- Present random-neuron decoder as the **primary** result
- Present top-1000 result as an **upper bound**
- Replace Figure 4 (decoder scaling) with a two-curve plot (random vs top-1000)
- Update Table 2 with both sets of numbers
- Update discussion: "With 1,000 randomly selected neurons, MAE = X.XX°; pre-selecting the most reliable neurons improves this to 2.80°"

### Issue 4 fix (text only)
- Add to methods: "R² values reflect fits to trial-averaged tuning curves (36 bins of ~128 trials), not single-trial predictions. Single-trial prediction accuracy would be lower."
- Report adjusted R² (0.718) alongside raw (0.750)

### Issue 5 fixes (text only)
- Replace "mouse V1" → "this V1 recording" throughout body text
- Soften Ganguli & Simoncelli citation
- Add heteroscedasticity caveat for OLS

---

## Files Needed on HPC

### From GitHub (uploaded to branch)
1. `data/processed/orientation_decoder_ready.npz` — 232 MB (the big one)
2. `data/processed/orientation_decoder_targets.npz` — 66 KB
3. `data/processed/orientation_decoder_ready_top1000.npz` — 9 MB (for comparison)
4. `data/processed/orientation_neuron_summary.parquet` — 1.8 MB
5. `data/processed/orientation_binned_tuning.npz` — 5.9 MB
6. `data/final/orientation_neuron_analysis.parquet` — 1.8 MB
7. `reports/tables/decoder_cv_summary.csv` — existing top-1000 results
8. `reports/tables/reliability_correlations.csv` — for report regeneration
9. `reports/tables/reliability_model_results.csv` — for report regeneration
10. `reports/tables/tuning_fit_summary.csv` — for report regeneration
11. `scripts/22_issue_fixes.py` — the fix script (Issue 3 portion)
12. `scripts/20_generate_report.py` — report generation script

### NOT needed
- `data/raw/stringer_orientations.npy` (937 MB) — not needed
- `data/processed/orientation_neuron_summary.csv` — parquet is sufficient

---

## Outputs Expected from HPC

1. `reports/tables/decoder_random_neurons_detail.csv` — per-repeat results
2. `reports/tables/issue3_decoder_comparison.csv` — combined summary
3. `reports/figures/issue3_decoder_random_vs_top1000.png` — two-curve scaling plot
4. `reports/BENG2800_Final_Report.docx` — regenerated with all fixes

---

## Verification Checklist

After HPC run completes:

- [ ] `decoder_random_neurons_detail.csv` exists and has rows for neuron counts [10, 25, 50, 100, 250, 500, 1000, 2000, 5000]
- [ ] Random-neuron MAE at 1000 neurons is between 3-8° (sanity check; should be ~5°)
- [ ] Random-neuron MAE at 10 neurons is between 20-40° (still below chance)
- [ ] Shuffle control MAE ≈ 45° (already confirmed)
- [ ] `issue3_decoder_random_vs_top1000.png` shows two distinct curves with random above top-1000
- [ ] Updated report mentions both random and top-1000 results
- [ ] Report reframes reliability finding with simulation context
- [ ] Report distinguishes untuned neurons from broadly tuned
- [ ] Report clarifies binned R² vs single-trial
- [ ] Report uses "this recording" not "mouse V1" for generalizations
