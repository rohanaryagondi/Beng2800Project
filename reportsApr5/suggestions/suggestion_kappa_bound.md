# Suggestion: Raise Kappa Upper Bound from 20 to 50

## Status

The user correctly noticed that the Apr5 report still says "bounded [0, 20]" for kappa. This was already identified in our investigation (see `Issues/v2_kappa_bound_investigation.md`), but the Apr5 report was generated before that fix was available.

## The Problem (Summary)

**1,909 neurons (8.1%) are piled up at kappa = 19.99** — they hit the upper bound constraint. This is visible as a spike at the right edge of the kappa histogram in Figure 2. The 95th and 99th percentiles of kappa are both exactly 20.0.

When refitted with bound = 50:
- Median kappa of ceiling neurons jumps from 20.0 to ~30
- Only 2.2% remain at the new ceiling
- R² improves by +0.03 for affected neurons
- Non-ceiling neurons are completely unaffected

When refitted with bound = 100:
- Median jumps to ~30 (similar to bound=50)
- Only 0.3% at ceiling
- But kappa > 50 means FWHM < 10 deg (< 2 bins), where fits are poorly constrained

## Recommendation: Raise to 50

**Why 50 and not higher:**
- At kappa = 50, FWHM ≈ 9.6 deg ≈ 2 bins — this is the resolution limit given 5-degree bins
- Beyond kappa = 50, the optimizer is fitting a sharp peak to 1–2 data points, where small noise fluctuations can push kappa from 40 to 80. The estimate becomes unreliable.
- A bound of 50 acts as sensible regularization

**Why not keep 20:**
- 8% of neurons at the ceiling is a significant artifact
- Figure 2's kappa distribution is visibly truncated — the right tail is an artifact, not biology
- The IQR and percentile statistics are distorted
- It's a one-line code change with a 15-minute refit

## Implementation

### Code change
In `scripts/04_fit_tuning.py`, line 59:
```python
# BEFORE
[ np.inf, np.inf, 20.0,  np.inf],

# AFTER
[ np.inf, np.inf, 50.0,  np.inf],
```

### Pipeline rerun
After changing the bound, rerun:
1. `scripts/04_fit_tuning.py` — refit all neurons (~15 min on M3 Pro)
2. `scripts/12_phase2_nonlinear_tuning.py` — regenerate analysis tables
3. `scripts/13_phase3_reliability.py` — recompute reliability correlation (will change slightly)
4. `scripts/15_phase5_synthesis.py` — regenerate figures
5. `scripts/31_generate_report_apr5.py` — regenerate report

### Text changes in the report
- Line 194: Change "bounded [0, 20]" to "bounded [0, 50]"
- The kappa distribution statistics (median, IQR, percentages) will update automatically from the CSV
- Figure 2 right panel will show a smoother right tail instead of a spike at 20
- The "20% very sharply tuned (kappa >= 10)" number may change slightly

### Also update in `scripts/30_hpc_issue_fixes.py`
Lines 146 and 171 also use kappa bound of 20 in the simulation. These should match the main analysis bound for consistency.

## Impact on Results

- **Figure 2:** The kappa histogram will have a smooth right tail instead of a pile-up at 20. More honest representation.
- **Kappa statistics:** Median will stay ~3.8 (not affected). IQR upper bound and high percentiles will increase.
- **Reliability correlation:** Negligible change — ceiling neurons all have similar reliability whether kappa is 20 or 30.
- **Decoder:** No change — decoder doesn't use kappa values.
- **Narrative:** The "heterogeneity" story gets stronger — the true range of kappa is wider than currently reported.
