# Issue 1: 100% Fit Convergence Masks Untuned Neurons

## The Claim

The report states that the von Mises model was "successfully fit to all 23,589 neurons (100% convergence rate)."

## The Problem

The code in `scripts/04_fit_tuning.py` marks a fit as "failed" only if R² < -1.0 — an almost impossible threshold. This means neurons with essentially flat tuning curves (R² near 0) are counted as "successful" fits. A flat line with R² = 0.01 passes the filter.

## What We Found

### R² Lower Tail

| Threshold | Neurons | Percentage |
|-----------|---------|------------|
| R² < 0.0  | 2       | 0.0%       |
| R² < 0.05 | 194     | 0.8%       |
| R² < 0.1  | 464     | 2.0%       |
| R² < 0.2  | 1,208   | 5.1%       |
| R² < 0.3  | 2,218   | 9.4%       |
| R² < 0.5  | 5,290   | 22.4%      |

### Characteristics of Low-R² Neurons

- **R² < 0.1 (464 neurons):** Median kappa = 0.000, median reliability = 0.461. These neurons have zero fitted tuning sharpness — they are essentially untuned.
- **R² < 0.3 (2,218 neurons):** Median kappa = 0.000, median reliability = 0.481. Nearly 10% of the population.
- **Kappa < 0.1 (3,726 neurons, 15.8%):** Their median R² = 0.373. These are neurons where the optimizer converged but the fitted model shows effectively no orientation tuning.

### Cross-tabulation

- R² < 0.1 AND kappa < 0.5: **405 neurons** — clearly untuned
- R² < 0.3 AND kappa < 0.5: **1,461 neurons** — likely untuned or very weakly tuned

## Impact

1. The "100% convergence" claim is misleading. While `curve_fit` technically converged for all neurons, ~2,200 neurons (9.4%) have R² < 0.3 and are poorly described by the model.

2. The report's statement that "22% of neurons were broadly tuned (kappa < 1)" conflates two populations: neurons that are genuinely broadly tuned AND neurons that are simply not orientation-selective. These are biologically distinct.

3. Including untuned neurons inflates the lower tail of the reliability distribution, which amplifies the apparent kappa-reliability correlation (see Issue 2).

## Recommended Fix

1. **Report adjusted numbers:** State that while 100% of fits converged numerically, only 77.6% (R² > 0.5) or 90.6% (R² > 0.3) produced meaningful tuning fits.

2. **Separate "untuned" from "broadly tuned":** Neurons with R² < 0.3 and kappa < 0.5 should be labeled as "untuned" or "non-orientation-selective" rather than "broadly tuned."

3. **Re-run the reliability analysis** excluding neurons with R² < 0.3 (see Issue 2 for how this affects results).

## Severity: MEDIUM

This doesn't invalidate any finding, but it inflates the apparent quality of the dataset and contributes to the reliability correlation issue. The fix is straightforward: better reporting and filtering.
