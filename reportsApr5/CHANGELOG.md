# Report Changelog — Apr 5 Version

## Changes from original report

### Issue 1: Untuned neurons (MEDIUM)
**Before:** "Successfully fit to all 23,589 neurons (100% convergence rate)"
**After:** Clarified that 2,218 (9.4%) have R² < 0.3 and are non-orientation-selective. Separated "untuned" from "broadly tuned."

### Issue 2: Reliability-sharpness tautology (HIGH)
**Before:** Presented Spearman r = 0.239 as evidence that "sharper tuning is associated with higher reliability" — framed as a biological finding.
**After:** Added simulation showing expected r = 0.906 when noise is independent of kappa. Reframed: the real r = 0.239 is *lower* than expected, suggesting biological noise sources decouple tuning strength from reliability. Added Figure 3 (simulation comparison).

### Issue 3: Decoder selection bias (HIGH)
**Before:** Headline result was 2.80° MAE using pre-selected top-1000 neurons. Scaling curve used only pre-selected pool.
**After:** Random-neuron decoder is now primary result (4.94° MAE at 1000 neurons). Top-1000 result (2.80°) presented as upper bound. Both curves shown in Figure 4. NEW FINDING: OLS overfits at >1,000 random neurons (MAE increases to 5.53° at 2000 and 7.71° at 5000), motivating regularization.

### Issue 4: Binned R² inflation (LOW-MEDIUM)
**Before:** Reported R² = 0.750 without context.
**After:** Clarified R² is on binned means (~128 trials/bin), not single-trial predictions. Added note that single-trial accuracy would be lower.

### Issue 5: Minor language (LOW)
- Changed "mouse V1" → "this V1 recording" / "this recording" for generalization claims
- Softened Ganguli & Simoncelli citation
- Added heteroscedasticity caveat for OLS reliability model
- Added OLS overfitting discussion for decoder

## Key numbers comparison

| Metric | Original | Apr 5 |
|--------|----------|-------|
| Decoder MAE (primary) | 2.80° (top-1000) | 4.94° (random 1000) |
| Decoder MAE (upper bound) | — | 2.80° (top-1000) |
| Reliability interpretation | Biological finding | Mathematical expectation, attenuated |
| Simulation Spearman r | — | 0.906 |
| Untuned neurons acknowledged | No | Yes (9.4%) |
| OLS overfitting noted | No | Yes (>1000 neurons) |
