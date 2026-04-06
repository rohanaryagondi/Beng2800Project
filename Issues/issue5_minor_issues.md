# Issue 5: Minor Issues (Language, Citations, Diagnostics)

These are lower-priority issues that should be fixed in the final version but don't affect the core analysis.

---

## 5a. Single-Mouse Generalization

### Problem
The report generalizes to "mouse V1" throughout (e.g., "mouse V1 contains a diverse population of orientation-tuned neurons") but all data comes from a single recording session in one mouse. With n = 1 animal, there is zero statistical power to assess inter-animal variability.

### Fix
Replace "mouse V1" with "this V1 recording" or "in this mouse" throughout. In the introduction, note that replication across animals would be needed to establish generality. The limitations section mentions this but the body text should be consistent.

---

## 5b. Ganguli & Simoncelli Citation Overreach

### Problem
The report states: "This diversity is consistent with theoretical models of efficient population coding, where a mixture of broadly and narrowly tuned neurons supports both coarse and fine orientation discrimination (Ganguli and Simoncelli, 2014)."

Ganguli & Simoncelli proved that heterogeneous populations can be *optimal* under specific noise models and loss functions. Simply observing heterogeneity does not mean it is efficient or optimal — any random population would also be heterogeneous. The claim confuses "consistent with" and "evidence for."

### Fix
Soften to: "The observed heterogeneity in tuning sharpness is consistent with the premise that diverse tuning profiles may support population coding (Ganguli and Simoncelli, 2014), though testing whether this distribution is optimal would require comparing against theoretical predictions."

---

## 5c. Missing Residual Diagnostics for OLS Models

### Problem
Both the reliability OLS model (Phase 3) and the decoder (Phase 4) use OLS but neither checks:
- **Residual normality** — needed for valid p-values
- **Heteroscedasticity** — reliability is bounded [-1, 1], so variance depends on the mean
- **Residual patterns** — nonlinearity would indicate model misspecification

### Impact
With n = 23,589, the central limit theorem makes normality violations almost irrelevant for p-values. But the heteroscedasticity issue is structural: reliability near 1.0 has compressed variance. This could bias standard errors.

### Fix
For a course report, this is not critical. If you want to address it:
1. Add a brief note: "OLS assumes homoscedastic errors; given the bounded reliability metric, standard errors should be interpreted with caution."
2. Optionally, apply a Fisher z-transform to reliability before regression (arctan transformation to unbounded space).

---

## 5d. Chance Level Justification

### Current Status
The report claims chance = ~45 degrees, and the shuffle control confirms this (45.06 +/- 0.28 degrees). This is correct: for orientations uniform on [0, 180], random circular guesses produce expected MAE = 45 degrees.

### No Fix Needed
The shuffle control validates the theoretical expectation. This is actually well-handled in the report.

---

## Overall Severity: LOW

These are polishing issues. Fix them for a better report, but they don't affect analytical validity.
