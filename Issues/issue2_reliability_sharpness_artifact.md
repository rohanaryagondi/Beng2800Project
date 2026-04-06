# Issue 2: The Reliability-Sharpness Correlation Is Mathematically Trivial

## The Claim

The report presents Spearman r = 0.239 between tuning sharpness (kappa) and split-half reliability as evidence that "sharper tuning is associated with higher reliability," framing this as a biological finding.

## The Problem

Both kappa and split-half reliability are derived from the same underlying tuning curves. The relationship between them is **not** a biological discovery — it is a **mathematical consequence** of how these quantities are defined.

### Why the Correlation Is Expected

Split-half reliability measures how correlated the odd-trial and even-trial tuning curves are. A neuron with high kappa has a tuning curve with strong modulation (large peak-to-trough difference). When you split this into halves, the large signal structure dominates the noise, producing high correlation. A neuron with low kappa has a nearly flat tuning curve — the correlation between the two halves is dominated by noise fluctuations, producing low reliability.

**This is not "sharper tuning causes more reliable responses." It is "a strongly modulated signal is easier to measure reliably than a weak one."**

## Simulation Results

### Simulation 1: Noise independent of kappa

We generated 5,000 synthetic neurons with:
- True kappa drawn from Exponential(4), independent of noise
- Noise standard deviation drawn independently from Normal(2, 1)
- Actual von Mises fitting applied to recover kappa

**Result:** Spearman(fitted_kappa, reliability) = **0.896**

This is *far higher* than the 0.239 observed in the real data. Even with noise completely independent of kappa, the mathematical coupling between these metrics produces a massive correlation.

### Simulation 2: All neurons have the SAME kappa

We gave all 5,000 neurons identical kappa = 4 but varied noise levels.

**Result:** Spearman(fitted_kappa, reliability) = **-0.002** (essentially zero)

This confirms: the correlation arises from kappa variation itself, not from noise properties.

### Interpretation

The simulation shows that:

1. **Any** population with variable tuning sharpness will show a positive kappa-reliability correlation, regardless of biology. It is a mathematical identity of the measurement system.

2. The fact that the real data shows r = 0.239 — much *lower* than the expected 0.90 from the simulation — actually suggests there are **other factors counteracting** the expected relationship. This could include noise correlations, non-stationarity, or recording artifacts that decouple kappa from reliability.

3. The OLS model's R² = 0.055 is not evidence of a "modest but real" biological effect. It is evidence that the expected mathematical coupling is being *attenuated* by other factors.

## What About the OLS Control for Mean Response?

The report includes mean_response as a covariate, arguing it controls for signal-to-noise confounds. This is insufficient because:

1. Mean response is a proxy for SNR but not the same thing. Two neurons can have the same mean response but very different noise levels.
2. The fundamental issue isn't SNR — it's that kappa directly determines how much structure exists in the tuning curve, which directly determines how correlatable the halves are. No covariate can "control away" a mathematical identity.

## Impact on the Report

The report's second main finding — "sharper tuning is moderately associated with higher trial-to-trial reliability (Spearman r = 0.239)" — is not wrong, but it is misleadingly framed. The correlation is expected by construction and says nothing about neural biology that wasn't already implied by the definition of kappa.

### How the Correlation Changes with Quality Filtering

| R² threshold | n neurons | Spearman r |
|-------------|-----------|------------|
| >= 0.0      | 23,587    | 0.239      |
| >= 0.1      | 23,125    | 0.220      |
| >= 0.3      | 21,371    | 0.194      |
| >= 0.5      | 18,299    | 0.181      |

Removing low-quality fits reduces the correlation because it removes the most extreme kappa = 0 / low reliability neurons — the easiest cases of the mathematical coupling.

## Recommended Fix

1. **Reframe the finding:** Instead of "sharper tuning predicts reliability," state: "Kappa and reliability are positively correlated (r = 0.239), consistent with the mathematical expectation that strongly modulated signals are more reliably measurable. This relationship is weaker than expected from pure signal structure (simulation: r ~ 0.9), suggesting additional noise sources decouple tuning strength from measurement reliability."

2. **Include the simulation** as a supplementary analysis to demonstrate the expected relationship.

3. **A more interesting question** would be: after accounting for the expected mathematical coupling, do any neurons deviate significantly? Are there sharply tuned but unreliable neurons, or flat-tuned but highly reliable ones? What characterizes those outliers?

## Severity: HIGH

This is the most significant issue. The report presents a mathematical tautology as a biological finding. The fix requires reframing, not just numbers — the interpretation needs to fundamentally change.
