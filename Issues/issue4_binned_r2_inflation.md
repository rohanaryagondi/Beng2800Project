# Issue 4: In-Sample R² on Binned Means Overstates Fit Quality

## The Claim

The report states "median R² of 0.750" for von Mises fits, implying the model explains 75% of the variance in neural responses.

## The Problem

The R² is computed on **binned tuning curve means** (36 bins, each averaging ~128 trials), not on single-trial responses. Averaging 128 trials per bin dramatically reduces noise, making any smooth model look good on the means even if single-trial predictions are poor.

Additionally, with 4 free parameters and only 36 data points, there is mild overfitting that inflates R².

## What We Found

### Adjusted R² (correcting for degrees of freedom)

- **Original R² median:** 0.7502
- **Adjusted R² median:** 0.7179
- **Difference:** 0.032

The adjustment is modest (n/p ratio = 9.0 is reasonably large), so overfitting per se is not a major concern.

### Signal-to-Noise Analysis

- **Median signal variance** (across bins): 12.45
- **Median noise variance** (SEM²): 1.72
- **Median SNR:** 7.0

The binned means have high SNR because averaging 128 trials reduces noise by a factor of sqrt(128) ≈ 11.3.

### Approximate Single-Trial R²

Using the relationship between binned and single-trial variance:

- **Approximate single-trial R² median: 0.648**
- **vs. binned R² median: 0.750**

The binned R² overstates predictive performance by about 10 percentage points in absolute terms. A neuron that appears to have "75% of variance explained" at the bin level would have roughly 65% at the single-trial level.

## How Significant Is This?

**Moderate.** The inflation exists but is not catastrophic:

1. The 0.032 difference from degrees-of-freedom adjustment is minor.
2. The ~10 percentage point gap between binned and single-trial R² is meaningful but does not change the qualitative story. Most neurons are still reasonably well-fit.
3. The report never claims the R² applies to single trials, so it is not technically wrong — just potentially misleading to readers who might assume it does.

## Recommended Fix

1. **Clarify in the methods** that R² is computed on binned means (36 bins of ~128 trials each), not single-trial responses.

2. **Report adjusted R²** alongside raw R² (0.718 vs 0.750) to acknowledge the 4-parameter model.

3. **Add a sentence noting** that single-trial prediction accuracy would be lower due to trial-to-trial variability. Something like: "Note that this R² reflects the model's fit to trial-averaged tuning curves; single-trial responses are substantially noisier."

4. **This does NOT require re-running analyses.** It's a reporting/framing fix.

## Severity: LOW-MEDIUM

The inflation exists but is modest and doesn't change conclusions. The fix is purely in the writing — add context about what the R² metric actually measures.
