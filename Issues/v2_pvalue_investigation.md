# V2 Issue: Are the P-Values Correct?

## The Concern

The report includes p-values like "p < 10^-300" for the Spearman correlation, and the OLS model has p-values reported as 0.0 (underflow). These seem impossibly small.

## Verdict: The p-values are numerically correct, but meaningless.

### Why 10^-300 Is Real

The Spearman test converts the correlation to a t-statistic:

    t = r * sqrt((n-2) / (1-r²))

With r = 0.239 and n = 23,589:

    t = 0.239 * sqrt(23587 / (1 - 0.057)) = 37.83

A t-statistic of 37.83 with 23,587 degrees of freedom produces a p-value of approximately **10^-303**. This is not a bug — scipy correctly computed it. The value is near the limit of float64 representation (smallest float64 ≈ 10^-308), which is why some p-values underflow to exactly 0.0.

### Verified via log-scale computation:

| Test | t-statistic | log10(p) |
|------|-------------|----------|
| Spearman (kappa vs reliability) | 37.83 | -303 |
| Pearson (kappa vs reliability) | 16.90 | -63 |
| OLS intercept | 307.63 | -inf (underflow) |
| OLS kappa coefficient | 18.17 | -73 |
| OLS mean_response coefficient | 32.62 | -228 |

All are consistent with the t-statistics and sample size.

### Why the p-values are meaningless despite being correct

With n = 23,589, **any nonzero effect** produces a vanishingly small p-value. To illustrate:

    Even r = 0.02 at n = 23,589 → t = 3.07, p = 0.002

A correlation of 0.02 is completely trivial — it explains 0.04% of variance — yet it would be "highly significant" at p < 0.01. At this sample size, p-values cannot distinguish meaningful effects from trivially small ones.

**The problem isn't that the p-values are wrong. It's that they're uninformative.** With n ≈ 24,000, statistical significance is guaranteed for any non-zero relationship. The report should de-emphasize p-values entirely and focus on effect sizes (R², correlation magnitude, confidence intervals).

## Recommended Fix

1. **Don't report exact p-values** when they're this small. Use "p < 10^-50" or just "p << 0.001" — the exact exponent carries no additional information.

2. **Emphasize effect sizes instead:**
   - Spearman r = 0.239 (CI: [0.227, 0.252])
   - OLS R² = 0.055
   - These tell the reader what matters: the effect is small but precisely estimated.

3. **Add context about sample size:** "With n = 23,589, all tested relationships are statistically significant; we focus on effect sizes and confidence intervals as more informative measures."

4. **The OLS p-values of exactly 0.0** in the CSV are a display issue. The code computes `2 * (1 - sp_stats.t.cdf(|t|, df))` which underflows to 0.0 for large t-statistics. If you need to report them, use the log-scale values: p ≈ 10^-73 for kappa, p ≈ 10^-228 for mean response.

## The Deeper Issue

The extreme p-values are a symptom of a broader problem: the report leans on statistical significance to validate findings that should be evaluated on effect size. The reliability-sharpness correlation (r = 0.239, R² = 0.055) is "significant" but explains only 5.5% of variance. Whether that matters depends on the scientific question, not the p-value.

## Severity: LOW (numerically correct, but framing should change)
