# Phase 3: Reliability Analysis

**Date:** 2026-03-31
**Runtime:** 32.2s

## Research Question

Are neurons with sharper tuning also more reliable across trials?

## Methods

- **Reliability metric:** Split-half reliability (Pearson r between odd/even trial tuning curves)
- **Tuning sharpness:** Fitted von Mises kappa (larger = sharper)
- **Sample:** 23,589 neurons with successful fits and valid reliability

### OLS Linear Model

    reliability = beta0 + beta1 * kappa + beta2 * mean_response

Fitted via `np.linalg.lstsq`. Standard errors computed from residual variance and (X'X)^-1.

## Results

### OLS Model (R2 = 0.0426)

| Parameter | Coefficient | SE | t | p |
|-----------|-----------|------|-------|--------|
| Intercept | 0.688841 | 0.002024 | 340.34 | 0.00e+00 |
| Kappa | 0.000762 | 0.000142 | 5.37 | 7.76e-08 |
| Mean response | 0.003942 | 0.000123 | 32.12 | 0.00e+00 |

### Correlations

| Method | r | p | 95% Bootstrap CI |
|--------|---|---|-----------------|
| Pearson | 0.0261 | 6.16e-05 | [0.0110, 0.0416] |
| Spearman | 0.2368 | 7.86e-298 | [0.2243, 0.2494] |

## Interpretation

**Yes, sharper tuning is associated with higher reliability.** The relationship is positive and statistically robust.

- The Spearman correlation between kappa and reliability is **0.237** (95% CI: [0.224, 0.249]), indicating a positive monotonic relationship.
- The OLS model shows that kappa has a significant effect on reliability (beta = 0.0008, p = 7.76e-08), even after controlling for mean response.
- Mean response also has a significant association with reliability (beta = 0.0039, p = 0.00e+00).
- The overall model R2 is 0.0426, indicating that kappa and mean response together explain about 4.3% of the variance in reliability.
- The binned analysis confirms the trend: neurons in higher kappa bins tend to have higher mean reliability.

**Effect size:** The correlation is moderate (0.237), suggesting that while sharper tuning is associated with higher reliability, there is substantial variability not captured by tuning sharpness alone.

**Note on reliability metric:** Split-half reliability can be bounded above by the signal-to-noise ratio of the responses. Neurons with higher mean responses tend to have higher SNR, which may partially confound the kappa-reliability relationship. The OLS model partially controls for this by including mean response as a covariate.

## Outputs

- `reports/tables/reliability_model_results.csv`
- `reports/tables/reliability_correlations.csv`
- `reports/figures/reliability_vs_sharpness_main.png`
- `reports/figures/reliability_vs_sharpness_binned.png`
- `reports/figures/reliability_vs_mean_response.png`
