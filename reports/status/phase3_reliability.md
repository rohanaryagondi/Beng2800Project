# Phase 3: Reliability Analysis

**Date:** 2026-03-31
**Runtime:** 24.3s

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

### OLS Model (R2 = 0.0546)

| Parameter | Coefficient | SE | t | p |
|-----------|-----------|------|-------|--------|
| Intercept | 0.670336 | 0.002179 | 307.63 | 0.00e+00 |
| Kappa | 0.004006 | 0.000220 | 18.17 | 0.00e+00 |
| Mean response | 0.003977 | 0.000122 | 32.62 | 0.00e+00 |

### Correlations

| Method | r | p | 95% Bootstrap CI |
|--------|---|---|-----------------|
| Pearson | 0.1094 | 1.12e-63 | [0.0959, 0.1230] |
| Spearman | 0.2392 | 3.70e-304 | [0.2269, 0.2519] |

## Interpretation

**Yes, sharper tuning is associated with higher reliability.** The relationship is positive and statistically robust.

- The Spearman correlation between kappa and reliability is **0.239** (95% CI: [0.227, 0.252]), indicating a positive monotonic relationship.
- The OLS model shows that kappa has a significant effect on reliability (beta = 0.0040, p = 0.00e+00), even after controlling for mean response.
- Mean response also has a significant association with reliability (beta = 0.0040, p = 0.00e+00).
- The overall model R2 is 0.0546, indicating that kappa and mean response together explain about 5.5% of the variance in reliability.
- The binned analysis confirms the trend: neurons in higher kappa bins tend to have higher mean reliability.

**Effect size:** The correlation is moderate (0.239), suggesting that while sharper tuning is associated with higher reliability, there is substantial variability not captured by tuning sharpness alone.

**Note on reliability metric:** Split-half reliability can be bounded above by the signal-to-noise ratio of the responses. Neurons with higher mean responses tend to have higher SNR, which may partially confound the kappa-reliability relationship. The OLS model partially controls for this by including mean response as a covariate.

## Outputs

- `reports/tables/reliability_model_results.csv`
- `reports/tables/reliability_correlations.csv`
- `reports/figures/reliability_vs_sharpness_main.png`
- `reports/figures/reliability_vs_sharpness_binned.png`
- `reports/figures/reliability_vs_mean_response.png`
