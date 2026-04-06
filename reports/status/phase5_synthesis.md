# Phase 5: Project Synthesis

**Date:** 2026-03-31
**Runtime:** 4.1s

## Final Figures Created

1. **final_fig1_tuning_examples.png** — 6 example tuning curves spanning quality range
2. **final_fig2_tuning_parameter_distributions.png** — Preferred orientation and kappa distributions
3. **final_fig3_reliability_vs_sharpness.png** — Scatter + binned reliability vs sharpness
4. **final_fig4_decoder_scaling.png** — Decoder error vs neuron count
5. **final_fig5_decoder_examples.png** — Predicted vs true orientation scatter

## Answers to Project Questions

### 1. How much do preferred orientation and tuning sharpness vary across neurons in mouse V1?

There is **substantial heterogeneity** in both parameters:

- **Preferred orientation** is distributed approximately uniformly across 0-180 deg, indicating no population-level orientation bias in this mouse V1 recording.
- **Tuning sharpness (kappa)** varies widely, with a median of 3.83 and an IQR of [1.31, 8.23]. About 22% of neurons are broadly tuned (kappa < 1), while 20% are very sharply tuned (kappa >= 10).
- The von Mises model fits most neurons well (median R-squared = 0.750), but fit quality itself is heterogeneous.

### 2. Are neurons with sharper tuning also more reliable across trials?

**Yes, there is a positive but moderate association.** Neurons with higher kappa tend to have higher split-half reliability:

- Spearman r = 0.239 (95% CI: [0.227, 0.252])
- This relationship holds after controlling for mean response in the OLS model (beta_kappa = 0.0040, p = 0.00e+00)
- However, kappa and mean response together explain only ~5.5% of reliability variance, indicating that other factors (noise correlations, non-stationarity, etc.) dominate trial-to-trial variability.

### 3. How accurately can a simple population decoder predict stimulus orientation from neural activity?

The OLS linear decoder performs **far above chance**:

- With 1000 neurons: MAE = 2.80 deg (chance = ~45 deg)
- This represents an ~16x improvement over random guessing
- Even 10 neurons achieve ~20 deg MAE

### 4. How does decoder performance change as the number of neurons increases?

Performance improves monotonically with neuron count, with **diminishing returns**:

| Neurons | MAE (deg) |
|---------|-----------|
| 10 | 19.96 |
| 25 | 10.27 |
| 50 | 6.33 |
| 100 | 4.48 |
| 250 | 3.44 |
| 500 | 2.98 |
| 1000 | 2.80 |

The largest gains occur from 10 to 100 neurons. Beyond 250 neurons, improvements are more gradual. Even at 1000 neurons, performance has not fully saturated, suggesting additional neurons could further improve decoding.

## Outputs

- `reports/tables/final_main_results_table.csv`
- `reports/tables/final_decoder_results_table.csv`
- 5 final figures in `reports/figures/`
