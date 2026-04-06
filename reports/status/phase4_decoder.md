# Phase 4: Linear Least-Squares Population Decoder

**Date:** 2026-03-31
**Runtime:** 28.5s

## Method

- **Model:** Ordinary least squares (OLS) via `np.linalg.lstsq`
- **Targets:** y_cos2 = cos(2*theta), y_sin2 = sin(2*theta)
- **Prediction:** theta_pred = arctan2(sin_pred, cos_pred) / 2, modulo pi
- **Evaluation:** 5-fold cross-validation
- **Error metric:** Circular MAE (degrees), modulo 180 deg
- **Neuron source:** Top 1000 most reliable neurons (split-half reliability 0.969-0.997)

## Neuron-Count Scaling Results

| Neurons | MAE (deg) | Std | Median AE (deg) | Repeats |
|---------|-----------|-----|------------------|---------|
| 10 | 19.96 | 4.30 | 12.05 | 20 |
| 25 | 10.27 | 1.53 | 6.62 | 20 |
| 50 | 6.33 | 0.66 | 4.67 | 20 |
| 100 | 4.48 | 0.24 | 3.51 | 20 |
| 250 | 3.44 | 0.10 | 2.80 | 20 |
| 500 | 2.98 | 0.09 | 2.43 | 20 |
| 1000 | 2.80 | n/a | 2.28 | 1 |

## Shuffle Control

- Shuffle MAE (100 neurons): **45.06 +/- 0.28 deg** (n=10)
- Real MAE (100 neurons): **4.48 deg** (vs chance ~45 deg)
- The decoder is clearly better than chance.

## Key Findings

1. **The decoder is substantially better than chance.** With 1000 neurons, MAE = 2.8 deg (chance = ~45 deg).

2. **Performance improves with neuron count.** MAE decreases from 20.0 deg (10 neurons) to 2.8 deg (1000 neurons).

3. **Gains appear to slow down at higher neuron counts.** The marginal improvement from adding neurons diminishes as the population grows, suggesting partial saturation.

4. **Shuffle control confirms signal.** Shuffled orientation labels yield MAE ~45.1 deg (near chance), confirming the decoder relies on genuine neural tuning.

## Outputs

- `data/final/orientation_decoder_results.parquet`
- `reports/tables/decoder_performance_by_neuron_count.csv`
- `reports/tables/decoder_cv_summary.csv`
- `reports/figures/decoder_error_vs_neuron_count.png`
- `reports/figures/decoder_predicted_vs_true_examples.png`
- `reports/figures/decoder_shuffle_control.png`
