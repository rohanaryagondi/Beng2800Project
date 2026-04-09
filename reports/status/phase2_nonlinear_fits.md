# Phase 2: Nonlinear Least-Squares Tuning Analysis

**Date:** 2026-03-31
**Runtime:** 3.2s

## Summary

All 23,589 neurons were previously fitted with the von Mises tuning model:

    r(theta) = b + a * exp(kappa * cos(2 * (theta - theta0)))

**No re-fitting was needed** — 100% of fits succeeded in the setup phase.

## Fit Quality Distribution

| Metric | Value |
|--------|-------|
| Successful fits | 23,589 / 23,589 (100%) |
| Mean R2 | 0.690 |
| Median R2 | 0.753 |
| R2 > 0.5 | 18,352 (77.8%) |
| R2 > 0.8 | 10,044 (42.6%) |
| R2 > 0.9 | 5,614 (23.8%) |

## Tuning Sharpness Distribution

| Metric | Value |
|--------|-------|
| Mean kappa | 6.95 |
| Median kappa | 3.83 |
| IQR | 6.92 |
| kappa < 1 (broadly tuned) | 5,210 (22.1%) |
| kappa 1-5 (moderate) | 8,786 (37.2%) |
| kappa 5-10 (sharp) | 4,850 (20.6%) |
| kappa >= 10 (very sharp) | 4,743 (20.1%) |

## Preferred Orientation Distribution

| Metric | Value |
|--------|-------|
| Mean preferred orientation | 91.9 deg |
| Std | 52.3 deg |
| Circular variance | 0.928 |

The preferred orientations are approximately uniformly distributed (circular variance ~ 0.928, where 1.0 = perfectly uniform), indicating no strong population-level orientation bias.

## Outputs Created

- `data/final/orientation_neuron_analysis.parquet` — 23,589 rows x 15 columns
- `data/final/orientation_neuron_analysis.csv`
- `reports/tables/tuning_fit_summary.csv`
- `reports/figures/tuning_examples_best_fits.png`
- `reports/figures/tuning_examples_moderate_fits.png`
- `reports/figures/tuning_examples_failed_or_noisy_fits.png`

## Biological Interpretation

The nonlinear least-squares tuning analysis reveals substantial **heterogeneity in orientation tuning** across mouse V1 neurons:

1. **Tuning sharpness varies widely.** The fitted kappa parameter spans from near 0 (essentially untuned) to the cap at 20 (very sharply tuned), with a median of 3.83 and an IQR of 6.92. This indicates that the V1 population contains a continuum of tuning widths, from broadly responsive neurons that fire to many orientations to narrowly tuned neurons that respond primarily to a single orientation.

2. **No population-level orientation bias.** Preferred orientations are distributed approximately uniformly across 0-180 deg, consistent with the idea that mouse V1 tiles all orientations without a strong cardinal bias (unlike cat or primate V1 where cardinal orientations can be overrepresented).

3. **Most neurons are well-described by the von Mises model.** With median R2 = 0.753 and 77.8% of neurons having R2 > 0.5, the single-peaked von Mises model captures the tuning of most neurons. The 9.3% with R2 < 0.3 likely include neurons with weak or complex (e.g., multi-peaked) orientation responses.

4. **The wide range of tuning sharpness supports efficient population coding.** A mixture of broadly and narrowly tuned neurons can support both coarse discrimination (via broadly tuned neurons) and fine discrimination (via sharply tuned neurons), consistent with theoretical models of optimal population coding.
