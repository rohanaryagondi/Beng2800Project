# Phase 1: Exploratory Data Analysis

**Date:** 2026-03-31
**Runtime:** 1.9s

## Figures Generated

| # | File | Description |
|---|------|-------------|
| 1 | eda_orientation_distribution.png | Trial orientation distribution (36 bins) |
| 2 | eda_response_histograms.png | Mean response + response std histograms |
| 3 | eda_reliability_distribution.png | Split-half reliability distribution |
| 4 | eda_pref_orientation_empirical.png | Empirical preferred orientation distribution |
| 5 | eda_pref_orientation_fitted.png | Fitted preferred orientation distribution |
| 6 | eda_tuning_sharpness_distribution.png | Tuning sharpness (kappa) distribution |
| 7 | eda_fit_r2_distribution.png | Fit R-squared distribution |
| 8 | eda_sharpness_vs_reliability.png | Sharpness vs reliability hexbin scatter |
| 9 | eda_mean_response_vs_reliability.png | Mean response vs reliability hexbin scatter |
| 10 | eda_example_tuning_curves.png | 12 example tuning curves (stratified by R2) |

## Descriptive Statistics

| Metric | Value |
|--------|-------|
| Neuron count | 23,589 |
| Trial count | 4,598 |
| Mean reliability | 0.7298 |
| Median reliability | 0.7847 |
| Mean fit R2 | 0.6881 |
| Median fit R2 | 0.7502 |
| Fraction successful fits | 1.0000 |
| Mean tuning sharpness | 5.8632 |
| Median tuning sharpness | 3.8263 |
| Median OSI | 0.6043 |
| Fraction reliability > 0.5 | 0.8537 |

## Key Observations

- Orientations are approximately uniformly distributed across trials (~128 per bin)
- Response distributions are right-skewed (many low-response neurons, a long tail of highly active ones)
- Majority of neurons (85.4%) have split-half reliability > 0.5
- Preferred orientations appear roughly uniformly distributed (both empirical and fitted)
- Tuning sharpness (kappa) is right-skewed with median 3.83 — moderate tuning on average
- Fit quality is generally good (median R2 = 0.750) with a broad distribution

## Example Neurons Selected

Neurons stratified by fit R2 quartiles (3 per quartile):
- IDs: [np.int64(16044), np.int64(1992), np.int64(18489), np.int64(2260), np.int64(16902), np.int64(20437), np.int64(12274), np.int64(23018), np.int64(17145), np.int64(10787), np.int64(17911), np.int64(2574)]
