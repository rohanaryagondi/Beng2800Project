# Methods

## Dataset

We analyzed publicly available two-photon calcium imaging data from mouse primary visual cortex (V1), collected by Stringer, Michaelos, and Pachitariu (2021). The dataset consists of simultaneous recordings from 23,589 neurons in V1 of a single head-fixed mouse during passive viewing of static oriented gratings. A total of 4,598 trials were presented, with stimulus orientations approximately uniformly distributed across 0-180 degrees. Stimulus directions (0-360 deg) were mapped to orientations (0-180 deg) via modular arithmetic, since static gratings at 0 deg and 180 deg are physically identical.

**Data source:** Stringer et al. (2021) "High-precision coding in visual cortex." Cell. DOI: 10.1016/j.cell.2021.03.042. Data downloaded from OSF (https://osf.io/ny4ut/download).

## Response summary

Neural responses were provided as deconvolved calcium activity (one scalar per neuron per trial), extracted from the original two-photon imaging data using Suite2p. No additional temporal averaging was required, as the data were already trial-summarized.

## Orientation tuning curve estimation

For each neuron, we estimated an empirical orientation tuning curve by binning trials into 36 bins of 5 degrees each (centers at 2.5, 7.5, ..., 177.5 deg) and computing the mean response per bin (~128 trials per bin).

## Nonlinear least-squares tuning model

We fit a von Mises orientation tuning function to each neuron's binned tuning curve using nonlinear least squares (scipy.optimize.curve_fit):

    r(theta) = b + a * exp(kappa * cos(2 * (theta - theta0)))

where:
- b is the baseline activity
- a is the response amplitude above baseline
- kappa is the tuning sharpness (higher = sharper; bounded [0, 20])
- theta0 is the preferred orientation

Initial parameter estimates were derived from the empirical tuning curve: b = 10th percentile of responses, a = peak response minus b, theta0 = orientation of peak response, kappa = 1.0. The cos(2*(theta - theta0)) form provides 180-degree periodicity appropriate for orientation (as opposed to direction).

Goodness of fit was assessed using R-squared between the fitted curve and the empirical bin means. The fit was considered successful if curve_fit converged, R-squared > -1, and the fitted amplitude was positive.

## Split-half reliability

Trial-to-trial reliability was assessed using a split-half method. Trials were divided into odd-indexed and even-indexed subsets, and the mean tuning curve was computed separately for each half. Reliability was defined as the Pearson correlation between the two half-tuning curves.

## Relationship between tuning sharpness and reliability

We assessed the relationship between tuning sharpness (kappa) and reliability using:

1. **OLS linear model:** reliability = beta0 + beta1 * kappa + beta2 * mean_response, fitted via np.linalg.lstsq. Standard errors were computed from the residual variance and the inverse of the normal equations matrix. This model controls for mean response strength as a potential confound.

2. **Correlation analyses:** Pearson and Spearman correlations between kappa and reliability, with 95% bootstrap confidence intervals (5000 resamples, seed = 42).

## Linear population decoder

We implemented a linear least-squares population decoder to predict stimulus orientation from neural population activity. The decoder used ordinary least squares (np.linalg.lstsq) to regress the trial-by-neuron response matrix onto two target vectors: y_cos = cos(2*theta) and y_sin = sin(2*theta). Predicted orientation was recovered as theta_pred = arctan2(sin_pred, cos_pred) / 2, mapped to [0, pi).

**Evaluation:** 5-fold cross-validation with circular mean absolute error (MAE) as the primary metric. Circular error was computed modulo 180 degrees: err = (pred - true + pi/2) mod pi - pi/2.

**Neuron-count scaling:** We evaluated decoder performance for neuron counts of 10, 25, 50, 100, 250, 500, and 1000 (the top 1000 most reliable neurons). For each count, 20 random neuron subsamples were drawn (except at 1000, where only one sample is possible), and 5-fold CV was run on each.

**Shuffle control:** At 100 neurons, we permuted orientation labels 10 times and re-ran the decoder to establish a null baseline.

## Software and reproducibility

All analyses were performed in Python 3.13 using NumPy, SciPy, Pandas, and Matplotlib. Random seed = 42 was used throughout.
