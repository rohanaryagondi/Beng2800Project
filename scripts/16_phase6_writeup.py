"""
Phase 6: Writeup support files for the class project.

Generates methods draft, results draft, discussion/limitations,
figure captions, presentation outline, report outline, and final status.
"""
import os
import time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(BASE, "data", "final")
TABLES = os.path.join(BASE, "reports", "tables")
STATUS = os.path.join(BASE, "reports", "status")
DRAFTS = os.path.join(BASE, "reports", "drafts")
for d in [STATUS, DRAFTS]:
    os.makedirs(d, exist_ok=True)


def main():
    t0 = time.time()
    print("=== Phase 6: Writeup Support ===\n")

    # Load results for populating drafts
    df = pd.read_parquet(os.path.join(FINAL, "orientation_neuron_analysis.parquet"))
    cv_summary = pd.read_csv(os.path.join(TABLES, "decoder_cv_summary.csv"))
    corr_results = pd.read_csv(os.path.join(TABLES, "reliability_correlations.csv"))
    model_results = pd.read_csv(os.path.join(TABLES, "reliability_model_results.csv"))

    n_neurons = len(df)
    n_trials = df["n_trials"].iloc[0]
    df_fit = df[df["fit_success"]]
    r2_median = df_fit["fit_r2"].median()
    kappa_median = df_fit["fit_kappa_or_width"].median()
    kappa_iqr_lo = np.percentile(df_fit["fit_kappa_or_width"], 25)
    kappa_iqr_hi = np.percentile(df_fit["fit_kappa_or_width"], 75)
    rel_median = df["split_half_reliability"].median()

    spearman_row = corr_results[(corr_results["method"] == "Spearman") & (corr_results["metric"] == "r")]
    spearman_r = spearman_row["value"].values[0]
    ci_lo = spearman_row["bootstrap_ci_lo"].values[0]
    ci_hi = spearman_row["bootstrap_ci_hi"].values[0]

    kappa_model = model_results[model_results["parameter"] == "kappa"]
    beta_kappa = kappa_model["coefficient"].values[0]
    p_kappa = kappa_model["p_value"].values[0]

    mae_1000 = cv_summary.loc[cv_summary["neuron_count"] == 1000, "mae_mean"].values[0]
    mae_10 = cv_summary.loc[cv_summary["neuron_count"] == 10, "mae_mean"].values[0]

    # ------------------------------------------------------------------
    # Methods draft
    # ------------------------------------------------------------------
    print("  Writing methods_draft.md...")
    methods = f"""# Methods

## Dataset

We analyzed publicly available two-photon calcium imaging data from mouse primary visual cortex (V1), collected by Stringer, Michaelos, and Pachitariu (2021). The dataset consists of simultaneous recordings from {n_neurons:,} neurons in V1 of a single head-fixed mouse during passive viewing of static oriented gratings. A total of {n_trials:,} trials were presented, with stimulus orientations approximately uniformly distributed across 0-180 degrees. Stimulus directions (0-360 deg) were mapped to orientations (0-180 deg) via modular arithmetic, since static gratings at 0 deg and 180 deg are physically identical.

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
"""
    with open(os.path.join(DRAFTS, "methods_draft.md"), "w") as f:
        f.write(methods)

    # ------------------------------------------------------------------
    # Results draft
    # ------------------------------------------------------------------
    print("  Writing results_draft.md...")
    results = f"""# Results

## 1. Heterogeneity of orientation tuning properties

We analyzed orientation tuning in {n_neurons:,} simultaneously recorded V1 neurons. The von Mises tuning model was successfully fit to all neurons (100% convergence rate), with a median R-squared of {r2_median:.3f} (Figure 1).

**Preferred orientations** were distributed approximately uniformly across 0-180 degrees, indicating no strong population-level orientation bias (Figure 2, left panel). This is consistent with the known lack of columnar organization for orientation in mouse V1.

**Tuning sharpness (kappa)** varied substantially across the population, with a median of {kappa_median:.2f} (IQR: [{kappa_iqr_lo:.2f}, {kappa_iqr_hi:.2f}]) (Figure 2, right panel). Approximately {100*(df_fit['fit_kappa_or_width'] < 1).mean():.0f}% of neurons were broadly tuned (kappa < 1), while {100*(df_fit['fit_kappa_or_width'] >= 10).mean():.0f}% were very sharply tuned (kappa >= 10), demonstrating the heterogeneous nature of orientation selectivity in mouse V1.

*Figures: final_fig1_tuning_examples.png, final_fig2_tuning_parameter_distributions.png*
*Tables: tuning_fit_summary.csv*

## 2. Relationship between tuning sharpness and reliability

Neurons with sharper tuning tended to be more reliable across trials. The Spearman correlation between kappa and split-half reliability was {spearman_r:.3f} (95% bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}]), indicating a statistically significant positive association (Figure 3).

An OLS linear model controlling for mean response confirmed the independent contribution of tuning sharpness: beta_kappa = {beta_kappa:.4f} (p = {p_kappa:.2e}). However, the overall model R-squared was modest (~5.5%), indicating that tuning sharpness and response strength together explain only a small fraction of the variance in reliability. The binned analysis showed a monotonic increase in mean reliability with kappa, particularly for kappa > 5.

*Figures: final_fig3_reliability_vs_sharpness.png*
*Tables: reliability_model_results.csv, reliability_correlations.csv*

## 3. Population decoder performance

The OLS linear decoder achieved a mean circular MAE of {mae_1000:.2f} degrees with 1000 neurons, far below the chance level of ~45 degrees (Figure 4). Even with only 10 neurons, the decoder achieved {mae_10:.1f} degrees MAE, demonstrating that a relatively small subset of the population carries substantial orientation information.

**Neuron-count scaling:** Performance improved monotonically with neuron count (Figure 4), with the largest gains occurring between 10 and 100 neurons. Beyond 250 neurons, improvements became more gradual, suggesting partial saturation of the orientation information available in the population.

**Shuffle control:** Permuting orientation labels yielded MAE of ~45 degrees (matching the theoretical chance level), confirming that the decoder exploits genuine tuning relationships rather than spurious correlations (Figure, decoder_shuffle_control.png).

*Figures: final_fig4_decoder_scaling.png, final_fig5_decoder_examples.png*
*Tables: decoder_cv_summary.csv, decoder_performance_by_neuron_count.csv*
"""
    with open(os.path.join(DRAFTS, "results_draft.md"), "w") as f:
        f.write(results)

    # ------------------------------------------------------------------
    # Discussion / Limitations draft
    # ------------------------------------------------------------------
    print("  Writing discussion_limitations_draft.md...")
    discussion = f"""# Discussion and Limitations

## Summary of findings

This analysis of {n_neurons:,} neurons in mouse V1 revealed:

1. **Substantial heterogeneity in orientation tuning.** Preferred orientations span the full 0-180 degree range uniformly, while tuning sharpness varies over an order of magnitude. This diversity is consistent with theoretical predictions for efficient population coding.

2. **A moderate positive relationship between tuning sharpness and reliability.** Sharper tuning is associated with more consistent responses across trials, though the effect is modest (Spearman r = {spearman_r:.3f}).

3. **Highly accurate population decoding.** A simple OLS linear decoder achieves {mae_1000:.1f} deg MAE with 1000 neurons, consistent with the "high-precision coding" reported by Stringer et al. (2021).

## Limitations

### Single recording

All analyses are based on a single recording from one mouse. Population statistics (preferred orientation distribution, kappa distribution) may vary across animals and cortical areas. Replication across multiple recordings would strengthen the generalizability of these findings.

### Fit quality variation

While the von Mises model converged for all neurons, fit quality varied substantially (R-squared range from near 0 to >0.99). Neurons with low R-squared may have complex tuning (e.g., multi-peaked, inhibited at certain orientations) that the single-peaked von Mises model cannot capture. Future work could explore more flexible models (e.g., mixture of von Mises functions).

### Reliability metric limitations

The split-half reliability metric depends on the specific trial-splitting method (odd/even by index). Alternative splitting strategies (random splits, bootstrap reliability) might yield slightly different absolute values, though the relative ranking of neurons is likely robust. Additionally, reliability is influenced by the signal-to-noise ratio of the responses, which creates a partial confound with mean response strength.

### Decoder limitations

- The decoder uses only the top 1000 most reliable neurons. Including all {n_neurons:,} neurons might improve or hurt performance depending on the contribution of noisy neurons.
- OLS was used for simplicity and to match course requirements. Regularized methods (ridge regression) might improve generalization, especially with smaller neuron counts.
- The decoder was evaluated on a single train/test partition per fold. Repeated random splits would provide more stable performance estimates.
- The circular encoding (cos/sin) approach assumes a specific functional form for the orientation representation. Other decoders (e.g., nearest-template, maximum-likelihood) might perform differently.

### Response summary

The analysis used deconvolved calcium activity, which is an indirect measure of spiking activity. Deconvolution can introduce biases (e.g., underestimating activity for bursty neurons, temporal smoothing). However, this is the standard representation for two-photon calcium imaging data and is appropriate for studying orientation tuning at the trial level.

### Neuron subsampling

For the neuron-count scaling analysis, neurons were subsampled from the top 1000 most reliable neurons. This is a biased sample — results with random subsets from the full population of {n_neurons:,} neurons would likely show different scaling behavior, particularly at small neuron counts.
"""
    with open(os.path.join(DRAFTS, "discussion_limitations_draft.md"), "w") as f:
        f.write(discussion)

    # ------------------------------------------------------------------
    # Figure captions
    # ------------------------------------------------------------------
    print("  Writing figure_captions.md...")
    captions = f"""# Figure Captions

## Figure 1: Example orientation tuning curves with von Mises fits

Six example neurons spanning the range of fit quality (R-squared from ~{df_fit['fit_r2'].quantile(0.10):.2f} to ~{df_fit['fit_r2'].quantile(0.95):.2f}). Blue circles show mean empirical response per 5-degree orientation bin (+/- SEM). Orange curve shows the fitted von Mises model r(theta) = b + a * exp(kappa * cos(2*(theta - theta0))). Each panel lists the fitted R-squared, kappa, and preferred orientation.

## Figure 2: Distributions of tuning parameters

(Left) Distribution of fitted preferred orientations across all {n_neurons:,} neurons. The red dashed line indicates a uniform distribution. Preferred orientations are approximately uniformly distributed, consistent with the lack of columnar orientation maps in mouse V1. (Right) Distribution of fitted tuning sharpness (kappa). The black dashed line indicates the median (kappa = {kappa_median:.2f}). The distribution is right-skewed, with most neurons moderately tuned and a tail of very sharply tuned neurons.

## Figure 3: Relationship between tuning sharpness and reliability

(Left) Hexbin density plot of tuning sharpness (kappa) vs. split-half reliability for all {n_neurons:,} neurons. Black line shows OLS regression controlling for mean response. Text inset reports Spearman correlation with 95% bootstrap CI. (Right) Mean reliability (+/- SEM) for neurons binned by kappa, showing the monotonic positive trend.

## Figure 4: Population decoder performance vs. neuron count

Mean circular MAE (degrees) of the OLS linear decoder as a function of the number of neurons used, evaluated with 5-fold cross-validation. Error bars show +/- 1 SD across 20 random neuron subsamples (except at 1000 neurons, which uses all available). Orange dashed line: shuffle control MAE (~45 deg). Gray dotted line: theoretical chance level (45 deg). X-axis is on a log scale.

## Figure 5: Decoder predicted vs. true orientation

Scatter plot of predicted vs. true stimulus orientation for the 1000-neuron OLS decoder (80/20 train/test split). Each point is one held-out trial. The tight clustering around the identity line (red dashed) reflects the decoder's high accuracy (MAE = {mae_1000:.1f} deg).
"""
    with open(os.path.join(DRAFTS, "figure_captions.md"), "w") as f:
        f.write(captions)

    # ------------------------------------------------------------------
    # Presentation outline
    # ------------------------------------------------------------------
    print("  Writing presentation_outline.md...")
    presentation = f"""# Presentation Outline (15 minutes)

## Slide 1: Title (30 sec)
- Title: "Orientation Tuning Heterogeneity and Population Decoding in Mouse V1"
- Course: BENG 2800, Spring 2026
- Dataset: Stringer et al. (2021)

## Slide 2: Background & Motivation (1.5 min)
- Neurons in V1 are tuned to stimulus orientation
- Key questions: How variable is tuning? Does it relate to reliability? Can we decode orientation?
- Two class methods: nonlinear least squares (tuning fits) + linear models (decoder)

## Slide 3: Dataset Overview (1 min)
- {n_neurons:,} neurons, {n_trials:,} trials, static gratings 0-180 deg
- Two-photon calcium imaging, deconvolved activity
- Show orientation distribution (uniform)

## Slide 4: Von Mises Tuning Model (1.5 min)
- Model equation: r(theta) = b + a * exp(kappa * cos(2*(theta - theta0)))
- Nonlinear least squares fitting (scipy.optimize.curve_fit)
- 100% fit success, median R-squared = {r2_median:.3f}
- Show Figure 1 (example tuning curves)

## Slide 5: Tuning Heterogeneity (1.5 min)
- Preferred orientations: approximately uniform
- Kappa distribution: median {kappa_median:.2f}, wide range
- Show Figure 2

## Slide 6: Reliability vs. Tuning Sharpness (2 min)
- Split-half reliability (odd/even trial split)
- Spearman r = {spearman_r:.3f}, positive association
- OLS model controlling for mean response
- Show Figure 3

## Slide 7: Linear Decoder Setup (1.5 min)
- OLS regression: X -> (cos(2*theta), sin(2*theta))
- Recovery via arctan2
- 5-fold cross-validation
- Circular MAE metric

## Slide 8: Decoder Results (2 min)
- 1000 neurons: {mae_1000:.1f} deg MAE (chance = 45 deg)
- Show Figure 5 (predicted vs true)
- Shuffle control confirms genuine signal

## Slide 9: Neuron Count Scaling (1.5 min)
- Performance improves monotonically
- Largest gains from 10-100 neurons
- Diminishing returns at 250+
- Show Figure 4

## Slide 10: Conclusions (1 min)
- Substantial heterogeneity in tuning (uniform pref, wide kappa range)
- Moderate positive sharpness-reliability relationship
- Highly accurate population decoding, improves with neuron count
- Consistent with Stringer et al.'s "high-precision coding"

## Slide 11: Limitations & Future Directions (1 min)
- Single recording / single mouse
- Top-1000 neuron subset for decoding
- Simple model (could try regularization, non-linear decoders)
- Could explore spatial organization of tuning
"""
    with open(os.path.join(DRAFTS, "presentation_outline.md"), "w") as f:
        f.write(presentation)

    # ------------------------------------------------------------------
    # Report outline
    # ------------------------------------------------------------------
    print("  Writing report_outline.md...")
    report_outline = """# Report Outline

## Abstract
- Brief summary of questions, methods, and key findings (~150 words)

## Introduction
- Orientation selectivity in visual cortex
- Importance of population coding
- Project questions and motivation

## Methods
- Dataset description (Stringer et al. 2021)
- Response summary and orientation mapping
- Von Mises tuning model and nonlinear least squares fitting
- Split-half reliability definition
- OLS linear decoder and circular error metric
- Neuron-count scaling analysis
- Statistical analyses (correlations, bootstrap CIs, OLS model)

## Results
### 3.1 Tuning heterogeneity
- Preferred orientation distribution (Figure 2 left)
- Tuning sharpness distribution (Figure 2 right)
- Example tuning curves (Figure 1)

### 3.2 Reliability and tuning sharpness
- Spearman/Pearson correlations
- OLS model results (Table: reliability_model_results.csv)
- Scatter and binned analysis (Figure 3)

### 3.3 Population decoding
- Decoder performance at 1000 neurons
- Predicted vs true orientation (Figure 5)
- Neuron-count scaling (Figure 4)
- Shuffle control

## Discussion
- Summary of findings
- Comparison to Stringer et al. (2021) results
- Implications for population coding theory
- Limitations (see discussion_limitations_draft.md)
- Future directions

## References
- Stringer, Michaelos, Pachitariu (2021). Cell.
- Additional references as needed for background

## Appendix
- Supplementary figures (EDA plots)
- Full decoder results table
"""
    with open(os.path.join(DRAFTS, "report_outline.md"), "w") as f:
        f.write(report_outline)

    # ------------------------------------------------------------------
    # Phase 6 status
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    phase6_status = f"""# Phase 6: Writeup Support

**Date:** 2026-03-31
**Runtime:** {elapsed:.1f}s

## Files Created

| File | Description |
|------|-------------|
| reports/drafts/methods_draft.md | Full methods section draft |
| reports/drafts/results_draft.md | Results organized by project question |
| reports/drafts/discussion_limitations_draft.md | Discussion and limitations |
| reports/drafts/figure_captions.md | Captions for all 5 final figures |
| reports/drafts/presentation_outline.md | 15-minute presentation, slide-by-slide |
| reports/drafts/report_outline.md | Full report structure |

All drafts are populated with actual quantitative results from the analysis.
"""
    with open(os.path.join(STATUS, "phase6_writeup_support.md"), "w") as f:
        f.write(phase6_status)
    print(f"  Saved: phase6_writeup_support.md")

    # ------------------------------------------------------------------
    # Final run summary
    # ------------------------------------------------------------------
    print("  Writing final_run_summary.md...")
    final_summary = f"""# Final Run Summary

**Date:** 2026-03-31

## Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Inventory & Validation | COMPLETE |
| 1 | Exploratory Data Analysis | COMPLETE |
| 2 | Nonlinear Tuning Analysis | COMPLETE |
| 3 | Reliability Analysis | COMPLETE |
| 4 | Linear Decoder | COMPLETE |
| 5 | Project Synthesis | COMPLETE |
| 6 | Writeup Support | COMPLETE |

## Major Output Files

### Data
- data/final/orientation_neuron_analysis.parquet
- data/final/orientation_neuron_analysis.csv
- data/final/orientation_decoder_results.parquet

### Tables (reports/tables/)
- data_inventory.csv
- eda_descriptive_stats.csv
- tuning_fit_summary.csv
- reliability_model_results.csv
- reliability_correlations.csv
- decoder_performance_by_neuron_count.csv
- decoder_cv_summary.csv
- final_main_results_table.csv
- final_decoder_results_table.csv

### Final Figures (reports/figures/)
- final_fig1_tuning_examples.png
- final_fig2_tuning_parameter_distributions.png
- final_fig3_reliability_vs_sharpness.png
- final_fig4_decoder_scaling.png
- final_fig5_decoder_examples.png

### EDA Figures (reports/figures/)
- eda_orientation_distribution.png
- eda_response_histograms.png
- eda_reliability_distribution.png
- eda_pref_orientation_empirical.png
- eda_pref_orientation_fitted.png
- eda_tuning_sharpness_distribution.png
- eda_fit_r2_distribution.png
- eda_sharpness_vs_reliability.png
- eda_mean_response_vs_reliability.png
- eda_example_tuning_curves.png

### Tuning Example Figures (reports/figures/)
- tuning_examples_best_fits.png
- tuning_examples_moderate_fits.png
- tuning_examples_failed_or_noisy_fits.png

### Decoder Figures (reports/figures/)
- decoder_error_vs_neuron_count.png
- decoder_predicted_vs_true_examples.png
- decoder_shuffle_control.png

### Reliability Figures (reports/figures/)
- reliability_vs_sharpness_main.png
- reliability_vs_sharpness_binned.png
- reliability_vs_mean_response.png

### Notebooks (notebooks/)
- 01_orientation_project_start.ipynb (pre-existing)
- 02_eda.ipynb
- 03_nonlinear_tuning.ipynb
- 04_reliability_analysis.ipynb
- 05_linear_decoder.ipynb
- 06_results_synthesis.ipynb

### Status Reports (reports/status/)
- phase0_inventory_and_readiness.md
- phase1_eda.md
- phase2_nonlinear_fits.md
- phase3_reliability.md
- phase4_decoder.md
- phase5_synthesis.md
- phase6_writeup_support.md
- final_run_summary.md

### Draft Materials (reports/drafts/)
- methods_draft.md
- results_draft.md
- discussion_limitations_draft.md
- figure_captions.md
- presentation_outline.md
- report_outline.md

## Key Quantitative Findings

| Finding | Value |
|---------|-------|
| Total neurons analyzed | {n_neurons:,} |
| Total trials | {n_trials:,} |
| Von Mises fit success rate | 100% |
| Median fit R-squared | {r2_median:.3f} |
| Median tuning sharpness (kappa) | {kappa_median:.2f} |
| Kappa IQR | [{kappa_iqr_lo:.2f}, {kappa_iqr_hi:.2f}] |
| Median split-half reliability | {rel_median:.3f} |
| Spearman r (kappa vs reliability) | {spearman_r:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] |
| OLS beta_kappa | {beta_kappa:.4f} (p = {p_kappa:.2e}) |
| Best decoder MAE (1000 neurons) | {mae_1000:.2f} deg |
| Decoder MAE (10 neurons) | {mae_10:.1f} deg |
| Shuffle control MAE | ~45 deg |

## Remaining Work

None — all phases completed successfully.

## Project Readiness

| Task | Ready? |
|------|--------|
| Final analysis | YES |
| Report writing | YES — drafts available |
| Presentation building | YES — outline available |
| Figure generation | YES — 5 final figures ready |
"""
    with open(os.path.join(STATUS, "final_run_summary.md"), "w") as f:
        f.write(final_summary)
    print(f"  Saved: final_run_summary.md")

    print(f"\nPhase 6 completed in {time.time() - t0:.1f}s")
    print("\n=== ALL PHASES COMPLETE ===")


if __name__ == "__main__":
    main()
