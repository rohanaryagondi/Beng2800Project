# Final Run Summary

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
| Total neurons analyzed | 23,589 |
| Total trials | 4,598 |
| Von Mises fit success rate | 100% |
| Median fit R-squared | 0.750 |
| Median tuning sharpness (kappa) | 3.83 |
| Kappa IQR | [1.31, 8.23] |
| Median split-half reliability | 0.785 |
| Spearman r (kappa vs reliability) | 0.239 [0.227, 0.252] |
| OLS beta_kappa | 0.0040 (p = 0.00e+00) |
| Best decoder MAE (1000 neurons) | 2.80 deg |
| Decoder MAE (10 neurons) | 20.0 deg |
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
