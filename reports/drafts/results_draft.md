# Results

## 1. Heterogeneity of orientation tuning properties

We analyzed orientation tuning in 23,589 simultaneously recorded V1 neurons. The von Mises tuning model was successfully fit to all neurons (100% convergence rate), with a median R-squared of 0.750 (Figure 1).

**Preferred orientations** were distributed approximately uniformly across 0-180 degrees, indicating no strong population-level orientation bias (Figure 2, left panel). This is consistent with the known lack of columnar organization for orientation in mouse V1.

**Tuning sharpness (kappa)** varied substantially across the population, with a median of 3.83 (IQR: [1.31, 8.23]) (Figure 2, right panel). Approximately 22% of neurons were broadly tuned (kappa < 1), while 20% were very sharply tuned (kappa >= 10), demonstrating the heterogeneous nature of orientation selectivity in mouse V1.

*Figures: final_fig1_tuning_examples.png, final_fig2_tuning_parameter_distributions.png*
*Tables: tuning_fit_summary.csv*

## 2. Relationship between tuning sharpness and reliability

Neurons with sharper tuning tended to be more reliable across trials. The Spearman correlation between kappa and split-half reliability was 0.239 (95% bootstrap CI: [0.227, 0.252]), indicating a statistically significant positive association (Figure 3).

An OLS linear model controlling for mean response confirmed the independent contribution of tuning sharpness: beta_kappa = 0.0040 (p = 0.00e+00). However, the overall model R-squared was modest (~5.5%), indicating that tuning sharpness and response strength together explain only a small fraction of the variance in reliability. The binned analysis showed a monotonic increase in mean reliability with kappa, particularly for kappa > 5.

*Figures: final_fig3_reliability_vs_sharpness.png*
*Tables: reliability_model_results.csv, reliability_correlations.csv*

## 3. Population decoder performance

The OLS linear decoder achieved a mean circular MAE of 2.80 degrees with 1000 neurons, far below the chance level of ~45 degrees (Figure 4). Even with only 10 neurons, the decoder achieved 20.0 degrees MAE, demonstrating that a relatively small subset of the population carries substantial orientation information.

**Neuron-count scaling:** Performance improved monotonically with neuron count (Figure 4), with the largest gains occurring between 10 and 100 neurons. Beyond 250 neurons, improvements became more gradual, suggesting partial saturation of the orientation information available in the population.

**Shuffle control:** Permuting orientation labels yielded MAE of ~45 degrees (matching the theoretical chance level), confirming that the decoder exploits genuine tuning relationships rather than spurious correlations (Figure, decoder_shuffle_control.png).

*Figures: final_fig4_decoder_scaling.png, final_fig5_decoder_examples.png*
*Tables: decoder_cv_summary.csv, decoder_performance_by_neuron_count.csv*
