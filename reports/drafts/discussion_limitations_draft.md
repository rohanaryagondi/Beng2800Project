# Discussion and Limitations

## Summary of findings

This analysis of 23,589 neurons in mouse V1 revealed:

1. **Substantial heterogeneity in orientation tuning.** Preferred orientations span the full 0-180 degree range uniformly, while tuning sharpness varies over an order of magnitude. This diversity is consistent with theoretical predictions for efficient population coding.

2. **A moderate positive relationship between tuning sharpness and reliability.** Sharper tuning is associated with more consistent responses across trials, though the effect is modest (Spearman r = 0.239).

3. **Highly accurate population decoding.** A simple OLS linear decoder achieves 2.8 deg MAE with 1000 neurons, consistent with the "high-precision coding" reported by Stringer et al. (2021).

## Limitations

### Single recording

All analyses are based on a single recording from one mouse. Population statistics (preferred orientation distribution, kappa distribution) may vary across animals and cortical areas. Replication across multiple recordings would strengthen the generalizability of these findings.

### Fit quality variation

While the von Mises model converged for all neurons, fit quality varied substantially (R-squared range from near 0 to >0.99). Neurons with low R-squared may have complex tuning (e.g., multi-peaked, inhibited at certain orientations) that the single-peaked von Mises model cannot capture. Future work could explore more flexible models (e.g., mixture of von Mises functions).

### Reliability metric limitations

The split-half reliability metric depends on the specific trial-splitting method (odd/even by index). Alternative splitting strategies (random splits, bootstrap reliability) might yield slightly different absolute values, though the relative ranking of neurons is likely robust. Additionally, reliability is influenced by the signal-to-noise ratio of the responses, which creates a partial confound with mean response strength.

### Decoder limitations

- The decoder uses only the top 1000 most reliable neurons. Including all 23,589 neurons might improve or hurt performance depending on the contribution of noisy neurons.
- OLS was used for simplicity and to match course requirements. Regularized methods (ridge regression) might improve generalization, especially with smaller neuron counts.
- The decoder was evaluated on a single train/test partition per fold. Repeated random splits would provide more stable performance estimates.
- The circular encoding (cos/sin) approach assumes a specific functional form for the orientation representation. Other decoders (e.g., nearest-template, maximum-likelihood) might perform differently.

### Response summary

The analysis used deconvolved calcium activity, which is an indirect measure of spiking activity. Deconvolution can introduce biases (e.g., underestimating activity for bursty neurons, temporal smoothing). However, this is the standard representation for two-photon calcium imaging data and is appropriate for studying orientation tuning at the trial level.

### Neuron subsampling

For the neuron-count scaling analysis, neurons were subsampled from the top 1000 most reliable neurons. This is a biased sample — results with random subsets from the full population of 23,589 neurons would likely show different scaling behavior, particularly at small neuron counts.
