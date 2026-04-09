# Suggestion: Address Suite2p Deconvolution Quality and Potential Additional Denoising

## The Observation

Looking at Figure 1, neuron 15732 (and others like it) shows noticeable scatter around the von Mises fit despite having reasonable tuning (R² = 0.76, reliability = 0.87). The question: is Suite2p's deconvolution leaving residual noise that deflates kappa estimates and fit quality?

## What We Found About Neuron 15732

| Metric | Value | Percentile |
|--------|-------|------------|
| R² | 0.759 | 51st (exactly median) |
| Kappa | 2.02 | 32nd (somewhat broad) |
| Reliability | 0.874 | decent |
| Mean response | 13.14 | high |
| Tuning curve range | 3.7 – 42.8 | large dynamic range |
| Mean SEM per bin | 2.53 | moderate |
| SNR (range / mean SEM) | 15.5 | good |

This neuron has good SNR (15.5) and large dynamic range, but its R² is only at the median. The residual scatter in the tuning curve — visible noise fluctuation around the smooth von Mises curve — suggests the bin-averaged responses still carry noise. Since each bin averages ~128 trials, this noise is from trial-to-trial variability that persists after deconvolution.

## Is Additional Denoising Warranted?

### Arguments for:

1. **Suite2p's deconvolution is designed for spike extraction, not tuning analysis.** It removes the calcium indicator dynamics (slow rise, exponential decay) to approximate spike rates, but it doesn't specifically target trial-to-trial noise sources like neuropil contamination, motion artifacts, or shared variability.

2. **Neuropil subtraction quality varies.** Suite2p applies a neuropil correction factor, but the default coefficient (0.7) may not be optimal for all neurons. Residual neuropil signal could add orientation-correlated noise from nearby neurons.

3. **Modern denoising methods exist.** DeepInterpolation (Lecoq et al., 2021) and related deep-learning denoisers can remove independent noise from calcium movies before ROI extraction. Cascade (Rupprecht et al., 2021) offers improved spike inference. These could reduce the noise floor in tuning curves.

4. **Impact on kappa estimates:** If noise flattens tuning curves, it systematically deflates kappa. This would affect the entire kappa distribution, particularly for neurons near the moderate-kappa range (like neuron 15732 at kappa = 2.0, which might be sharper in reality).

### Arguments against (or caveats):

1. **We're working with pre-processed data.** The Stringer et al. (2021) dataset provides deconvolved traces, not raw calcium movies. Applying additional denoising to the deconvolved output is methodologically questionable — you'd be denoising a derived signal, not the raw data.

2. **Bin-averaging already reduces noise substantially.** With ~128 trials per bin, the SEM is reduced by sqrt(128) ≈ 11x compared to single trials. The residual scatter in Figure 1 is small relative to the signal.

3. **Median R² = 0.75 is typical for calcium imaging.** This is actually good by field standards. Electrophysiology would give higher R², but calcium imaging inherently adds noise from indicator kinetics, neuropil, and deconvolution.

4. **Re-processing the raw data is out of scope.** The raw calcium movies aren't in the dataset — only the deconvolved traces. True denoising would require going back to the raw .tiff stacks.

## Recommended Action for the Report

**Don't reprocess the data.** Instead, add a limitations paragraph acknowledging the issue:

### Suggested text for Section 4.2 Limitations:

Add this as a new bullet point (or expand the existing calcium imaging bullet):

> Deconvolved calcium activity, while standard for population-level analyses, is an indirect measure of spiking that introduces noise at multiple stages: indicator kinetics, neuropil contamination, and the deconvolution algorithm itself. Residual noise after deconvolution may systematically deflate tuning sharpness estimates, as noise fluctuations flatten the tuning curve peak. Additional denoising approaches (e.g., DeepInterpolation applied to raw imaging data) could improve tuning curve estimates, but would require access to the raw calcium movies, which are not included in the public dataset.

### Suggested text for Section 2.1 Dataset:

Expand the existing sentence about deconvolution:

> Neural responses were provided as deconvolved calcium activity (one scalar per neuron per trial) extracted using Suite2p, which applies neuropil subtraction and a non-negative deconvolution algorithm. We note that deconvolved traces retain trial-to-trial variability from noise sources including neuropil contamination and indicator kinetics; our analyses characterize tuning properties as measured through this processing pipeline.

## Impact

Text-only change. Acknowledges a real limitation without requiring reanalysis. Demonstrates awareness of the processing chain's impact on results — good for a course report.
