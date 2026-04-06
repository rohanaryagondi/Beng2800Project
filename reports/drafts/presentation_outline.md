# Presentation Outline (15 minutes)

## Slide 1: Title (30 sec)
- Title: "Orientation Tuning Heterogeneity and Population Decoding in Mouse V1"
- Course: BENG 2800, Spring 2026
- Dataset: Stringer et al. (2021)

## Slide 2: Background & Motivation (1.5 min)
- Neurons in V1 are tuned to stimulus orientation
- Key questions: How variable is tuning? Does it relate to reliability? Can we decode orientation?
- Two class methods: nonlinear least squares (tuning fits) + linear models (decoder)

## Slide 3: Dataset Overview (1 min)
- 23,589 neurons, 4,598 trials, static gratings 0-180 deg
- Two-photon calcium imaging, deconvolved activity
- Show orientation distribution (uniform)

## Slide 4: Von Mises Tuning Model (1.5 min)
- Model equation: r(theta) = b + a * exp(kappa * cos(2*(theta - theta0)))
- Nonlinear least squares fitting (scipy.optimize.curve_fit)
- 100% fit success, median R-squared = 0.750
- Show Figure 1 (example tuning curves)

## Slide 5: Tuning Heterogeneity (1.5 min)
- Preferred orientations: approximately uniform
- Kappa distribution: median 3.83, wide range
- Show Figure 2

## Slide 6: Reliability vs. Tuning Sharpness (2 min)
- Split-half reliability (odd/even trial split)
- Spearman r = 0.239, positive association
- OLS model controlling for mean response
- Show Figure 3

## Slide 7: Linear Decoder Setup (1.5 min)
- OLS regression: X -> (cos(2*theta), sin(2*theta))
- Recovery via arctan2
- 5-fold cross-validation
- Circular MAE metric

## Slide 8: Decoder Results (2 min)
- 1000 neurons: 2.8 deg MAE (chance = 45 deg)
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
