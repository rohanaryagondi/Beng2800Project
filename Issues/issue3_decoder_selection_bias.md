# Issue 3: Decoder Performance Is Substantially Inflated by Neuron Selection Bias

## The Claim

The report's headline decoder result is 2.80 degrees MAE with 1,000 neurons, described as "remarkably accurate orientation prediction." The scaling analysis shows improvement from 19.96 degrees (10 neurons) to 2.80 degrees (1,000 neurons).

## The Problem

All neurons used in decoding are drawn from the **top 1,000 most reliable neurons** in the entire recording (split-half reliability 0.969–0.997). This is the top 4.2% of the population. The scaling analysis (10, 25, 50, ..., 1000) draws subsets from this pre-selected pool, meaning even "10 neurons" actually means "10 of the best neurons in 23,589."

## What We Found

We re-ran the decoder using neurons randomly selected from the full population of 23,589, with 10 repeats per neuron count and 5-fold CV.

### Comparison Table

| Neurons | Top-1000 Pool MAE | Random Population MAE | Difference |
|---------|------------------|-----------------------|------------|
| 10      | 19.96 +/- 3.90   | 30.85 +/- 2.98        | +10.88 deg |
| 50      | 6.23 +/- 0.48    | 15.81 +/- 1.68        | +9.58 deg  |
| 100     | 4.42 +/- 0.37    | 10.66 +/- 0.70        | +6.24 deg  |
| 500     | 2.97 +/- 0.05    | 5.48 +/- 0.30         | +2.51 deg  |
| 1000    | 2.80 +/- 0.00    | 4.95 +/- 0.14         | +2.15 deg  |

### Key Finding

**At every neuron count, random selection from the full population performs substantially worse:**

- At 10 neurons: random is **55% worse** (30.85 vs 19.96 degrees)
- At 100 neurons: random is **141% worse** (10.66 vs 4.42 degrees)
- At 1000 neurons: random is **77% worse** (4.95 vs 2.80 degrees)

The 1000-neuron random decoder still achieves 4.95 degrees MAE — well above chance (45 degrees) and still impressive — but nearly double the reported 2.80 degrees.

### The Scaling Curve Shape Changes Too

With random neurons, the performance at 1000 neurons (4.95 degrees) is comparable to what the top-1000 pool achieves with just 100 neurons (4.42 degrees). The "diminishing returns" narrative changes: with random neurons, gains continue more steeply at higher neuron counts because you're adding progressively more informative neurons to a mixed-quality pool.

## Impact on the Report

1. **The headline number (2.80 degrees) overstates typical population decoding by ~77%.** The report should present the random-neuron result (4.95 degrees) as the primary finding, with the top-1000 result as an upper bound.

2. **The scaling analysis is biased at every point.** The shape of the scaling curve differs between pre-selected and random neurons, affecting claims about "diminishing returns" and "partial saturation."

3. **Comparison with Stringer et al.** is valid only if they also pre-selected neurons. Need to check their methods.

## Recommended Fix

1. **Add the random-neuron decoder** as the primary result. Present both:
   - "Random 1,000 neurons: 4.95 +/- 0.14 degrees MAE"
   - "Top 1,000 most reliable neurons: 2.80 degrees MAE"

2. **Reframe the scaling analysis** to show both curves (random and pre-selected) on the same plot. This actually tells a more interesting story: it shows how much neuron quality matters for decoding.

3. **Discuss the selection bias explicitly** in the results, not just the limitations section. The gap between random and top-1000 reveals something biologically interesting about how reliability relates to information content.

4. **Consider extending the random scaling** to larger neuron counts (2000, 5000, 10000) since you have 23,589 neurons available. This would show whether random selection eventually catches up.

## Severity: HIGH

The headline result is nearly doubled when the bias is removed. This doesn't invalidate the decoder approach (it still far exceeds chance), but the reported precision is meaningfully inflated. This needs to be corrected before submission.
