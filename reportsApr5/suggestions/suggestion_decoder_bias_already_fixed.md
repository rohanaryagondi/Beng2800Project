# Suggestion: Decoder Top-1000 Bias — Already Fixed in Apr5

## Your Question

> "We used the 'top 1000' most reliable neurons among 22000, does this create bias in our result? Should we pool random 1000 instead."

## Answer: Yes, and we already fixed it.

Great instinct — this was exactly Issue 3 from the original critical review. The Apr5 report already addresses it. Here's what was done:

### What changed from the original report

**Original report:** Decoder used ONLY the top 1,000 most reliable neurons (reliability 0.97–1.0, top 4.2%). Headline result: 2.80 degrees MAE. All scaling points (10, 25, ..., 1000) drew from this elite pool.

**Apr5 report:** Random-neuron decoder is now the PRIMARY result. Both are presented:

| Neurons | Random Population MAE | Top-1000 Pool MAE |
|---------|----------------------|-------------------|
| 10      | 30.85 +/- 2.98      | 19.96 +/- 3.90    |
| 50      | 15.81 +/- 1.68      | 6.23 +/- 0.48     |
| 100     | 10.66 +/- 0.70      | 4.42 +/- 0.37     |
| 500     | 5.48 +/- 0.30       | 2.97 +/- 0.05     |
| 1000    | 4.94 +/- 0.14       | 2.80              |

The random-neuron decoder at 1,000 neurons (4.94 degrees) is 77% worse than the pre-selected one (2.80 degrees) — confirming significant selection bias.

### Where to see it in the Apr5 report

- **Section 2.4:** Methods now state neurons are drawn from "all 23,589 neurons" with top-1000 as a separate upper bound
- **Section 3.3:** Random-neuron MAE (4.94 degrees) is the headline number; top-1000 (2.80 degrees) is presented as an upper bound
- **Table 2:** Shows both random and top-1000 side by side
- **Figure 4:** Shows both scaling curves on the same plot
- **Section 4.1:** Discussion frames the 77% gap as revealing how neuron quality affects decoding

### Additional finding: OLS overfitting

The Apr5 report also discovered that when using random neurons, MAE gets WORSE at >1,000 neurons (5.53 degrees at 2,000, 7.71 degrees at 5,000) because OLS overfits when neuron count exceeds training samples per fold (~3,678). This motivates ridge regression as a future direction.

## No further action needed

This suggestion is already fully implemented in the Apr5 report. If the report you're looking at doesn't show this, you may be looking at the original version rather than the Apr5 version.
