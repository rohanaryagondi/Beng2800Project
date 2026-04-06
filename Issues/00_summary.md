# Issue Investigation Summary

**Date:** 2026-04-05
**Report:** BENG 2800 Final Report — Orientation Tuning in Mouse V1

---

## Issues Investigated

| # | Issue | Severity | Finding | Action Required |
|---|-------|----------|---------|-----------------|
| 1 | [100% fit convergence masks untuned neurons](issue1_untuned_neurons.md) | MEDIUM | 2,218 neurons (9.4%) have R² < 0.3 with kappa = 0; 3,726 (15.8%) have kappa < 0.1 | Better reporting; separate "untuned" from "broadly tuned" |
| 2 | [Reliability-sharpness correlation is mathematically trivial](issue2_reliability_sharpness_artifact.md) | HIGH | Simulation shows r = 0.90 expected purely from measurement math; the observed 0.239 is actually *lower* than expected | Reframe the finding entirely |
| 3 | [Decoder inflated by neuron selection bias](issue3_decoder_selection_bias.md) | HIGH | Random 1000 neurons: 4.95 deg MAE vs reported 2.80 deg (77% inflation) | Add random-neuron results as primary; top-1000 as upper bound |
| 4 | [Binned R² overstates fit quality](issue4_binned_r2_inflation.md) | LOW-MEDIUM | Single-trial R² ≈ 0.65 vs reported 0.75 on binned means | Clarify what R² measures in the text |
| 5 | [Minor issues: language, citations, diagnostics](issue5_minor_issues.md) | LOW | Single-mouse generalization, citation overreach, missing residual checks | Text fixes |

---

## Key Findings That Change the Story

### The reliability-sharpness correlation (Issue 2) is not what it seems

The report's second main finding — that sharper neurons are more reliable — is a **mathematical tautology**, not a biological discovery. Neurons with higher kappa have more structured tuning curves, which are inherently more correlated across split halves regardless of biological noise properties. Our simulation shows that even with noise completely independent of kappa, the expected correlation is r ≈ 0.90. The real data showing only r = 0.239 actually suggests something is *weakening* the expected relationship, which is the more interesting story.

### The decoder is good but not as good as reported (Issue 3)

The headline 2.80-degree result uses pre-selected elite neurons (top 4.2% by reliability). With randomly selected neurons, performance is 4.95 degrees — still impressive (far above 45-degree chance), but 77% worse than reported. The selection bias also distorts the scaling curve shape.

### The tuning characterization is mostly solid (Issues 1, 4)

The von Mises fitting and parameter distributions are reasonable. The main fix is acknowledging that ~10% of neurons are not meaningfully orientation-tuned despite "successful" fits, and that R² values refer to binned averages not single trials.

---

## Recommended Priority of Fixes

1. **Reframe reliability analysis** (Issue 2) — change from "biological finding" to "expected mathematical relationship, with interesting deviations"
2. **Add random-neuron decoder results** (Issue 3) — present both biased and unbiased numbers
3. **Improve reporting of fit quality** (Issues 1, 4) — separate untuned neurons, clarify R² context
4. **Language/citation fixes** (Issue 5) — soften generalization claims
