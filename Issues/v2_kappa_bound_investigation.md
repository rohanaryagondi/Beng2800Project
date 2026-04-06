# V2 Issue: Should the Kappa Upper Bound Be Increased?

## The Concern

The von Mises fit constrains kappa to [0, 20]. Figure 2 shows a heavy right tail in the kappa distribution — but since the bound is 20, this tail is artificially truncated. Are neurons being incorrectly compressed at the ceiling?

## Verdict: YES, the bound is too low. 8% of neurons are piled at the ceiling.

### Evidence of Ceiling Compression

**1,909 neurons (8.1%) have fitted kappa in [19.99, 20.0]** — they are jammed against the upper bound. This is not a gradual tail; it's a spike at exactly the constraint limit, which is a clear sign of an active constraint.

Distribution near the ceiling:
| Range | Neurons | % of population |
|-------|---------|-----------------|
| kappa >= 15 | 2,918 | 12.4% |
| kappa >= 18 | 2,227 | 9.4% |
| kappa >= 19 | 2,057 | 8.7% |
| kappa >= 19.9 | 1,928 | 8.2% |
| kappa >= 19.99 | 1,909 | 8.1% |

The 95th and 99th percentiles are both exactly 20.0 — the bound is clearly biting.

### Refit Results with Higher Bounds

We refitted all 1,928 ceiling neurons with bounds of 50 and 100:

| Bound | Median kappa | Mean kappa | Still at ceiling | R² median |
|-------|-------------|------------|------------------|-----------|
| 20 | 20.0 | 20.0 | 99.9% | 0.732 |
| 50 | 29.9 | 33.3 | 2.2% | 0.758 |
| 100 | 29.8 | 39.1 | 0.3% | 0.762 |

**Key findings:**

1. **Median kappa jumps from 20.0 to ~30** when unconstrained. Half the ceiling neurons "want" to be between 20 and 30.
2. **The tail is very long:** 22% exceed kappa = 50, 8.8% exceed kappa = 80.
3. **R² improves modestly** (+0.030 from bound=20 to bound=100), meaning the fits do get slightly better.
4. **Non-ceiling neurons are completely unaffected** (R² unchanged), confirming the refit is safe.
5. **At bound=50, only 2.2% remain at the ceiling.** At bound=100, only 0.3%.

### Where the unconstrained kappas land

From the bound=100 refit of ceiling neurons:
| Range | % of ceiling neurons |
|-------|---------------------|
| kappa 20–30 | ~50% |
| kappa 30–50 | ~28% |
| kappa 50–80 | ~14% |
| kappa 80–100 | ~8% |

### Physical Interpretation: Are These Neurons Real or Artifacts?

| Kappa | FWHM (deg) | Bins above half-max |
|-------|-----------|---------------------|
| 20 | 15.1 | ~3 bins |
| 30 | 12.3 | ~2.5 bins |
| 50 | 9.6 | ~2 bins |
| 100 | 6.7 | ~1.4 bins |

The ceiling neurons respond in only **~3 bins** on average (vs 6 bins for moderate-kappa neurons). With 5-degree bin width:

- **kappa = 30 (FWHM = 12 deg):** The tuning curve spans ~2.5 bins. This is narrow but still resolvable with 36 bins. Biologically plausible for sharply tuned V1 neurons.
- **kappa = 50 (FWHM = 10 deg):** Down to ~2 bins. At this point, we're fitting a 4-parameter model to essentially 2–3 data points above baseline. The kappa estimate becomes unreliable — small noise fluctuations in neighboring bins can push kappa from 40 to 80.
- **kappa > 50:** Approaching the bin resolution limit. These fits are poorly constrained — the optimizer may be chasing noise in 1–2 bins.

**Quality check:** Ceiling neurons have **lower** reliability (0.773 vs 0.785 population median) and **lower** R² (0.731 vs 0.750) than the full population. They are not the best-quality neurons — they are neurons with narrow but somewhat noisy tuning.

### What Happens to Downstream Analyses?

**Impact on the kappa distribution (Figure 2):**
The current histogram shows a sharp spike at kappa = 20. With bound = 50, this spike would spread into a smoother right tail from 20–50. The IQR would widen, and the "20% very sharply tuned" claim would be replaced by a more continuous distribution.

**Impact on reliability correlation (Issue 2):**
Minimal. The ceiling neurons all get similar reliability values whether kappa = 20 or 30, so spreading them out slightly changes the Spearman correlation but doesn't alter the fundamental finding.

**Impact on decoder (Issue 3):**
None. The decoder doesn't use kappa values.

## Recommended Fix

### Option A: Raise the bound to 50 (Recommended)

- Resolves the ceiling for 98% of affected neurons
- kappa up to ~50 is still physically interpretable (FWHM ≈ 10 deg ≈ 2 bins)
- Values beyond 50 are poorly constrained by the data, so a bound there acts as a reasonable regularizer
- Requires refitting all 23,589 neurons (takes ~15 min on M3 Pro)

### Option B: Raise the bound to 100

- Essentially unconstrained for this data
- Resolves 99.7% of ceiling cases
- But kappa > 50 values are unreliable given the 5-degree bin width
- Could lead to some extreme fitted values that are more noise than signal

### Option C: Keep bound at 20 but report honestly

- Note in methods that 8% of neurons hit the kappa = 20 ceiling
- State this is a lower bound on their true sharpness
- Acknowledge the Figure 2 tail is truncated
- Lowest effort, but the report contains a known artifact

### Implementation notes (for Option A)

1. Modify `scripts/04_fit_tuning.py` line 59: change `20.0` to `50.0` in the upper bounds
2. Rerun the fitting pipeline (Phase 2 onward)
3. Update the report text: change "bounded [0, 20]" to "bounded [0, 50]"
4. The kappa distribution statistics (median, IQR, percentages) will all change slightly

## Severity: MEDIUM

The current bound creates a visible artifact in Figure 2 (the pile-up at 20) and makes the "20% very sharply tuned (kappa >= 10)" statistic inaccurate — the true fraction with kappa >= 10 is similar, but their actual kappa values are wrong. The main conclusions don't change, but the quantitative description of tuning heterogeneity is distorted.
