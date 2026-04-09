# Report Changelog — Apr 8 Version (from Apr 5)

## Suggestions Applied

### Suggestion 1: Improved Introduction with Definitions
- Paragraph 1 rewritten: defines orientation selectivity, preferred orientation, and tuning sharpness in plain language before they appear in the research questions
- Removed disconnected prosthetics/BCI mention
- Added explicit reliability definition after the questions list

### Suggestion 2: Kappa Upper Bound Raised from 20 to 50
- `scripts/04_fit_tuning.py` bound changed from 20.0 to 50.0
- All 23,589 neurons refitted (~5 min)
- Old pile-up at kappa=20: **1,909 neurons (8.1%)** — now freed
- New pile-up at kappa=50: only **42 neurons (0.2%)**
- Median kappa unchanged: 3.83 (non-ceiling neurons unaffected)
- R² improved slightly for formerly capped neurons
- All downstream analyses (Phase 2, 3, 5) regenerated with new kappa values
- Report text updated: "bounded [0, 50]" with FWHM resolution justification

### Suggestion 3: Decoder Bias
- Already fixed in Apr 5 — no further action needed

### Suggestion 4: Denoising / Suite2p Limitations
- Section 2.1 expanded: mentions Suite2p neuropil subtraction and deconvolution, notes retained noise sources
- New limitation bullet: deconvolution noise may deflate kappa estimates; DeepInterpolation cited as potential improvement requiring raw data
- Lecoq et al. (2021) added to references

## Updated Numbers (Apr 8 vs Apr 5)

| Metric | Apr 5 | Apr 8 | Change |
|--------|-------|-------|--------|
| Kappa bound | [0, 20] | [0, 50] | Raised |
| Neurons at ceiling | 1,909 (8.1%) | 42 (0.2%) | Fixed |
| Kappa median | 3.83 | 3.83 | Unchanged |
| Kappa IQR | [1.31, 8.23] | [1.32, 8.23] | Negligible |
| Median R² | 0.750 | 0.753 | Slight improvement |
| Spearman r (kappa-reliability) | 0.239 | 0.237 | Negligible |
| OLS beta_kappa | 0.0040 | 0.0008 | Smaller (wider kappa range) |
| Pearson r (kappa-reliability) | 0.109 | 0.026 | Dropped (kappa spread out) |
| Decoder results | Unchanged | Unchanged | Not affected by kappa refit |
