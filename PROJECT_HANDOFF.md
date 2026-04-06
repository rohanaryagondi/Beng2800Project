# Stringer Orientation Tuning Project — Full Handoff Summary

## What this is

A BENG 2800 (Yale, Spring 2026) class project analyzing orientation tuning in mouse primary visual cortex (V1) using the public Stringer et al. (2021) two-photon calcium imaging dataset. The project is fully set up with analysis-ready processed files and a starter notebook.

## Project Questions

1. How heterogeneous are orientation tuning properties across neurons in mouse V1?
2. Are neurons with sharper tuning also more reliable across trials?
3. How well can a simple linear population decoder recover stimulus orientation from the neural population?

## Two Class Methods

1. **Nonlinear least squares** — Fit per-neuron orientation tuning curves (von Mises model, `scipy.optimize.curve_fit`)
2. **Linear models / least squares in matrix form** — Build a population decoder from neural responses to orientation (`numpy.linalg.lstsq`)

---

## Data Source

| Field | Value |
|-------|-------|
| Paper | Stringer, Michaelos, Pachitariu (2021) "High-precision coding in visual cortex" |
| DOI | [10.1016/j.cell.2021.03.042](https://doi.org/10.1016/j.cell.2021.03.042) |
| Download URL | https://osf.io/ny4ut/download |
| OSF project | https://osf.io/hygbm/ |
| GitHub reference | https://github.com/MouseLand/stringer-et-al-2019 |
| Recording ID | `stringer2019_v1_mouse1` |
| Experiment type | Static oriented gratings presented to a head-fixed mouse |

---

## Project Directory Structure

All paths are relative to the project root.

```
.
├── data/
│   ├── raw/
│   │   ├── stringer_orientations.npy          # 937 MB — raw downloaded file (do not modify)
│   │   └── raw_file_inventory.md              # documents every variable in the raw file
│   └── processed/
│       ├── orientation_decoder_ready.npz      # 232 MB — full trial×neuron matrix + angles
│       ├── orientation_decoder_ready_top1000.npz  # 9 MB — top 1000 reliable neurons
│       ├── orientation_decoder_targets.npz    # 66 KB — decoder target vectors (y_cos2, y_sin2)
│       ├── orientation_binned_tuning.npz      # 6 MB — binned tuning curves per neuron
│       ├── orientation_neuron_summary.parquet  # 2 MB — one row per neuron, all stats + fit results
│       ├── orientation_neuron_summary.csv      # 5 MB — same in CSV
│       └── orientation_project_metadata.json   # provenance and summary stats
├── scripts/                                    # all processing scripts (01–08), re-runnable
├── notebooks/
│   └── 01_orientation_project_start.ipynb      # starter notebook (loads data, example plots, decoder demo)
├── reports/
│   ├── orientation_data_summary.md             # QA report
│   └── figures/                                # 5 QC plots (histograms, tuning curves, pref dist)
└── PROJECT_HANDOFF.md                          # this file
```

**Total raw size:** 937 MB | **Total processed size:** 254 MB

---

## Raw Data — What's Inside `stringer_orientations.npy`

Load with: `dat = np.load(path, allow_pickle=True).item()`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `sresp` | (23589, 4598) | float64 | **Deconvolved calcium responses.** Rows = neurons, columns = trials. Each value is a single scalar per neuron per trial (already trial-summarized; no time bins). Range: 0–2865. |
| `istim` | (4598,) | float64 | **Stimulus direction** in radians, range 0.001–6.28 (0°–360°). Map to orientation via `istim % np.pi` (0–180°). 3313 unique values — effectively continuous. |
| `xyz` | (3, 23589) | float64 | Neuron spatial positions in microns (x, y, z). **Note: shape is 3×N, not N×3.** |
| `run` | (4598,) | float32 | Running speed per trial (arbitrary units). Mean 12.2, range 0–80.6. |
| `stat` | (23589,) | object | Suite2p per-neuron statistics (list of dicts). |
| `u_spont` | (23589, 128) | float64 | Top 128 PCs of spontaneous activity (neuron loadings). |
| `v_spont` | (128, 910) | float64 | Top 128 PC timecourses during spontaneous period. |
| `mean_spont` / `std_spont` | (23589,) | float64 | Per-neuron spontaneous activity stats. |
| `stimtimes` | (4598,) | int64 | Stimulus presentation frame indices. |
| `frametimes` | (25839,) | int64 | Imaging frame timestamps. |
| `camtimes` | (450538,) | int64 | Camera timestamps. |
| `info` | str | — | `'responses of 23589 neurons to 4598 static gratings'` |

---

## Processed Files — Detailed Schemas

### `orientation_decoder_ready.npz` (232 MB)

The main trial-level response matrix, ready for decoding.

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `X` | (4598, 23589) | float32 | Trial × neuron response matrix. Transposed from raw `sresp`. |
| `theta_deg` | (4598,) | float32 | Stimulus orientation in degrees (0–180°). Derived from `istim % pi`. |
| `theta_rad` | (4598,) | float32 | Same in radians (0–pi). |
| `neuron_ids` | (23589,) | int32 | Neuron indices 0–23588. |
| `recording_id` | (1,) | str | `['stringer2019_v1_mouse1']` |

### `orientation_decoder_ready_top1000.npz` (9 MB)

Subset of the 1000 most reliable neurons — recommended for laptop-friendly decoding.

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `X` | (4598, 1000) | float32 | Trial × neuron matrix (top 1000 by split-half reliability). |
| `y_cos2` | (4598,) | float32 | `cos(2 * theta_rad)` — decoder target. |
| `y_sin2` | (4598,) | float32 | `sin(2 * theta_rad)` — decoder target. |
| `theta_deg` | (4598,) | float32 | Orientation in degrees. |
| `theta_rad` | (4598,) | float32 | Orientation in radians. |
| `neuron_ids` | (1000,) | int32 | Which neurons (from the full 23589) are included. |
| `recording_id` | (1,) | str | `['stringer2019_v1_mouse1']` |

Top-1000 reliability range: **0.969 – 0.997**.

### `orientation_decoder_targets.npz` (66 KB)

Decoder target vectors only (does **not** contain X — load X from `orientation_decoder_ready.npz` or the top-1000 file).

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `y_cos2` | (4598,) | float32 | `cos(2 * theta_rad)` |
| `y_sin2` | (4598,) | float32 | `sin(2 * theta_rad)` |
| `theta_deg` | (4598,) | float32 | Orientation in degrees. |
| `theta_rad` | (4598,) | float32 | Orientation in radians. |

**Why cos/sin?** Orientation is periodic over 180°. Representing the angle as `(cos(2θ), sin(2θ))` converts it to a pair of continuous targets suitable for linear regression. To recover the predicted angle: `theta_pred = arctan2(y_sin2_pred, y_cos2_pred) / 2`.

### `orientation_binned_tuning.npz` (6 MB)

Pre-computed binned orientation tuning curves for every neuron.

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `tuning_mean` | (23589, 36) | float32 | Mean response per orientation bin. |
| `tuning_sem` | (23589, 36) | float32 | Standard error of the mean per bin. |
| `angle_bin_centers_deg` | (36,) | float32 | Bin centers: [2.5, 7.5, 12.5, ..., 177.5] degrees. |
| `neuron_ids` | (23589,) | int32 | Neuron indices. |

Binning: 36 bins × 5° each over 0–180°. ~128 trials per bin (4598 / 36).

### `orientation_neuron_summary.parquet` / `.csv` (2 MB / 5 MB)

One row per neuron (23,589 rows × 15 columns).

| Column | Dtype | Description |
|--------|-------|-------------|
| `neuron_id` | int32 | 0-indexed neuron identifier |
| `recording_id` | str | `stringer2019_v1_mouse1` |
| `n_trials` | int32 | 4598 for all neurons |
| `mean_response` | float32 | Mean deconvolved activity across all trials |
| `response_std` | float32 | Std across all trials |
| `pref_orientation_deg_empirical` | float32 | Bin center (°) with highest mean response |
| `osi_empirical` | float32 | Orientation Selectivity Index = (R_pref - R_orth) / (R_pref + R_orth) |
| `split_half_reliability` | float32 | Pearson r between odd/even trial tuning curves |
| `fit_success` | bool | Whether the von Mises fit converged |
| `fit_r2` | float64 | R² of the von Mises fit |
| `fit_baseline` | float64 | Fitted baseline parameter (b) |
| `fit_amplitude` | float64 | Fitted amplitude parameter (a) |
| `fit_pref_orientation_deg` | float64 | Fitted preferred orientation in degrees (0–180°) |
| `fit_kappa_or_width` | float64 | Fitted tuning sharpness (κ; larger = sharper; range 0–20) |
| `fit_method` | str | `von_mises_nlls` |

### `orientation_project_metadata.json`

Provenance, parameters, and summary statistics. Includes source DOI/URL, recording IDs, file sizes, orientation range, bin count, response summary definition, random seed (42).

---

## Tuning Curve Model

**Von Mises orientation tuning:**

```
r(θ) = b + a * exp(κ * cos(2 * (θ - θ₀)))
```

| Parameter | Meaning | Typical Range |
|-----------|---------|---------------|
| b | Baseline activity | ≥ 0 |
| a | Response amplitude above baseline | ≥ 0 |
| κ (kappa) | Tuning sharpness (larger = sharper) | 0–20 (capped) |
| θ₀ | Preferred orientation | 0–180° |

The `cos(2*(θ - θ₀))` form gives 180° periodicity, appropriate for orientation (as opposed to direction).

Fitted with `scipy.optimize.curve_fit` on the 36-bin tuning curves. Initial parameters derived from empirical tuning (b = 10th percentile, a = max - b, θ₀ = argmax, κ = 1.0).

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Total neurons | 23,589 |
| Total trials | 4,598 |
| Unique orientations | ~1,684 (continuous, binned into 36) |
| Orientation distribution | ~255 trials per 10° bin (approximately uniform) |
| Mean response | 9.06 (median 6.03) |
| Von Mises fit success rate | **100%** (23,589 / 23,589) |
| Median fit R² | **0.750** (mean 0.688) |
| Median fit κ | **3.83** (IQR: 1.31–8.23) |
| Median OSI (empirical) | **0.604** |
| Median split-half reliability | **0.785** (mean 0.730) |
| Fraction reliability > 0.5 | **85.4%** |
| Fraction reliability > 0.2 | **97.6%** |
| Fraction reliability < 0 | **0.5%** |

---

## How to Load Everything (Quick Reference)

```python
import numpy as np
import pandas as pd

PROC = "data/processed"

# Neuron summary (all stats + fit results)
df = pd.read_parquet(f"{PROC}/orientation_neuron_summary.parquet")

# Binned tuning curves
bd = np.load(f"{PROC}/orientation_binned_tuning.npz")
tuning_mean = bd["tuning_mean"]          # (23589, 36)
centers_deg = bd["angle_bin_centers_deg"] # (36,)

# Full decoder matrix
d = np.load(f"{PROC}/orientation_decoder_ready.npz")
X = d["X"]                # (4598, 23589) trials × neurons
theta_deg = d["theta_deg"] # (4598,) orientation in degrees

# Top-1000 reliable neurons (recommended for laptop decoding)
d1k = np.load(f"{PROC}/orientation_decoder_ready_top1000.npz")
X_top = d1k["X"]           # (4598, 1000)

# Decoder targets
dt = np.load(f"{PROC}/orientation_decoder_targets.npz")
y_cos2 = dt["y_cos2"]      # (4598,)  cos(2*theta)
y_sin2 = dt["y_sin2"]      # (4598,)  sin(2*theta)

# Von Mises tuning function
def von_mises_tuning(theta, b, a, kappa, theta0):
    """theta in radians, all params from neuron summary fit columns."""
    return b + a * np.exp(kappa * np.cos(2.0 * (theta - theta0)))
```

---

## Processing Pipeline (Re-runnable)

All scripts live in `scripts/` and are numbered in execution order:

| Script | Purpose | Runtime |
|--------|---------|---------|
| `01_download.py` | Downloads raw file from OSF (skips if exists) | ~5 min |
| `02_inspect.py` | Inspects raw file, writes `raw_file_inventory.md` | ~5 sec |
| `03_process.py` | Creates decoder matrix, neuron summary, binned tuning, metadata | ~30 sec |
| `04_fit_tuning.py` | Fits von Mises curves (all 23,589 neurons), updates summary | ~4 min |
| `05_decoder_targets.py` | Creates `y_cos2`/`y_sin2` targets + top-1000 subset | ~10 sec |
| `06_qc_plots.py` | Generates 5 QC figures | ~5 sec |
| `07_qa_report.py` | Generates QA markdown report | ~5 sec |
| `08_create_notebook.py` | Creates starter Jupyter notebook | instant |

To re-run the full pipeline:
```bash
cd "path/to/Project/Claude"
python3 scripts/01_download.py
python3 scripts/02_inspect.py
python3 scripts/03_process.py
python3 scripts/04_fit_tuning.py
python3 scripts/05_decoder_targets.py
python3 scripts/06_qc_plots.py
python3 scripts/07_qa_report.py
python3 scripts/08_create_notebook.py
```

---

## Design Decisions & Known Deviations

1. **Direction → orientation mapping:** Raw `istim` is direction (0–2π). Mapped to orientation via `istim % pi` (0–π = 0–180°). This is correct because the experiment uses static gratings where 0° and 180° are identical.

2. **No X in decoder_targets.npz:** The spec requested X in `orientation_decoder_targets.npz`, but this would duplicate the 232 MB matrix already in `orientation_decoder_ready.npz`. X was excluded to keep total processed size under 400 MB (actual: 254 MB). Load X from `orientation_decoder_ready.npz` or the top-1000 file instead.

3. **No DSI column:** The spec suggested `dsi_empirical` if direction information is meaningful. Since we map to orientation (collapsing direction), DSI is not meaningful and was omitted.

4. **κ capped at 20:** The von Mises `kappa` parameter is bounded at [0, 20] during fitting. κ > 20 produces a tuning curve so sharp it is essentially a delta function; capping prevents numerical instability.

5. **Split-half reliability method:** Odd/even trial split by trial index (not random), then Pearson correlation of per-bin mean responses across the two halves.

6. **Response summary:** Used `dat['sresp']` directly — the raw data already provides one scalar per neuron per trial (deconvolved calcium activity). No additional time-window extraction was needed.

7. **Random seed:** All randomized operations use `seed=42`.

---

## What's Ready for Analysis

| Analysis Task | Status | Key Files |
|--------------|--------|-----------|
| Nonlinear tuning curve fitting | **Done** — all 23,589 fits complete | `orientation_binned_tuning.npz`, `orientation_neuron_summary.parquet` |
| Re-fit or explore tuning models | Ready — binned data available | `orientation_binned_tuning.npz` |
| Reliability analysis | **Done** — `split_half_reliability` computed | `orientation_neuron_summary.parquet` |
| Tuning sharpness vs reliability | Ready — both columns in summary | `orientation_neuron_summary.parquet` |
| Linear population decoder | Ready — X and y targets prepared | `orientation_decoder_ready_top1000.npz`, `orientation_decoder_targets.npz` |
| Full-population decoder | Ready — requires ~868 MB RAM | `orientation_decoder_ready.npz`, `orientation_decoder_targets.npz` |
| Spatial analysis of tuning | Ready — `xyz` in raw file | `stringer_orientations.npy` key `xyz` (shape 3×23589) |
| Running speed effects | Ready — `run` in raw file | `stringer_orientations.npy` key `run` (shape 4598) |

---

## Suggested Next Steps

1. **Explore the starter notebook** — `notebooks/01_orientation_project_start.ipynb` loads all data, shows an example tuning curve with von Mises fit, and demonstrates a working least-squares decoder.

2. **Investigate tuning heterogeneity** — Use `orientation_neuron_summary.parquet` to characterize the distribution of OSI, preferred orientation, and κ across the population.

3. **Sharpness–reliability relationship** — Scatter `fit_kappa_or_width` vs `split_half_reliability`. Compute Spearman correlation. This directly addresses project question #2.

4. **Build and evaluate the linear decoder** — Use `orientation_decoder_ready_top1000.npz` for train/test split. Fit with `np.linalg.lstsq`. Evaluate with circular MAE. Explore how performance scales with neuron count (subsets of 50, 100, 200, 500, 1000).

5. **Generate publication-quality figures** — Tuning curve examples (good/bad/mediocre), population tuning heatmap sorted by preferred orientation, decoder confusion matrix, reliability vs sharpness scatter.

---

## Dependencies

- Python 3.13
- numpy, scipy, pandas, pyarrow, matplotlib, jupyter

All installed and verified on the current machine.
