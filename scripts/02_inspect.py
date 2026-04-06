"""
Inspect the raw downloaded file and write data/raw/raw_file_inventory.md.
"""
import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw", "stringer_orientations.npy")
OUT = os.path.join(BASE, "data", "raw", "raw_file_inventory.md")


def describe_value(key, val):
    if isinstance(val, np.ndarray) and val.dtype.kind in ('f', 'i', 'u', 'c'):
        try:
            vmin = float(np.nanmin(val.astype(float)))
            vmax = float(np.nanmax(val.astype(float)))
            n_nan = int(np.isnan(val.astype(float)).sum()) if np.issubdtype(val.dtype, np.floating) else 0
            return (f"  - **{key}**: ndarray shape={val.shape}, dtype={val.dtype}, "
                    f"min={vmin:.4g}, max={vmax:.4g}, n_nan={n_nan}")
        except Exception:
            return f"  - **{key}**: ndarray shape={val.shape}, dtype={val.dtype} (could not compute stats)"
    elif isinstance(val, np.ndarray):
        return f"  - **{key}**: ndarray shape={val.shape}, dtype={val.dtype}"
    elif isinstance(val, (list, tuple)):
        elem_info = f" (first element type: {type(val[0]).__name__})" if len(val) > 0 else ""
        return f"  - **{key}**: {type(val).__name__} len={len(val)}{elem_info}"
    else:
        return f"  - **{key}**: {type(val).__name__} = {repr(val)[:200]}"


def main():
    if not os.path.exists(RAW):
        print(f"ERROR: {RAW} not found. Run 01_download.py first.", file=sys.stderr)
        sys.exit(1)

    size_bytes = os.path.getsize(RAW)
    size_mb = size_bytes / 1e6

    print(f"Loading {RAW} ({size_mb:.1f} MB)...")
    dat = np.load(RAW, allow_pickle=True).item()

    print("Keys:", list(dat.keys()))
    lines = []
    lines.append("# Raw File Inventory\n")
    lines.append(f"**File:** `data/raw/stringer_orientations.npy`  ")
    lines.append(f"**Size:** {size_mb:.1f} MB ({size_bytes:,} bytes)  ")
    lines.append(f"**Source URL:** https://osf.io/ny4ut/download  ")
    lines.append(f"**OSF project:** https://osf.io/hygbm/ (Neuromatch Academy)  ")
    lines.append(f"**Paper:** Stringer, Michaelos, Pachitariu (2021) Cell. DOI: 10.1016/j.cell.2021.03.042  ")
    lines.append(f"**Load command:** `dat = np.load(path, allow_pickle=True).item()`\n")

    lines.append("## Variable Summary\n")
    for key, val in dat.items():
        lines.append(describe_value(key, val))

    lines.append("")
    lines.append("## Axis Meanings\n")

    if "sresp" in dat:
        sr = dat["sresp"]
        lines.append(f"**sresp** shape {sr.shape}:")
        lines.append(f"  - Axis 0: neurons (n={sr.shape[0]})")
        lines.append(f"  - Axis 1: trials (n={sr.shape[1]})")
        lines.append(f"  - Values: deconvolved calcium activity (approximately spike rates)")
        lines.append(f"  - Memory: {sr.nbytes/1e6:.1f} MB\n")

    if "istim" in dat:
        ist = dat["istim"]
        lines.append(f"**istim** shape {ist.shape}:")
        lines.append(f"  - Axis 0: trials (n={ist.shape[0]})")
        lines.append(f"  - Values: stimulus angle in radians")
        lines.append(f"  - Range: {float(ist.min()):.4f} to {float(ist.max()):.4f} rad "
                     f"({float(ist.min())*180/np.pi:.1f}° to {float(ist.max())*180/np.pi:.1f}°)")
        unique_angles = np.unique(np.round(ist, 3))
        lines.append(f"  - Unique angles (rounded): {len(unique_angles)} distinct values")
        lines.append(f"  - First 10 unique values (deg): {[round(float(a)*180/np.pi,1) for a in unique_angles[:10]]}\n")

    if "xyz" in dat:
        xyz = dat["xyz"]
        lines.append(f"**xyz** shape {xyz.shape}:")
        lines.append(f"  - Axis 0: neurons (n={xyz.shape[0]})")
        lines.append(f"  - Axis 1: x, y, z spatial coordinates in microns\n")

    if "run" in dat:
        run = dat["run"]
        lines.append(f"**run** shape {run.shape}:")
        lines.append(f"  - Running speed of the mouse per trial (arbitrary units)")
        lines.append(f"  - Mean: {float(run.mean()):.3f}, Std: {float(run.std()):.3f}\n")

    lines.append("## Notes\n")
    lines.append("- `sresp` is in neurons × trials format — transpose to trials × neurons for decoder (X matrix).")
    lines.append("- `istim` values 0–2π suggest direction coding. Map to orientation via `istim % np.pi` for 0–π range.")
    lines.append("- No explicit time axis — each column in `sresp` is already a single-trial summary (deconvolved).")
    lines.append("- All neurons are from a single recording session in mouse V1.")

    text = "\n".join(lines)
    with open(OUT, "w") as f:
        f.write(text)

    print(f"\nInventory written to: {OUT}")
    print(f"\nKey shapes:")
    for key, val in dat.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape} {val.dtype}")


if __name__ == "__main__":
    main()
