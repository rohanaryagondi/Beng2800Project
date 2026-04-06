"""
Download the Stringer 2019 orientation tuning dataset from OSF.

Source: https://osf.io/ny4ut/  (part of osf.io/hygbm — Neuromatch Academy data store)
Paper:  Stringer, Michaelos, Pachitariu (2021) "High-precision coding in visual cortex"
        DOI: 10.1016/j.cell.2021.03.042
"""
import os
import sys
import hashlib
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "raw", "stringer_orientations.npy")
URL = "https://osf.io/ny4ut/download"
EXPECTED_SIZE_MB = 893  # approximate

def download_with_progress(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1e6
            sys.stdout.write(f"\r  {pct:3d}%  {mb:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()

def verify(path):
    size_mb = os.path.getsize(path) / 1e6
    print(f"File size: {size_mb:.1f} MB")
    assert size_mb > 100, f"File suspiciously small: {size_mb:.1f} MB — download may have failed"
    print("Size check passed.")

if __name__ == "__main__":
    if os.path.exists(OUT):
        size_mb = os.path.getsize(OUT) / 1e6
        print(f"Already exists: {OUT} ({size_mb:.1f} MB)")
        if size_mb > 100:
            print("Skipping download.")
            sys.exit(0)
        else:
            print("File too small, re-downloading.")

    download_with_progress(URL, OUT)
    verify(OUT)
    print("Download complete.")
