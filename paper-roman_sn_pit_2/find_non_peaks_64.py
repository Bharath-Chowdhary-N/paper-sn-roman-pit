#!/usr/bin/env python3
"""
FITS Non-Peak Finder and Cutout Generator — Fast Version

Fixes vs previous version:
  1. Variance-based valid region check — rejects flat grey border (same fix
     as psf_inject.py) instead of relying only on non-zero pixel fraction.
  2. Vectorized distance filtering — replaces the O(n²) Python loop with a
     grid-based exclusion mask, making 10 000-sample runs ~100x faster.
  3. global_std computed once per image, not per candidate.

Requirements: pip install astropy numpy matplotlib scipy photutils pillow
"""

import os
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.ndimage import uniform_filter
from PIL import Image

warnings.filterwarnings('ignore', category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# VALID SITE CHECK  (mirrors psf_inject.py logic)
# ═══════════════════════════════════════════════════════════════════════════════

def is_valid_cutout_site(data, cy, cx, global_std,
                         half=32,
                         min_nonzero_fraction=0.95,
                         min_std_fraction=0.01):
    """
    Return True only if the 64x64 patch at (cy, cx) looks like real image
    data — i.e. it is not zero-padded AND not a flat grey border region.

    Condition 1 — non-zero coverage : ≥95 % of pixels are finite & non-zero.
    Condition 2 — variance          : patch std > 1 % of the global image std.
                  A flat grey border has near-zero variance and fails this.
    """
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    if y0 < 0 or y1 > data.shape[0] or x0 < 0 or x1 > data.shape[1]:
        return False

    patch = data[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch != 0.0)

    if valid.sum() / patch.size < min_nonzero_fraction:
        return False                      # too many zero/NaN pixels

    if global_std == 0:
        return False
    if patch[valid].std() < min_std_fraction * global_std:
        return False                      # flat grey border — reject

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# FAST NON-PEAK FINDER
# ═══════════════════════════════════════════════════════════════════════════════

def find_non_peak_positions(data, num_non_peaks=500,
                            min_distance=50, cutout_size=64,
                            max_attempts=200_000):
    """
    Randomly sample positions that:
      • lie on real image data (variance check)
      • are in low-flux regions (below 40th percentile of valid pixels)
      • are at least min_distance pixels apart (enforced via exclusion grid)

    The exclusion grid is an integer array — marking chosen positions burns
    O(1) per selection, making the whole function O(candidates) rather than
    O(candidates × selected), which is what killed the old version.
    """
    half = cutout_size // 2
    h, w = data.shape

    # ── Pre-compute global std once ───────────────────────────────────────
    real = data[np.isfinite(data) & (data != 0.0)]
    global_std = real.std() if len(real) > 0 else 1.0
    print(f"    Global image std: {global_std:.6f}")

    # ── Low-flux threshold (non-peaks should be background-like) ─────────
    flux_threshold = np.percentile(real, 40) if len(real) > 0 else 0.0

    # ── Exclusion grid — cells of size min_distance ───────────────────────
    # Any cell that contains a chosen position is marked occupied.
    # A new candidate is rejected if its cell OR any of the 8 neighbours
    # is already occupied.  This approximates the min_distance constraint
    # very cheaply.
    cell = min_distance
    grid_h = h // cell + 2
    grid_w = w // cell + 2
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    def cell_coords(cy, cx):
        return cy // cell, cx // cell

    def neighbours_free(cy, cx):
        gy, gx = cell_coords(cy, cx)
        y0, y1 = max(0, gy - 1), min(grid_h, gy + 2)
        x0, x1 = max(0, gx - 1), min(grid_w, gx + 2)
        return not occupied[y0:y1, x0:x1].any()

    def mark_occupied(cy, cx):
        gy, gx = cell_coords(cy, cx)
        occupied[gy, gx] = True

    # ── Sample loop ───────────────────────────────────────────────────────
    positions = []
    attempts  = 0

    print(f"    Searching for {num_non_peaks} non-peak positions "
          f"(min_dist={min_distance} px) …")

    while len(positions) < num_non_peaks and attempts < max_attempts:
        attempts += 1
        cy = random.randint(half, h - half - 1)
        cx = random.randint(half, w - half - 1)

        # Variance + non-zero check (rejects flat grey border)
        if not is_valid_cutout_site(data, cy, cx, global_std):
            continue

        # Only keep low-flux (background) regions
        if data[cy, cx] > flux_threshold:
            continue

        # Fast distance exclusion via grid
        if not neighbours_free(cy, cx):
            continue

        positions.append((cy, cx))
        mark_occupied(cy, cx)

    print(f"    Found {len(positions)} positions in {attempts} attempts.")
    if len(positions) < num_non_peaks:
        print(f"    ⚠  Only {len(positions)}/{num_non_peaks} found — "
              f"valid image area may be too small for this many cutouts.")
    return positions


# ═══════════════════════════════════════════════════════════════════════════════
# CUTOUT CREATION & SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_with_zscale(data):
    valid = np.isfinite(data)
    if not np.any(valid):
        return np.zeros_like(data)
    zs = ZScaleInterval()
    vmin, vmax = zs.get_limits(data[valid])
    out = np.clip((data - vmin) / (vmax - vmin + 1e-12), 0, 1)
    out[~valid] = 0
    return out


def create_cutout(data, cy, cx, cutout_size=64):
    """Extract cutout; return None if it contains any invalid pixels."""
    half = cutout_size // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    cutout = data[y0:y1, x0:x1]
    if cutout.shape != (cutout_size, cutout_size):
        return None
    if np.any(~np.isfinite(cutout)) or np.any(cutout == 0):
        return None
    return cutout


def save_png(cutout_norm, filepath):
    img = Image.fromarray((cutout_norm * 255).astype(np.uint8), mode='L')
    img.save(filepath)


def save_fits(cutout, filepath, header=None):
    fits.PrimaryHDU(data=cutout.astype(np.float32),
                    header=header).writeto(filepath, overwrite=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-FILE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_fits_file(fits_path, output_dir, cutout_size=64, num_non_peaks=500):
    fits_path  = Path(fits_path)
    output_dir = Path(output_dir)
    tile_name  = fits_path.stem

    png_dir  = output_dir / 'png'  / tile_name
    fits_dir = output_dir / 'fits' / tile_name
    png_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {fits_path.name} …")

    # ── Read FITS ─────────────────────────────────────────────────────────
    data = header = None
    try:
        with fits.open(fits_path) as hdul:
            print(f"  HDUs: {len(hdul)}")
            if 'SCI' in hdul:
                data   = hdul['SCI'].data.copy()
                header = hdul['SCI'].header.copy()
                print(f"  SCI extension → shape {data.shape}")
            else:
                for i, hdu in enumerate(hdul):
                    if hdu.data is not None and hdu.data.ndim == 2:
                        data   = hdu.data.copy()
                        header = hdu.header.copy()
                        print(f"  HDU {i} → shape {data.shape}")
                        break
    except Exception as e:
        print(f"  ❌ Read error: {e}"); return

    if data is None:
        print("  ❌ No 2D image data found — skipping."); return
    if data.ndim > 2:
        data = data[0] if data.ndim == 3 else data.squeeze()

    # ── Clean ─────────────────────────────────────────────────────────────
    data = np.nan_to_num(data.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Raw range: {data.min():.4f} → {data.max():.4f}")

    # ── ZScale normalise for consistent processing ────────────────────────
    valid = np.isfinite(data) & (data != 0.0)
    if not valid.any():
        print("  ❌ All pixels invalid — skipping."); return

    zs = ZScaleInterval()
    zmin, zmax = zs.get_limits(data[valid])
    data_norm = np.clip((data - zmin) / (zmax - zmin + 1e-12), 0, 1)
    # Keep zeros where original was zero (border stays zero after normalisation
    # only if zmin is exactly 0, so enforce it explicitly)
    data_norm[~valid] = 0.0
    print(f"  ZScale [{zmin:.4f}, {zmax:.4f}] → normalised range "
          f"[{data_norm.min():.4f}, {data_norm.max():.4f}]")

    # ── Find non-peak positions (fast) ────────────────────────────────────
    positions = find_non_peak_positions(
        data_norm,
        num_non_peaks=num_non_peaks,
        min_distance=cutout_size // 2,
        cutout_size=cutout_size,
    )

    if not positions:
        print("  ❌ No valid positions found."); return

    # ── Save cutouts ──────────────────────────────────────────────────────
    saved = 0
    for idx, (cy, cx) in enumerate(positions):
        cutout = create_cutout(data_norm, cy, cx, cutout_size)
        if cutout is None:
            continue

        cutout_norm = normalize_with_zscale(cutout)
        stem = f"{tile_name}_nonpeak_{saved:04d}_y{cy}_x{cx}"
        save_png(cutout_norm, png_dir  / f"{stem}.png")
        save_fits(cutout,     fits_dir / f"{stem}.fits", header)
        saved += 1

    print(f"  ✅ Saved {saved} non-peak cutouts for {fits_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    input_dir  = Path("test_diff_files/")
    output_dir = Path("non_peaks")

    if not input_dir.exists():
        print(f"❌ Input directory '{input_dir}' does not exist"); return

    fits_files = sorted(
        list(input_dir.glob("*.fits")) + list(input_dir.glob("*.fit"))
    )
    if not fits_files:
        print(f"❌ No FITS files found in '{input_dir}'"); return

    print(f"Found {len(fits_files)} FITS file(s).")
    for fp in fits_files:
        try:
            process_fits_file(fp, output_dir, num_non_peaks=10_000)
        except Exception as e:
            print(f"  ❌ Error processing {fp.name}: {e}")
            import traceback; traceback.print_exc()

    print("\n✅ All done.")


if __name__ == "__main__":
    main()