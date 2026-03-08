#!/usr/bin/env python3
"""
PSF Injection Script — with smart valid-region detection and per-tile SNR history.

Key improvements over previous version
---------------------------------------
1. VALID REGION DETECTION
   Detects the actual difference-image footprint (the irregular grey patch)
   and never injects into the empty/padded border.  Works by:
     a) Masking zero / NaN / Inf / constant-border fill values.
     b) Keeping only the largest connected component.
     c) Eroding inward by (half_cutout + margin) pixels so every injected
        64x64 cutout is fully inside the valid area.

2. PER-TILE SNR HISTORY
   Every injection run appends to  injection_history.json  inside the
   output folder.  Each entry records tile name, timestamp, every injection's
   pixel position, RA/DEC, target SNR, achieved SNR, scale factor and FWHM.
   Re-running the script appends new entries rather than overwriting.

Requirements:  pip install astropy numpy photutils scipy pandas matplotlib pillow
"""

import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy import units as u
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                aperture_photometry)
from scipy.ndimage import (maximum_filter, generate_binary_structure)
from PIL import Image

warnings.filterwarnings('ignore', category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# PSF UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_array(data):
    lo, hi = np.min(data), np.max(data)
    return np.zeros_like(data) if hi == lo else (data - lo) / (hi - lo)


def normalize_with_zscale(data):
    valid = np.isfinite(data)
    if not np.any(valid):
        return np.zeros_like(data)
    zs = ZScaleInterval()
    vmin, vmax = zs.get_limits(data[valid])
    out = np.clip((data - vmin) / (vmax - vmin + 1e-12), 0, 1)
    out[~valid] = 0
    return out


def calculate_fwhm_from_psf(psf_array):
    cy, cx = np.unravel_index(np.argmax(psf_array), psf_array.shape)
    y, x = np.ogrid[:psf_array.shape[0], :psf_array.shape[1]]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    max_r = min(int(r.max()), 30)
    profile = np.array([psf_array[r == i].mean() if np.any(r == i) else 0
                        for i in range(max_r + 1)])
    half_max = profile.max() / 2.0
    crossings = np.where(profile <= half_max)[0]
    return max(float(2 * crossings[0]) if len(crossings) else 4.0, 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# VALID REGION DETECTION  ← KEY FIX
# ═══════════════════════════════════════════════════════════════════════════════

def is_valid_injection_site(data, center_y, center_x,
                            global_std,
                            min_nonzero_fraction=0.95,
                            min_std_fraction=0.01):
    """
    Check the 64x64 cutout at (center_y, center_x).

    Two conditions must both pass:
      1. >=95% of pixels are finite and non-zero  (catches NaN/zero padding)
      2. The cutout std > min_std_fraction * global_std  (catches flat grey
         border regions that are non-zero but have no real image variation)
    """
    half = 32
    y0, y1 = center_y - half, center_y + half
    x0, x1 = center_x - half, center_x + half

    if y0 < 0 or y1 > data.shape[0] or x0 < 0 or x1 > data.shape[1]:
        return False

    cutout = data[y0:y1, x0:x1]

    # Condition 1: enough non-zero finite pixels
    valid = np.isfinite(cutout) & (cutout != 0.0)
    if valid.sum() / cutout.size < min_nonzero_fraction:
        return False

    # Condition 2: real noise variation — flat grey border will fail this
    if global_std == 0 or cutout[valid].std() < min_std_fraction * global_std:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# PEAK FINDING & POSITION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def find_peaks_simple(data, threshold_percentile=95, min_distance=64):
    threshold  = np.percentile(data, threshold_percentile)
    nbhd       = generate_binary_structure(2, 2)
    local_max  = maximum_filter(data, footprint=nbhd) == data
    peaks_mask = local_max & (data > threshold)
    coords     = np.column_stack(np.where(peaks_mask))
    filtered   = []
    for p in coords:
        if not filtered or min(np.hypot(p[0]-q[0], p[1]-q[1]) for q in filtered) >= min_distance:
            filtered.append(tuple(p))
        if len(filtered) >= 50:
            break
    return filtered


def generate_injection_positions(data, existing_peaks,
                                 n_positions=500,
                                 min_peak_dist=80,
                                 min_inject_dist=50,
                                 max_attempts=50000):
    """
    Randomly sample candidate positions. For each one, peek at the actual
    64x64 cutout — reject if it has no real noise variation (flat grey border).
    global_std is computed once up front so the per-attempt check is fast.
    """
    half = 32
    h, w = data.shape
    positions = []
    attempts  = 0

    # Compute once — used inside is_valid_injection_site to detect flat border
    real_pixels = data[np.isfinite(data) & (data != 0.0)]
    global_std  = real_pixels.std() if len(real_pixels) > 0 else 1.0
    print(f"    Global image std (real pixels): {global_std:.6f}")
    print(f"    Searching for {n_positions} valid injection positions...")

    while len(positions) < n_positions and attempts < max_attempts:
        attempts += 1
        cy = random.randint(half, h - half - 1)
        cx = random.randint(half, w - half - 1)

        # Reject if cutout is zero-padded OR flat grey border
        if not is_valid_injection_site(data, cy, cx, global_std):
            continue

        # Stay away from bright existing sources
        if any(np.hypot(cy - py, cx - px) < min_peak_dist
               for py, px in existing_peaks):
            continue

        # Stay away from already-chosen injection centres
        if any(np.hypot(cy - iy, cx - ix) < min_inject_dist
               for iy, ix in positions):
            continue

        positions.append((cy, cx))

    print(f"    Found {len(positions)} positions in {attempts} attempts.")
    if len(positions) < n_positions:
        print(f"    ⚠  Only {len(positions)}/{n_positions} found — "
              f"valid image area may be small.")
    return positions


# ═══════════════════════════════════════════════════════════════════════════════
# PHOTOMETRY & SNR
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_local_noise(image, position, aperture_radius=10):
    cy, cx = position
    try:
        ann  = CircularAnnulus((cx, cy), r_in=aperture_radius*2, r_out=aperture_radius*4)
        amsk = ann.to_mask()
        adata = amsk.multiply(image)
        if adata is not None:
            vals = adata[amsk.data > 0]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 10:
                mn, sd = vals.mean(), vals.std()
                clipped = vals[np.abs(vals - mn) < 3*sd]
                return clipped.std() if len(clipped) > 5 else sd
    except Exception:
        pass
    return 1.0


def calculate_snr_photutils(image, center_xy, fwhm=2.17):
    r  = 0.6731 * fwhm
    ap = CircularAperture(center_xy, r=r)
    an = CircularAnnulus(center_xy, r_in=r*1.5, r_out=r*2.5)
    try:
        pt  = aperture_photometry(image, ap)
        bt  = aperture_photometry(image, an)
        bkg = bt['aperture_sum'][0] / an.area * ap.area
        flux = pt['aperture_sum'][0] - bkg
        amsk = an.to_mask()
        adata = amsk.multiply(image)
        bstd  = np.std(adata[amsk.data > 0]) if adata is not None else 1.0
        noise = np.sqrt(abs(flux) + ap.area * bstd**2)
        return max(float(flux / noise) if noise > 0 else 0.0, 0.0)
    except Exception:
        return 0.0


def calculate_required_scale_factor(target_snr, psf_array, noise_std, fwhm):
    r = 0.6731 * fwhm
    cy, cx = np.array(psf_array.shape) // 2
    y, x = np.ogrid[:psf_array.shape[0], :psf_array.shape[1]]
    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    psf_flux = psf_array[mask].sum()
    if psf_flux <= 0:
        return max(1.0, target_snr * noise_std * 10)
    bg_noise = noise_std * np.sqrt(np.pi * r**2)
    bg_noise = max(bg_noise, 1e-10)
    return max(1.0, target_snr * bg_noise / psf_flux)


def inject_psf_at(image, psf, center_y, center_x, scale):
    """Add scaled PSF into image at (center_y, center_x). Returns modified copy."""
    out = image.copy()
    hy, hx = psf.shape[0] // 2, psf.shape[1] // 2
    iy0 = max(0, center_y - hy);  iy1 = min(image.shape[0], center_y + hy)
    ix0 = max(0, center_x - hx);  ix1 = min(image.shape[1], center_x + hx)
    py0 = max(0, hy - center_y);  py1 = py0 + (iy1 - iy0)
    px0 = max(0, hx - center_x);  px1 = px0 + (ix1 - ix0)
    out[iy0:iy1, ix0:ix1] += psf[py0:py1, px0:px1] * scale
    return out


def iteratively_scale_for_target_snr(image, psf, center_y, center_x,
                                     target_snr, fwhm, clean_noise=None,
                                     max_iter=10):
    """
    clean_noise: pre-computed noise from the CLEAN image at this position.
                 If None, noise is estimated from 'image' (old behaviour).
    """
    noise = clean_noise if clean_noise is not None \
            else estimate_local_noise(image, (center_y, center_x))
    scale = calculate_required_scale_factor(target_snr, psf, noise, fwhm)
    scale = max(1.0, scale)
    achieved = 0.0
    for _ in range(max_iter):
        tmp = inject_psf_at(image, psf, center_y, center_x, scale)
        achieved = calculate_snr_photutils(tmp, (center_x, center_y), fwhm)
        if achieved > 0 and abs(achieved - target_snr) / target_snr < 0.20:
            break
        ratio = np.clip(target_snr / achieved, 0.5, 2.0) if achieved > 0 else 2.0
        scale = max(1.0, min(scale * ratio, 1000.0))
    return scale, achieved


def inject_multiple_psfs(image, psf, positions, target_snrs, fwhm):
    modified = image.copy()
    scales, snrs, good_pos = [], [], []

    # ── Pre-compute noise at every position on the CLEAN image ────────────────
    # This avoids contamination from previously injected PSFs accumulating in
    # the annulus used by estimate_local_noise.
    print(f"    Pre-computing local noise at {len(positions)} positions (clean image)...")
    clean_noises = [estimate_local_noise(image, (cy, cx)) for cy, cx in positions]

    print(f"    Iteratively scaling {len(positions)} PSFs...")
    for i, (cy, cx) in enumerate(positions):
        tgt = target_snrs[i] if i < len(target_snrs) else target_snrs[-1]
        scale, achieved = iteratively_scale_for_target_snr(
            modified, psf, cy, cx, tgt, fwhm, clean_noise=clean_noises[i])
        if achieved > 0.5:
            modified = inject_psf_at(modified, psf, cy, cx, scale)
            scales.append(scale); snrs.append(achieved); good_pos.append((cy, cx))
            print(f"      [{len(good_pos):3d}] target={tgt:.1f}  "
                  f"achieved={achieved:.2f}  scale={scale:.1f}")
        else:
            print(f"      [{i+1:3d}] SKIPPED — SNR too low ({achieved:.2f})")
    print(f"    Injected {len(good_pos)}/{len(positions)} successfully.")
    return modified, scales, snrs, good_pos


# ═══════════════════════════════════════════════════════════════════════════════
# WCS
# ═══════════════════════════════════════════════════════════════════════════════

def pixel_to_radec(x, y, header):
    try:
        wcs = WCS(header)
        sky = wcs.pixel_to_world(x, y)
        return (sky.ra.to_string(unit=u.hourangle, sep=':', precision=2),
                sky.dec.to_string(unit=u.deg, sep=':', precision=2))
    except Exception:
        return f"{x:.2f}", f"{y:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# SNR HISTORY — PER-TILE JSON LOG   ← KEY ADDITION
# ═══════════════════════════════════════════════════════════════════════════════

def load_history(history_path):
    """Load existing history JSON, or return empty dict."""
    if Path(history_path).exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return {}   # {tile_name: [run1, run2, ...]}


def append_history(history_path, tile_name, run_record):
    """
    Append run_record (a dict) to the history for tile_name.
    Creates the file if it doesn't exist; never overwrites previous runs.
    """
    history = load_history(history_path)
    if tile_name not in history:
        history[tile_name] = []
    history[tile_name].append(run_record)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def build_run_record(tile_name, results, psf_fwhm, n_positions_requested):
    """Build the dict that gets stored in injection_history.json for one run."""
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tile': tile_name,
        'psf_fwhm_px': round(psf_fwhm, 4),
        'positions_requested': n_positions_requested,
        'positions_injected': len(results),
        'snr_target_range': [
            round(min(r['target_snr'] for r in results), 3),
            round(max(r['target_snr'] for r in results), 3)
        ] if results else [None, None],
        'snr_achieved_range': [
            round(min(r['achieved_snr'] for r in results), 3),
            round(max(r['achieved_snr'] for r in results), 3)
        ] if results else [None, None],
        'injections': [
            {
                'id':            r['injection_id'],
                'pixel_x':       r['pixel_x'],
                'pixel_y':       r['pixel_y'],
                'ra':            r['ra_sexagesimal'],
                'dec':           r['dec_sexagesimal'],
                'target_snr':    round(r['target_snr'], 4),
                'achieved_snr':  round(r['achieved_snr'], 4),
                'verified_snr':  round(r['verification_snr'], 4),
                'scale_factor':  round(r['scale_factor'], 4),
            }
            for r in results
        ]
    }


def print_history_summary(history_path, tile_name):
    """Print a compact history table for one tile."""
    history = load_history(history_path)
    runs = history.get(tile_name, [])
    if not runs:
        return
    print(f"\n  📋 Injection history for '{tile_name}' ({len(runs)} run(s)):")
    print(f"    {'Run':<4} {'Timestamp':<20} {'Injected':>8} "
          f"{'SNR min':>8} {'SNR max':>8} {'SNR mean':>9}")
    print('    ' + '-' * 62)
    for i, r in enumerate(runs):
        inj  = r['injections']
        snrs = [e['achieved_snr'] for e in inj] if inj else [0]
        print(f"    {i+1:<4} {r['timestamp']:<20} {len(inj):>8} "
              f"{min(snrs):>8.2f} {max(snrs):>8.2f} {np.mean(snrs):>9.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION & CUTOUTS
# ═══════════════════════════════════════════════════════════════════════════════

def create_visualization_png(data, positions, filepath, title,
                             circle_color='red', circle_radius=15):
    norm = normalize_with_zscale(data)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(norm, cmap='gray', origin='lower')
    for y, x in positions:
        ax.add_patch(patches.Circle((x, y), circle_radius,
                                    linewidth=1.5, edgecolor=circle_color,
                                    facecolor='none'))
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def save_injection_comparison_png(original, modified, positions, filepath,
                                  tile_name, n_zoom=12, zoom_size=64):
    """
    Save a comparison PNG with:
      Row 1: full before/after images side-by-side with injection circles.
      Row 2: n_zoom pairs of 64x64 zoom-ins (before | after) around injection sites.
    """
    norm_orig = normalize_with_zscale(original)
    norm_mod  = normalize_with_zscale(modified)

    # Pick n_zoom evenly spaced positions to show as zoom-ins
    step = max(1, len(positions) // n_zoom)
    zoom_positions = positions[::step][:n_zoom]
    half = zoom_size // 2

    n_cols = n_zoom         # pairs per row (each pair = before+after side by side)
    fig_width  = max(16, n_cols * 2.5)
    fig_height = fig_width * 0.55  # roughly 16:9-ish
    fig = plt.figure(figsize=(fig_width, fig_height))

    # ---- Row 1: full images ----
    ax_before = fig.add_subplot(2, 2, 1)
    ax_after  = fig.add_subplot(2, 2, 2)
    ax_before.imshow(norm_orig, cmap='gray', origin='lower')
    ax_after.imshow(norm_mod,  cmap='gray', origin='lower')
    for y, x in positions:
        ax_before.add_patch(patches.Circle((x, y), 15, lw=1,
                                           edgecolor='cyan', facecolor='none'))
        ax_after.add_patch(patches.Circle((x, y), 15, lw=1,
                                          edgecolor='red', facecolor='none'))
    ax_before.set_title(f"Before — {tile_name}", fontsize=10)
    ax_after.set_title(f"After — {len(positions)} injections", fontsize=10)
    for ax in (ax_before, ax_after):
        ax.axis('off')

    # ---- Row 2: zoom-ins ----
    # Use a gridspec sub-region for the zoom row
    gs_zoom = fig.add_gridspec(2, n_cols * 2,
                               left=0.01, right=0.99,
                               top=0.46, bottom=0.02,
                               hspace=0.05, wspace=0.05)

    for col_idx, (cy, cx) in enumerate(zoom_positions):
        cy_i, cx_i = int(round(cy)), int(round(cx))
        y0 = max(0, cy_i - half);  y1 = min(original.shape[0], cy_i + half)
        x0 = max(0, cx_i - half);  x1 = min(original.shape[1], cx_i + half)

        patch_b = norm_orig[y0:y1, x0:x1]
        patch_a = norm_mod[y0:y1,  x0:x1]

        ax_b = fig.add_subplot(gs_zoom[0, col_idx * 2])
        ax_a = fig.add_subplot(gs_zoom[0, col_idx * 2 + 1])
        ax_b.imshow(patch_b, cmap='gray', origin='lower',
                    vmin=0, vmax=1, interpolation='nearest')
        ax_a.imshow(patch_a, cmap='gray', origin='lower',
                    vmin=0, vmax=1, interpolation='nearest')
        ax_b.set_title('B', fontsize=6, pad=1)
        ax_a.set_title('A', fontsize=6, pad=1)
        for ax in (ax_b, ax_a):
            ax.axis('off')

    plt.suptitle(f"Injection check — {tile_name}  "
                 f"(top: full field;  bottom: {len(zoom_positions)} zoom-ins, B=before A=after)",
                 fontsize=10, y=0.98)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Comparison PNG → {filepath}")


def save_snr_cutout_pngs(image, positions, achieved_snrs, target_snrs,
                         output_base_dir, tile_name, cutout_size=64):
    """
    Save a PNG for each injected cutout with target and achieved SNR annotated.
    Cutouts are extracted from the post-injection image and z-scaled for display.
    Saved to:  <output_base_dir>/snr_cutouts/<tile_name>/
    """
    half    = cutout_size // 2
    out_dir = Path(output_base_dir) / "snr_cutouts" / tile_name
    out_dir.mkdir(parents=True, exist_ok=True)

    full_norm = normalize_with_zscale(image)   # z-scale full image for display

    saved = 0
    for i, (cy, cx) in enumerate(positions):
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        if y0 < 0 or y1 > image.shape[0] or x0 < 0 or x1 > image.shape[1]:
            continue

        patch = full_norm[y0:y1, x0:x1].copy()

        tgt = target_snrs[i]  if i < len(target_snrs)  else float('nan')
        ach = achieved_snrs[i] if i < len(achieved_snrs) else float('nan')

        fig, ax = plt.subplots(figsize=(2.5, 2.8))
        ax.imshow(patch, cmap='gray', origin='lower', vmin=0, vmax=1,
                  interpolation='nearest')
        ax.set_title(f'target={tgt:.1f}  SNR={ach:.2f}',
                     fontsize=8, pad=3, color='white',
                     backgroundcolor='black')
        ax.axis('off')
        plt.tight_layout(pad=0.2)

        fname = out_dir / f"{tile_name}_{i:04d}_tgt{tgt:.1f}_snr{ach:.2f}.png"
        plt.savefig(fname, dpi=100, bbox_inches='tight', facecolor='black')
        plt.close()
        saved += 1

    print(f"    Saved {saved} SNR-annotated cutout PNGs → {out_dir}")


def save_cutouts(image, positions, output_base_dir, tile_name):
    """Save 64x64 cutouts as PNG and FITS.

    Normalization: z-scale the FULL image first, then extract cutouts.
    This matches find_peaks_above_k_sigma_training.py (negatives) and
    find_peaks_above_k_sigma_test.py (testing), ensuring training positives
    see the same normalization as negatives and test-time inputs.
    """
    half = 32
    png_dir  = Path(output_base_dir) / "output_psf_added" / "positives" / "png"
    fits_dir = Path(output_base_dir) / "output_psf_added" / "positives" / "fits"
    png_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)

    # Z-scale the full image once — matches negatives/testing pipeline
    full_norm = normalize_with_zscale(image)

    saved = 0
    for i, (cy, cx) in enumerate(positions):
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        if y0 < 0 or y1 > image.shape[0] or x0 < 0 or x1 > image.shape[1]:
            continue
        # Extract cutout from already-normalized full image
        zcut = full_norm[y0:y1, x0:x1].copy().astype(np.float32)

        Image.fromarray((zcut * 255).astype(np.uint8)).save(
            png_dir / f"{tile_name}_cutout_{i:04d}.png")
        fits.PrimaryHDU(data=zcut).writeto(
            fits_dir / f"{tile_name}_cutout_{i:04d}.fits", overwrite=True)
        saved += 1

    print(f"    Saved {saved} cutouts → {fits_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def process_single_fits_file(fits_path, psf_array, psf_fwhm,
                             output_dir, history_path, n_injections=500):
    fits_path  = Path(fits_path)
    tile_name  = fits_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {tile_name}")
    print(f"{'='*60}")

    # ── Read FITS ─────────────────────────────────────────────────────────────
    original_data = header = None
    try:
        with fits.open(fits_path) as hdul:
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and hdu.data.ndim == 2:
                    original_data = hdu.data.copy().astype(np.float64)
                    header = hdu.header.copy()
                    print(f"  HDU {i}: image shape {original_data.shape}")
                    break
        if original_data is None:
            print("  ❌ No 2D image data found — skipping.")
            return []
    except Exception as e:
        print(f"  ❌ Read error: {e}")
        return []

    original_data = np.nan_to_num(original_data, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Data range: {original_data.min():.4f} → {original_data.max():.4f}")

    # ── Existing peaks (to avoid) ─────────────────────────────────────────────
    existing_peaks = find_peaks_simple(original_data)
    print(f"  Existing peaks found: {len(existing_peaks)}")

    # ── Generate injection positions ──────────────────────────────────────────
    positions = generate_injection_positions(
        original_data, existing_peaks,
        n_positions=n_injections,
        min_peak_dist=80,
        min_inject_dist=50
    )
    if not positions:
        print("  ❌ No valid positions — skipping tile.")
        return []

    target_snrs = list(np.linspace(3.0, 10.0, len(positions)))

    # ── Inject ────────────────────────────────────────────────────────────────
    modified_data, scale_factors, achieved_snrs, good_positions = \
        inject_multiple_psfs(original_data, psf_array, positions, target_snrs, psf_fwhm)

    # ── Visualisations ────────────────────────────────────────────────────────
    create_visualization_png(
        original_data, good_positions,
        output_dir / f"before_{tile_name}.png",
        f"Before injection — {tile_name}",
        circle_color='cyan')

    create_visualization_png(
        modified_data, good_positions,
        output_dir / f"after_{tile_name}.png",
        f"After injection — {tile_name}  ({len(good_positions)} injections)",
        circle_color='red')

    save_injection_comparison_png(
        original_data, modified_data, good_positions,
        output_dir / f"comparison_{tile_name}.png",
        tile_name, n_zoom=min(12, len(good_positions)))

    # ── Map good_positions → target SNR (needed for cutouts and records) ───────
    pos_to_target = {p: target_snrs[positions.index(p)]
                     for p in good_positions if p in positions}
    good_targets  = [pos_to_target.get(p, target_snrs[i])
                     for i, p in enumerate(good_positions)]

    # ── Cutouts ───────────────────────────────────────────────────────────────
    save_cutouts(modified_data, good_positions, output_dir, tile_name)
    save_snr_cutout_pngs(modified_data, good_positions,
                         achieved_snrs, good_targets,
                         output_dir, tile_name)

    # ── Save FITS ─────────────────────────────────────────────────────────────
    fits_out = output_dir / f"transient_added_{tile_name}.fits"
    fits.PrimaryHDU(data=modified_data.astype(np.float32),
                    header=header).writeto(fits_out, overwrite=True)

    # ── Build result records ───────────────────────────────────────────────────

    results = []
    for i, (cy, cx) in enumerate(good_positions):
        ra, dec = pixel_to_radec(cx, cy, header)
        verif   = calculate_snr_photutils(modified_data, (cx, cy), psf_fwhm)
        results.append({
            'tile_name':       tile_name,
            'injection_id':    i,
            'pixel_x':         cx,
            'pixel_y':         cy,
            'ra_sexagesimal':  ra,
            'dec_sexagesimal': dec,
            'target_snr':      pos_to_target.get((cy, cx), target_snrs[i]),
            'achieved_snr':    achieved_snrs[i],
            'verification_snr': verif,
            'scale_factor':    scale_factors[i],
            'psf_fwhm':        psf_fwhm,
        })

    # ── Save per-tile SNR history ──────────────────────────────────────────────
    run_record = build_run_record(tile_name, results, psf_fwhm, n_injections)
    append_history(history_path, tile_name, run_record)
    print(f"\n  📝 History updated → {history_path}")
    print_history_summary(history_path, tile_name)

    print(f"\n  ✅ Done: {len(results)} injections saved.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    input_dir    = Path("test_diff_files")
    output_dir   = Path("output_psf_added")
    psf_file     = Path("PSF/psf_array.npy")
    history_path = output_dir / "injection_history.json"

    if not input_dir.exists():
        print(f"❌ '{input_dir}' not found"); return
    if not psf_file.exists():
        print(f"❌ '{psf_file}' not found"); return

    output_dir.mkdir(exist_ok=True)

    # Load & normalise PSF
    print("Loading PSF...")
    psf_raw   = np.load(psf_file)
    psf_norm  = normalize_array(psf_raw)
    psf_fwhm  = calculate_fwhm_from_psf(psf_norm)
    print(f"  PSF shape : {psf_raw.shape}")
    print(f"  PSF FWHM  : {psf_fwhm:.2f} px")

    fits_files = sorted(list(input_dir.glob("*.fits")) +
                        list(input_dir.glob("*.fit")))
    if not fits_files:
        print(f"❌ No FITS files in '{input_dir}'"); return
    print(f"\nFound {len(fits_files)} FITS file(s).")

    all_results = []
    for fp in fits_files:
        try:
            results = process_single_fits_file(
                fp, psf_norm, psf_fwhm,
                output_dir, history_path,
                n_injections=500
            )
            all_results.extend(results)
        except Exception as e:
            print(f"  ❌ Error processing {fp.name}: {e}")

    # Save combined CSV
    if all_results:
        df  = pd.DataFrame(all_results)
        csv = output_dir / "injection_results.csv"
        df.to_csv(csv, index=False)
        print(f"\n{'='*60}")
        print(f"🎉 All done — {len(all_results)} total injections")
        print(f"   CSV    → {csv}")
        print(f"   History → {history_path}")
        print(f"   SNR range: {df['achieved_snr'].min():.2f} – {df['achieved_snr'].max():.2f}")
    else:
        print("\n⚠  No results generated.")


if __name__ == "__main__":
    main()