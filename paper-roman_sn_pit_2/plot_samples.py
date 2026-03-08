#!/usr/bin/env python3
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval

random.seed(7)

POS_DIR     = Path("split_folders/train/positives")
NEG_DIR     = Path("split_folders/train/negatives")
CSV_PATH    = Path("output_psf_added/injection_results.csv")


def load_fits(path):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def zscale(data):
    valid = np.isfinite(data) & (data != 0.0)
    if not valid.any():
        return np.zeros_like(data)
    zs = ZScaleInterval()
    vmin, vmax = zs.get_limits(data[valid])
    out = np.clip((data - vmin) / (vmax - vmin + 1e-12), 0, 1)
    out[~valid] = 0.0
    return out


def is_clean(path):
    """
    Reject cutouts that produce visual artifacts after ZScale rendering.
    Criteria: combined fraction of exactly-0 and exactly-1 pixels after
    ZScale must be < 5% (avoids checkerboard / saturated patches).
    """
    img = zscale(load_fits(path))
    frac_extremes = (img == 0.0).mean() + (img == 1.0).mean()
    return frac_extremes < 0.05


# ── Build filename → achieved_snr lookup from injection CSV ──────────────────
df = pd.read_csv(CSV_PATH)
# Reconstruct the cutout filename: {tile_name}_cutout_{injection_id:04d}.fits
df["filename"] = df.apply(
    lambda r: f"{r['tile_name']}_cutout_{int(r['injection_id']):04d}.fits", axis=1
)
snr_lookup = dict(zip(df["filename"], df["achieved_snr"]))

# ── Filter positive files in the train split to those that have SNR entries ───
SKIP = {"decorr_diff_R062_10050_11_-_R062_7752_3_cutout_0052.fits"}
pos_files = [p for p in POS_DIR.glob("*.fits")
             if p.name in snr_lookup and is_clean(p) and p.name not in SKIP]
print(f"Positives with known SNR: {len(pos_files)}")

snr_records = sorted([(snr_lookup[p.name], p) for p in pos_files], key=lambda x: x[0])
snr_min = snr_records[0][0]
snr_max = snr_records[-1][0]
print(f"SNR range: {snr_min:.2f} – {snr_max:.2f}")

# ── Pick 5 positives spanning the full SNR range evenly ──────────────────────
targets = np.linspace(snr_min, snr_max, 5)
pos_sample = []
used = set()
for t in targets:
    best = min(
        ((abs(snr - t), i, snr, p)
         for i, (snr, p) in enumerate(snr_records)
         if i not in used),
        key=lambda x: x[0]
    )
    _, idx, snr, path = best
    used.add(idx)
    pos_sample.append((snr, path))
    print(f"  target SNR={t:.1f} → matched SNR={snr:.2f}  ({path.name})")

# ── Random negatives: mix of peak-finder peaks and background non-peaks ───────
all_neg  = list(NEG_DIR.glob("*.fits"))
peak_neg = [f for f in all_neg if "_peak_"    in f.name and is_clean(f)]
bkg_neg  = [f for f in all_neg if "_nonpeak_" in f.name and is_clean(f)]
neg_sample = random.sample(peak_neg, 3) + random.sample(bkg_neg, 2)
random.shuffle(neg_sample)

# ── Plot ──────────────────────────────────────────────────────────────────────
panel = 1.5
fig_w = 5 * panel
fig_h = 2 * panel

fig = plt.figure(figsize=(fig_w, fig_h))

gs = gridspec.GridSpec(
    2, 5,
    figure=fig,
    left=0.0, right=1.0,
    top=0.97, bottom=0.03,
    wspace=0.02, hspace=0.02,
)

for col, (snr, fpath) in enumerate(pos_sample):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(zscale(load_fits(fpath)), cmap="gray", origin="lower",
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(f"SNR: {snr:.1f}", fontsize=8, pad=2, color="black",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=1.0, ec="none"))
    if col == 0:
        ax.set_ylabel("Transient", fontsize=9, labelpad=4)

for col, fpath in enumerate(neg_sample):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(zscale(load_fits(fpath)), cmap="gray", origin="lower",
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    if col == 0:
        ax.set_ylabel("Non-transient", fontsize=9, labelpad=4)

fig.savefig("sample_cutouts.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig("sample_cutouts.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("Saved sample_cutouts.pdf and sample_cutouts.png")
plt.close(fig)
