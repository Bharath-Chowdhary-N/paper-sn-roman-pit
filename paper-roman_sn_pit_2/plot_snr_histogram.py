#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("output_psf_added/injection_results.csv")

# ── Build filename → SNR lookup ───────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["filename"] = df.apply(
    lambda r: f"{r['tile_name']}_cutout_{int(r['injection_id']):04d}.fits", axis=1
)
snr_lookup = dict(zip(df["filename"], df["achieved_snr"]))

# ── Collect SNR values per split ──────────────────────────────────────────────
splits = {}
for split in ["train", "test"]:
    files = list(Path(f"split_folders/{split}/positives").glob("*.fits"))
    splits[split] = sorted([snr_lookup[f.name] for f in files if f.name in snr_lookup])
    print(f"{split}: {len(splits[split])} samples, "
          f"SNR {min(splits[split]):.2f} – {max(splits[split]):.2f}, "
          f"median {np.median(splits[split]):.2f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
bins = np.linspace(0, 13, 27)   # 0.5-wide bins across SNR 0–13

colors = {"train": "#2166ac", "test": "#d6604d"}
labels = {"train": "Training set", "test": "Test set"}

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=False)

for ax, split in zip(axes, ["train", "test"]):
    snrs = splits[split]
    ax.hist(snrs, bins=bins, color=colors[split], edgecolor="white",
            linewidth=0.4, alpha=0.9)
    ax.axvline(np.median(snrs), color="black", linewidth=1.2,
               linestyle="--", label=f"Median = {np.median(snrs):.1f}")
    ax.set_xlabel("SNR", fontsize=10)
    ax.set_ylabel("Number of samples", fontsize=10)
    ax.set_title(labels[split], fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, 13)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.8)
fig.savefig("snr_histogram.pdf", dpi=300, bbox_inches="tight")
fig.savefig("snr_histogram.png", dpi=300, bbox_inches="tight")
print("Saved snr_histogram.pdf and snr_histogram.png")
plt.close(fig)
