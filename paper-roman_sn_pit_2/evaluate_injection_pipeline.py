#!/usr/bin/env python3
"""
Injection-Recovery Evaluation Pipeline

Workflow
--------
1. Inject PSFs (SNR 3–10, same distribution as training) into every FITS file
   in test_directory/ using functions from psf_injection_script.py.
2. Run find_peaks_above_k_sigma_test.process_single_diff_file on the injected image.
3. Match detected peaks to injection positions:
      distance <= MATCH_RADIUS px  →  label = 1 (positive / transient)
      otherwise                    →  label = 0 (negative)
4. Classify every detected peak with the full 6-family ensemble.
5. Save:
      injection_eval_results/auroc_curve.png        – single ensemble AUROC
      injection_eval_results/confusion_matrices.png – one 2×2 matrix per family
      injection_eval_results/peak_results.csv
      injection_eval_results/injection_results.csv
      injection_eval_results/summary.txt

Usage
-----
    python evaluate_injection_pipeline.py
"""

import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── make sibling scripts importable ──────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from psf_injection_script import (
    normalize_array, calculate_fwhm_from_psf,
    find_peaks_simple, generate_injection_positions,
    inject_multiple_psfs,
)
from find_peaks_above_k_sigma_test import process_single_diff_file

# ============================================================================
# Configuration
# ============================================================================
N_INJECTIONS  = 500
SNR_MIN       = 1.0
SNR_MAX       = 10.0
MATCH_RADIUS  = 5          # pixels — peak within this distance → positive
CUTOUT_SIZE   = 64
SIGMA_THRESH  = 3.5        # same as find_peaks_above_k_sigma_test.py main()
OUTPUT_DIR    = Path("injection_eval_results")
TEST_DIR      = Path("test_directory")
PSF_FILE      = Path("PSF/psf_array.npy")

# ============================================================================
# Normalization — identical to training_script.py
# ============================================================================
def normalize_with_zscale(data):
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data)
    zscale = ZScaleInterval()
    try:
        vmin, vmax = zscale.get_limits(data[valid_mask])
        if vmax > vmin:
            normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        else:
            normalized = np.zeros_like(data)
        normalized[~valid_mask] = 0
    except Exception:
        if data.max() > data.min():
            normalized = (data - data.min()) / (data.max() - data.min())
        else:
            normalized = np.zeros_like(data)
    return normalized.astype(np.float32)

# ============================================================================
# Model architectures — identical to testing_script.py / training_script.py
# ============================================================================

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def bn_function(self, inputs):
        prev = [inputs] if isinstance(inputs, torch.Tensor) else inputs
        return self.conv1(self.relu1(self.norm1(torch.cat(prev, 1))))

    def forward(self, input):
        prev = [input] if isinstance(input, torch.Tensor) else input
        new_features = self.conv2(self.relu2(self.norm2(self.bn_function(prev))))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f'denselayer{i+1}', _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.items():
            features.append(layer(features))
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 32, 32),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=7,
                                                    stride=2, padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features //= 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.features(x), inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.classifier(torch.flatten(out, 1))
        return torch.sigmoid(out).squeeze(1)

def create_densenet(num_classes=1, drop_rate=0.0, **kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32),
                    num_classes=num_classes, drop_rate=drop_rate)

class TimmClassifier(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=False, dropout=0.3, img_size=64):
        super().__init__()
        kw = {}
        if 'deit' in model_name or 'vit' in model_name:
            kw['img_size'] = img_size
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=0, global_pool='avg', **kw)
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return torch.sigmoid(self.classifier(self.backbone(x))).squeeze(1)

def create_resnext(num_classes=1, dropout=0.3, **kw):
    return TimmClassifier('resnext50_32x4d', num_classes=num_classes, dropout=dropout)

def create_regnety(num_classes=1, dropout=0.3, **kw):
    return TimmClassifier('regnety_016', num_classes=num_classes, dropout=dropout)

def create_efficientnet(num_classes=1, dropout=0.3, **kw):
    return TimmClassifier('efficientnet_b0', num_classes=num_classes, dropout=dropout)

def create_convnext(num_classes=1, dropout=0.3, **kw):
    return TimmClassifier('convnext_tiny', num_classes=num_classes, dropout=dropout)

class DeiTClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        self.deit = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained,
                                      num_classes=0, img_size=64, global_pool='avg')
        feature_dim = self.deit.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return torch.sigmoid(self.classifier(self.deit(x))).squeeze(1)

def create_deit(num_classes=1, **kw):
    return DeiTClassifier(num_classes=num_classes)

ENSEMBLE_CONFIG = [
    ('DenseNet169',    create_densenet,     False),
    ('ResNeXt50',      create_resnext,      False),
    ('RegNetY016',     create_regnety,      False),
    ('EfficientNetB0', create_efficientnet, False),
    ('ConvNeXtTiny',   create_convnext,     False),
    ('DeiTTiny',       create_deit,         True),
]

def load_family_models(type_name, create_fn, device):
    models = []
    for path in sorted(Path('.').glob(f'{type_name}_Ensemble_Model*_best.pth')):
        try:
            model = create_fn(num_classes=1)
            ckpt  = torch.load(path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device).eval()
            models.append(model)
            print(f"    Loaded {path.name}")
        except Exception as e:
            print(f"    Failed {path.name}: {e}")
    return models

# ============================================================================
# Step 1 — Injection
# ============================================================================
def run_injection(fits_path, psf_norm, psf_fwhm, n_injections, output_dir):
    """
    Inject PSFs into fits_path, save the injected FITS, return metadata.

    Returns
    -------
    injected_path : Path
    good_positions : list of (cy, cx) tuples
    achieved_snrs  : list of float
    target_snrs    : list of float (aligned with good_positions)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                raw = hdu.data.copy().astype(np.float64)
                header = hdu.header.copy()
                break
        else:
            print(f"  No valid data in {fits_path.name}")
            return None, [], [], []

    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if raw.ndim > 2:
        raw = raw[0] if raw.ndim == 3 else raw.squeeze()

    existing_peaks = find_peaks_simple(raw)
    positions = generate_injection_positions(
        raw, existing_peaks,
        n_positions=n_injections,
        min_peak_dist=80,
        min_inject_dist=50,
    )
    if not positions:
        print(f"  No valid injection positions found in {fits_path.name}")
        return None, [], [], []

    target_snrs_all = list(np.linspace(SNR_MIN, SNR_MAX, len(positions)))
    # build lookup so we can recover target SNR for each surviving position
    pos_to_target = {p: t for p, t in zip(positions, target_snrs_all)}

    modified, _, achieved_snrs, good_positions = inject_multiple_psfs(
        raw, psf_norm, positions, target_snrs_all, psf_fwhm)

    target_snrs = [pos_to_target.get(p, 0.0) for p in good_positions]

    injected_path = output_dir / f"injected_{fits_path.stem}.fits"
    fits.PrimaryHDU(data=modified.astype(np.float32),
                    header=header).writeto(injected_path, overwrite=True)
    print(f"  Injected {len(good_positions)}/{len(positions)} PSFs → {injected_path.name}")
    return injected_path, good_positions, achieved_snrs, target_snrs, modified

# ============================================================================
# Step 1b — Save injection preview PNGs (before classification)
# ============================================================================
def save_injection_preview_pngs(modified_image, good_positions,
                                 achieved_snrs, target_snrs,
                                 output_dir, stem):
    """
    Save one PNG per injected source immediately after injection,
    before peak-finding or classification begins.

    Each PNG is a 64×64 cutout (z-scaled from the full injected image)
    with target SNR and achieved SNR shown in the title.

    Output: <output_dir>/injection_previews/<stem>/
            <stem>_<i:04d>_tgt<T>_snr<A>.png
    """
    half    = CUTOUT_SIZE // 2
    out_dir = output_dir / "injection_previews" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    full_norm = normalize_with_zscale(modified_image.astype(np.float32))

    saved = 0
    for i, (cy, cx) in enumerate(good_positions):
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        if y0 < 0 or y1 > modified_image.shape[0] \
                   or x0 < 0 or x1 > modified_image.shape[1]:
            continue

        patch = full_norm[y0:y1, x0:x1]
        tgt   = target_snrs[i]   if i < len(target_snrs)   else float('nan')
        ach   = achieved_snrs[i] if i < len(achieved_snrs) else float('nan')

        fig, ax = plt.subplots(figsize=(2.5, 2.8))
        ax.imshow(patch, cmap='gray', origin='lower', vmin=0, vmax=1,
                  interpolation='nearest')
        ax.set_title(f'target={tgt:.1f}  SNR={ach:.2f}',
                     fontsize=8, pad=3, color='white',
                     backgroundcolor='black')
        ax.axis('off')
        plt.tight_layout(pad=0.2)
        fname = out_dir / f"{stem}_{i:04d}_tgt{tgt:.1f}_snr{ach:.2f}.png"
        plt.savefig(fname, dpi=100, bbox_inches='tight', facecolor='black')
        plt.close()
        saved += 1

    print(f"  Injection previews ({saved} PNGs) → {out_dir}")


# ============================================================================
# Step 2 — Peak finder (reuses find_peaks_above_k_sigma_test)
# ============================================================================
def run_peak_finder(injected_fits_path, peak_cutouts_dir):
    """Run the standard test-pipeline peak finder on the injected image."""
    peak_cutouts_dir.mkdir(parents=True, exist_ok=True)
    n = process_single_diff_file(
        injected_fits_path,
        peak_cutouts_dir,
        cutout_size=CUTOUT_SIZE,
        sigma_threshold=SIGMA_THRESH,
        centering_method='centroid',
        edge_buffer=32,
        max_peaks=5000,
        allow_edge_peaks=True,
    )
    return n

# ============================================================================
# Step 3 — Match peaks to injections
# ============================================================================
_POS_RE = re.compile(r'_y(\d+)_x(\d+)_')

def parse_peak_pos(fname):
    """Extract (y, x) integer position from cutout filename."""
    m = _POS_RE.search(fname)
    return (int(m.group(1)), int(m.group(2))) if m else None

def match_peaks_to_injections(peak_files, good_positions, radius=MATCH_RADIUS):
    """
    Returns
    -------
    peak_labels        : int array (n_peaks,)  — 1 if within radius of any injection
    injection_recovered: bool array (n_inj,)   — True if any peak matched this injection
    inj_to_peak        : int array (n_inj,)    — index into peak_files for matched peak
                                                  (-1 if injection was not recovered)
    """
    inj_arr = np.array(good_positions, dtype=float)  # (N, 2) [y, x]
    n_inj   = len(inj_arr)

    peak_labels         = np.zeros(len(peak_files), dtype=int)
    injection_recovered = np.zeros(n_inj, dtype=bool)
    inj_to_peak         = np.full(n_inj, -1, dtype=int)

    for pi, pf in enumerate(peak_files):
        pos = parse_peak_pos(pf.name)
        if pos is None or n_inj == 0:
            continue
        py, px = pos
        dists = np.hypot(inj_arr[:, 0] - py, inj_arr[:, 1] - px)
        nearest = int(np.argmin(dists))
        if dists[nearest] <= radius:
            peak_labels[pi] = 1
            injection_recovered[nearest] = True
            inj_to_peak[nearest] = pi

    return peak_labels, injection_recovered, inj_to_peak

# ============================================================================
# Step 4 — Classification
# ============================================================================
def classify_cutout(fits_path, models, device):
    """
    Load cutout FITS (already 1st-z-scaled by peak finder),
    apply 2nd z-scale to match training, return mean probability.
    """
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data.astype(np.float32)
        if data.ndim > 2:
            data = data.squeeze()
        if data.shape != (CUTOUT_SIZE, CUTOUT_SIZE):
            return None
        data   = normalize_with_zscale(data)            # 2nd z-scale
        tensor = (torch.from_numpy(data).float()
                  .unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device))
        probs = []
        with torch.no_grad():
            for model in models:
                out = model(tensor)
                probs.append(out.item() if out.dim() == 0 else out[0].item())
        return float(np.mean(probs))
    except Exception:
        return None

# ============================================================================
# Evaluation helpers
# ============================================================================
def compute_roc(y_true, y_score):
    """ROC curve without sklearn. Returns (fpr, tpr, auc)."""
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return np.array([0., 1.]), np.array([0., 1.]), 0.5

    thresholds = np.concatenate([[1.0 + 1e-9],
                                 np.sort(np.unique(y_score))[::-1],
                                 [-1e-9]])
    tprs, fprs = [], []
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / pos)
        fprs.append(fp / neg)

    fprs = np.array(fprs); tprs = np.array(tprs)
    idx  = np.argsort(fprs)
    fprs, tprs = fprs[idx], tprs[idx]
    auc_val = float(np.trapz(tprs, fprs))
    return fprs, tprs, auc_val

def confusion_at_05(y_true, y_score):
    """2×2 confusion matrix [[TN,FP],[FN,TP]] at threshold 0.5."""
    pred = (np.asarray(y_score) >= 0.5).astype(int)
    y    = np.asarray(y_true)
    TP = int(((pred == 1) & (y == 1)).sum())
    FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum())
    TN = int(((pred == 0) & (y == 0)).sum())
    return np.array([[TN, FP], [FN, TP]])

def confusion_at_threshold(y_true, y_score, threshold):
    """2×2 confusion matrix [[TN,FP],[FN,TP]] at a specific threshold."""
    pred = (np.asarray(y_score) >= threshold).astype(int)
    y    = np.asarray(y_true)
    TP = int(((pred == 1) & (y == 1)).sum())
    FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum())
    TN = int(((pred == 0) & (y == 0)).sum())
    return np.array([[TN, FP], [FN, TP]])

def find_threshold_at_tpr(y_true, y_score, target_tpr):
    """
    Find the highest threshold (most selective) where TPR ≥ target_tpr.
    Scans thresholds from high to low; TPR rises as threshold falls.
    Returns the first threshold at which TPR reaches the target.
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = int((y_true == 1).sum())
    if pos == 0:
        return 0.5
    thresholds = np.sort(np.unique(y_score))[::-1]  # high → low
    for thr in thresholds:
        tp  = int(((y_score >= thr) & (y_true == 1)).sum())
        tpr = tp / pos
        if tpr >= target_tpr:
            return float(thr)
    return float(thresholds[-1])   # fallback: lowest threshold

def compute_family_thresholds(y_true, family_names, family_scores,
                               target_tprs=(0.9, 0.75, 0.5)):
    """
    Returns dict: {family_name: {tpr_target: threshold}}.
    """
    thresholds = {}
    for name, scores in zip(family_names, family_scores):
        thresholds[name] = {
            t: find_threshold_at_tpr(y_true, scores, t) for t in target_tprs
        }
    return thresholds

# ============================================================================
# Plots
# ============================================================================
_FAMILY_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3', '#a65628']

def plot_roc_curves(y_true, family_names, family_scores, ensemble_scores, output_path):
    """One ROC curve per model family + ensemble, all on one figure."""
    fig, ax = plt.subplots(figsize=(7, 6))

    auc_vals = {}
    for i, (name, scores) in enumerate(zip(family_names, family_scores)):
        fprs, tprs, auc_val = compute_roc(y_true, scores)
        auc_vals[name] = auc_val
        ax.plot(fprs, tprs, color=_FAMILY_COLORS[i % len(_FAMILY_COLORS)],
                lw=1.8, label=f'{name}  (AUC={auc_val:.3f})')

    # ensemble on top — thicker black line
    fprs_ens, tprs_ens, auc_ens = compute_roc(y_true, ensemble_scores)
    auc_vals['Ensemble'] = auc_ens
    ax.plot(fprs_ens, tprs_ens, 'k-', lw=2.8,
            label=f'Ensemble  (AUC={auc_ens:.3f})')
    ax.fill_between(fprs_ens, tprs_ens, alpha=0.07, color='black')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Injection Recovery Evaluation', fontsize=13)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ROC curves saved → {output_path}")
    for name, auc in auc_vals.items():
        print(f"    {name:<22}  AUC = {auc:.3f}")
    return auc_vals


def plot_snr_detection(inj_df, family_names, output_path,
                       snr_bins=None, threshold=0.5):
    """
    2×3 grid of SNR vs detection-rate plots, one per model family.

    Detection rate for a given family = fraction of injections in that SNR bin
    where the family's mean probability >= threshold.
    Injections not recovered by the peak finder count as undetected (prob=0).

    A shared dashed grey line shows the raw peak-finder recovery rate
    (family-independent) for reference.
    """
    if snr_bins is None:
        snr_bins = np.arange(SNR_MIN, SNR_MAX + 0.01, 1.0)   # [3,4,5,6,7,8,9,10]

    bin_centres = 0.5 * (snr_bins[:-1] + snr_bins[1:])
    snr_vals    = inj_df['achieved_snr'].values

    # peak-finder recovery per bin (same for every subplot)
    finder_rate = []
    for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
        mask = (snr_vals >= lo) & (snr_vals < hi)
        n = mask.sum()
        finder_rate.append(inj_df.loc[mask, 'recovered'].sum() / n if n > 0 else np.nan)
    finder_rate = np.array(finder_rate)

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.5),
                             sharey=True, sharex=True)
    axes = axes.flatten()

    for idx, (name, color) in enumerate(zip(family_names, _FAMILY_COLORS)):
        ax = axes[idx]
        prob_col = f'{name}_prob'
        if prob_col not in inj_df.columns:
            ax.set_visible(False)
            continue

        det_rate = []
        n_per_bin = []
        for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
            mask = (snr_vals >= lo) & (snr_vals < hi)
            n = mask.sum()
            n_per_bin.append(n)
            if n == 0:
                det_rate.append(np.nan)
            else:
                detected = (inj_df.loc[mask, prob_col] >= threshold).sum()
                det_rate.append(detected / n)
        det_rate = np.array(det_rate)

        # bar chart for detection rate
        ax.bar(bin_centres, det_rate, width=0.8, color=color, alpha=0.65,
               label='Classified transient')
        # overlay finder recovery rate
        ax.step(np.concatenate([[snr_bins[0]], bin_centres, [snr_bins[-1]]]),
                np.concatenate([[finder_rate[0]], finder_rate, [finder_rate[-1]]]),
                where='mid', color='grey', lw=1.5, ls='--', label='Peak finder recovery')

        ax.set_title(name, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(snr_bins[0] - 0.2, snr_bins[-1] + 0.2)
        ax.set_xticks(snr_bins)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')

        # annotate bar tops with count
        for bc, dr, n in zip(bin_centres, det_rate, n_per_bin):
            if not np.isnan(dr) and n > 0:
                ax.text(bc, min(dr + 0.03, 0.98), f'n={n}',
                        ha='center', va='bottom', fontsize=6.5)

    for idx in range(len(family_names), len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel('Achieved SNR', fontsize=12)
    fig.supylabel('Detection Rate', fontsize=12)
    fig.suptitle(f'SNR vs Detection Rate per Model Family  (threshold={threshold})',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  SNR vs detection rate saved → {output_path}")


def plot_snr_detection_tpr(inj_df, family_names, thresholds, tpr_label,
                            output_path, snr_bins=None):
    """
    2×3 SNR vs detection-rate grid using per-family TPR-based thresholds.

    Annotation shows 'detected/total' (different per panel).
    thresholds : dict {family_name: threshold}
    tpr_label  : str, e.g. '90%TP'
    """
    if snr_bins is None:
        snr_bins = np.arange(SNR_MIN, SNR_MAX + 0.01, 1.0)

    bin_centres = 0.5 * (snr_bins[:-1] + snr_bins[1:])
    snr_vals    = inj_df['achieved_snr'].values

    # Peak-finder recovery per bin (shared across families)
    finder_rate = []
    for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
        mask = (snr_vals >= lo) & (snr_vals < hi)
        n    = mask.sum()
        finder_rate.append(inj_df.loc[mask, 'recovered'].sum() / n if n > 0 else np.nan)
    finder_rate = np.array(finder_rate)

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.5),
                             sharey=True, sharex=True)
    axes = axes.flatten()

    for idx, (name, color) in enumerate(zip(family_names, _FAMILY_COLORS)):
        ax       = axes[idx]
        prob_col = f'{name}_prob'
        if prob_col not in inj_df.columns:
            ax.set_visible(False)
            continue

        thr = thresholds.get(name, 0.5)

        det_rate   = []
        n_det_list = []
        n_tot_list = []
        for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
            mask = (snr_vals >= lo) & (snr_vals < hi)
            n    = int(mask.sum())
            n_tot_list.append(n)
            if n == 0:
                det_rate.append(np.nan)
                n_det_list.append(0)
            else:
                det = int((inj_df.loc[mask, prob_col] >= thr).sum())
                det_rate.append(det / n)
                n_det_list.append(det)
        det_rate = np.array(det_rate)

        ax.bar(bin_centres, det_rate, width=0.8, color=color, alpha=0.65,
               label='Classified transient')
        ax.step(np.concatenate([[snr_bins[0]], bin_centres, [snr_bins[-1]]]),
                np.concatenate([[finder_rate[0]], finder_rate, [finder_rate[-1]]]),
                where='mid', color='grey', lw=1.5, ls='--', label='Peak finder')

        ax.set_title(f'{name}\n(thr={thr:.3f})', fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_xlim(snr_bins[0] - 0.2, snr_bins[-1] + 0.2)
        ax.set_xticks(snr_bins)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')

        # Annotate with 'detected/total' — different per family
        for bc, dr, nd, nt in zip(bin_centres, det_rate, n_det_list, n_tot_list):
            if not np.isnan(dr) and nt > 0:
                ax.text(bc, min(dr + 0.04, 1.05), f'{nd}/{nt}',
                        ha='center', va='bottom', fontsize=6.5)

    for idx in range(len(family_names), len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel('Achieved SNR', fontsize=12)
    fig.supylabel('Detection Rate', fontsize=12)
    fig.suptitle(f'SNR vs Detection Rate per Model Family  ({tpr_label})',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  SNR vs detection rate ({tpr_label}) saved → {output_path}")

def plot_confusion_matrices(y_true, family_names, family_scores, output_path):
    """2×3 grid of confusion matrices, one per model family."""
    n = len(family_names)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.8))
    axes = axes.flatten()

    cell_labels = [['TN', 'FP'], ['FN', 'TP']]
    row_titles  = ['Actual: Neg', 'Actual: Pos']
    col_titles  = ['Pred: Neg', 'Pred: Pos']

    for i, (name, scores) in enumerate(zip(family_names, family_scores)):
        cm = confusion_at_05(y_true, scores)
        ax = axes[i]
        im = ax.imshow(cm, cmap='Blues', vmin=0)

        for r in range(2):
            for c in range(2):
                val  = cm[r, c]
                txt  = f"{cell_labels[r][c]}\n{val}"
                dark = val > cm.max() / 2
                ax.text(c, r, txt, ha='center', va='center',
                        fontsize=13, fontweight='bold',
                        color='white' if dark else 'black')

        ax.set_xticks([0, 1]); ax.set_xticklabels(col_titles, fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(row_titles, fontsize=9)

        n_pos = int(y_true.sum()); n_neg = len(y_true) - n_pos
        cm2x2 = cm
        prec = cm2x2[1,1] / (cm2x2[1,1]+cm2x2[0,1]) if (cm2x2[1,1]+cm2x2[0,1]) > 0 else 0
        rec  = cm2x2[1,1] / (cm2x2[1,1]+cm2x2[1,0]) if (cm2x2[1,1]+cm2x2[1,0]) > 0 else 0
        ax.set_title(f"{name}\nPrec={prec:.2f}  Rec={rec:.2f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Confusion Matrices per Model Family  (threshold = 0.5)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrices saved → {output_path}")


def plot_confusion_matrices_at_tpr(y_true, family_names, family_scores,
                                    thresholds, tpr_label, output_path):
    """
    2×3 confusion matrix grid at per-family TPR-based thresholds.

    thresholds : dict {family_name: threshold_value}
    tpr_label  : e.g. '90%TP' (used only for the title / filename)
    """
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.8))
    axes = axes.flatten()

    cell_labels = [['TN', 'FP'], ['FN', 'TP']]
    row_titles  = ['Actual: Neg', 'Actual: Pos']
    col_titles  = ['Pred: Neg', 'Pred: Pos']

    for i, (name, scores) in enumerate(zip(family_names, family_scores)):
        thr = thresholds.get(name, 0.5)
        cm  = confusion_at_threshold(y_true, scores, thr)
        ax  = axes[i]
        im  = ax.imshow(cm, cmap='Blues', vmin=0)

        for r in range(2):
            for c in range(2):
                val  = cm[r, c]
                txt  = f"{cell_labels[r][c]}\n{val}"
                dark = val > cm.max() / 2
                ax.text(c, r, txt, ha='center', va='center',
                        fontsize=13, fontweight='bold',
                        color='white' if dark else 'black')

        ax.set_xticks([0, 1]); ax.set_xticklabels(col_titles, fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(row_titles, fontsize=9)

        cm2 = cm
        prec = cm2[1,1] / (cm2[1,1]+cm2[0,1]) if (cm2[1,1]+cm2[0,1]) > 0 else 0
        rec  = cm2[1,1] / (cm2[1,1]+cm2[1,0]) if (cm2[1,1]+cm2[1,0]) > 0 else 0
        ax.set_title(f"{name}\nthr={thr:.3f}  Prec={prec:.2f}  Rec={rec:.2f}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(family_names), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Confusion Matrices per Model Family  ({tpr_label})',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrices ({tpr_label}) saved → {output_path}")

# ============================================================================
# TP / FP cutout grids
# ============================================================================
def _load_cutout_for_display(fits_path):
    """Load FITS cutout data; return float32 array in [0,1] or None."""
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data.astype(np.float32)
        if data.ndim > 2:
            data = data.squeeze()
        return np.clip(data, 0, 1)
    except Exception:
        return None


def save_tp_fp_grids(df, family_names, thresholds, tpr_label, output_dir):
    """
    For each family at the given TPR-based threshold:
      • Save a 5×5 PNG grid of TPs and a 5×5 PNG grid of FPs.
      • Save true_positives.csv and false_positives.csv.

    df must contain 'label', '{family}_prob', and 'cutout_path' columns.
    thresholds : dict {family_name: threshold}
    tpr_label  : str, e.g. '90pTP'
    """
    grid_root = output_dir / f"cutout_grids_{tpr_label}"

    for fam in family_names:
        prob_col = f'{fam}_prob'
        if prob_col not in df.columns:
            continue

        thr     = thresholds.get(fam, 0.5)
        pred_pos = df[prob_col] >= thr

        tp_df = df[pred_pos & (df['label'] == 1)].copy()
        fp_df = df[pred_pos & (df['label'] == 0)].copy()

        fam_dir = grid_root / fam
        fam_dir.mkdir(parents=True, exist_ok=True)

        # Save CSVs (drop cutout_path from CSVs to keep them tidy)
        csv_cols = [c for c in df.columns if c != 'cutout_path']
        tp_df[csv_cols].to_csv(fam_dir / 'true_positives.csv',  index=False)
        fp_df[csv_cols].to_csv(fam_dir / 'false_positives.csv', index=False)

        for label_str, subset_df in [('TP', tp_df), ('FP', fp_df)]:
            samples = subset_df.head(25)
            if len(samples) == 0:
                continue

            nrows, ncols = 5, 5
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
            axes_flat = axes.flatten()

            for i, (_, row) in enumerate(samples.iterrows()):
                ax = axes_flat[i]
                cp = row.get('cutout_path', '')
                if cp and Path(cp).exists():
                    data = _load_cutout_for_display(cp)
                    if data is not None:
                        ax.imshow(data, cmap='gray', origin='lower',
                                  vmin=0, vmax=1, aspect='equal')
                        ax.set_title(f'p={row[prob_col]:.2f}', fontsize=6, pad=2)
                    else:
                        ax.text(0.5, 0.5, 'ERR', ha='center', va='center',
                                transform=ax.transAxes, fontsize=7)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=7)
                ax.axis('off')

            for i in range(len(samples), nrows * ncols):
                axes_flat[i].set_visible(False)

            title = (f'{fam} — {label_str}s  '
                     f'(thr={thr:.3f}, {tpr_label})  '
                     f'[showing {len(samples)} of {len(subset_df)}]')
            fig.suptitle(title, fontsize=8)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(fam_dir / f'{label_str}_grid.png', dpi=150,
                        bbox_inches='tight')
            plt.close()

        print(f"  {fam:22} TP={len(tp_df):4d}  FP={len(fp_df):4d}  → {fam_dir}")


# ============================================================================
# SNR-bin cutout directories
# ============================================================================
def save_snr_bin_cutouts(inj_df, family_names, thresholds, snr_bins, output_dir):
    """
    For each family × each SNR bin, copy cutout PNGs to:
      snr_bin_cutouts/{family}/SNR_{lo}-{hi}/detected/
      snr_bin_cutouts/{family}/SNR_{lo}-{hi}/non_detected/

    Injections not recovered by the peak finder are counted as non-detected
    (no image to save).  recovered injections below the classifier threshold
    go to non_detected WITH their image.

    inj_df must contain 'achieved_snr', 'recovered', 'peak_cutout_path',
    and '{family}_prob' columns.
    thresholds : dict {family_name: threshold}
    snr_bins   : array of bin edges (e.g. [3,4,5,...,10])
    """
    from PIL import Image as PILImage

    bin_root = output_dir / "snr_bin_cutouts"
    snr_vals = inj_df['achieved_snr'].values

    for fam in family_names:
        prob_col = f'{fam}_prob'
        if prob_col not in inj_df.columns:
            continue
        thr = thresholds.get(fam, 0.5)

        for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
            mask   = (snr_vals >= lo) & (snr_vals < hi)
            bin_df = inj_df[mask]
            if len(bin_df) == 0:
                continue

            bin_label = f'SNR_{lo:.0f}-{hi:.0f}'
            det_dir  = bin_root / fam / bin_label / 'detected'
            ndet_dir = bin_root / fam / bin_label / 'non_detected'
            det_dir.mkdir(parents=True, exist_ok=True)
            ndet_dir.mkdir(parents=True, exist_ok=True)

            for _, row in bin_df.iterrows():
                recovered  = bool(row.get('recovered', False))
                cp         = str(row.get('peak_cutout_path', ''))
                prob       = float(row.get(prob_col, 0.0))
                classified = prob >= thr

                if not recovered or not cp or not Path(cp).exists():
                    # No cutout available — not recoverable, skip image
                    continue

                data = _load_cutout_for_display(cp)
                if data is None:
                    continue

                fname = Path(cp).stem + f'_p{prob:.3f}.png'
                dest  = det_dir if classified else ndet_dir
                img   = PILImage.fromarray(
                    (data * 255).astype(np.uint8), mode='L')
                img.save(dest / fname)

    print(f"  SNR bin cutouts saved → {bin_root}")


# ============================================================================
# Main
# ============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── PSF ────────────────────────────────────────────────────────────────────
    if not PSF_FILE.exists():
        print(f"PSF file not found: {PSF_FILE}"); return
    psf_raw  = np.load(PSF_FILE)
    psf_norm = normalize_array(psf_raw)
    psf_fwhm = calculate_fwhm_from_psf(psf_norm)
    print(f"PSF FWHM: {psf_fwhm:.2f} px")

    # ── Models ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    loaded = {}
    for type_name, create_fn, _ in ENSEMBLE_CONFIG:
        print(f"Loading {type_name}...")
        models = load_family_models(type_name, create_fn, device)
        if models:
            loaded[type_name] = models
            print(f"  → {len(models)} member(s)")
        else:
            print(f"  → no checkpoints found, skipping")

    active_families = [n for n, _, _ in ENSEMBLE_CONFIG if n in loaded]
    if not active_families:
        print("No models loaded — train first."); return

    # ── Test FITS files ───────────────────────────────────────────────────────
    fits_files = sorted(list(TEST_DIR.glob("*.fits")) + list(TEST_DIR.glob("*.fit")))
    if not fits_files:
        print(f"No FITS files in {TEST_DIR}"); return
    print(f"\nFound {len(fits_files)} test file(s) in {TEST_DIR}\n")

    all_peak_rows = []
    all_inj_rows  = []

    for fits_path in fits_files:
        print(f"\n{'='*60}")
        print(f"File: {fits_path.name}")
        print('='*60)

        # Step 1: inject
        inj_dir = OUTPUT_DIR / "injected"
        injected_path, good_positions, achieved_snrs, target_snrs, modified_image = \
            run_injection(fits_path, psf_norm, psf_fwhm, N_INJECTIONS, inj_dir)

        if not good_positions:
            continue

        # Step 1b: save per-injection cutout PNGs NOW — before classification
        save_injection_preview_pngs(
            modified_image, good_positions, achieved_snrs, target_snrs,
            OUTPUT_DIR, fits_path.stem)

        # Step 2: peak finder on injected image
        peak_dir = OUTPUT_DIR / "peak_cutouts" / fits_path.stem
        run_peak_finder(injected_path, peak_dir)
        peak_files = sorted(peak_dir.glob("*.fits"))

        if not peak_files:
            print("  Peak finder found no peaks.")
            # record all injections as not recovered, probs = 0
            for i, (cy, cx) in enumerate(good_positions):
                inj_row = {
                    'file':             fits_path.name,
                    'inj_y':            int(cy), 'inj_x': int(cx),
                    'target_snr':       round(target_snrs[i], 3),
                    'achieved_snr':     round(achieved_snrs[i], 3),
                    'recovered':        False,
                    'ensemble_prob':    0.0,
                    'peak_cutout_path': '',
                }
                for fam in active_families:
                    inj_row[f'{fam}_prob'] = 0.0
                all_inj_rows.append(inj_row)
            continue

        # Step 3: match peaks → labels + injection→peak index map
        peak_labels, injection_recovered, inj_to_peak = match_peaks_to_injections(
            peak_files, good_positions, radius=MATCH_RADIUS)

        n_rec = int(injection_recovered.sum())
        print(f"  Injections: {len(good_positions)}  |  "
              f"Peaks found: {len(peak_files)}  |  "
              f"Recovered: {n_rec} / {len(good_positions)} "
              f"({100*n_rec/max(len(good_positions),1):.1f}%)")

        # Step 4: classify every detected peak; keep results in a list indexed by pi
        print(f"  Classifying {len(peak_files)} peak cutouts...")
        file_peak_rows = []
        for pi, pf in enumerate(tqdm(peak_files, desc="  ", leave=False)):
            row = {
                'file':        fits_path.name,
                'cutout':      pf.name,
                'cutout_path': str(pf),
                'label':       int(peak_labels[pi]),
            }
            fam_probs = []
            for fam in active_families:
                prob = classify_cutout(pf, loaded[fam], device)
                if prob is None:
                    prob = 0.0
                row[f'{fam}_prob'] = round(prob, 5)
                fam_probs.append(prob)
            row['ensemble_prob'] = round(float(np.mean(fam_probs)), 5)
            file_peak_rows.append(row)
            all_peak_rows.append(row)

        # record per-injection info; attach classification probs from matched peak
        for i, (cy, cx) in enumerate(good_positions):
            inj_row = {
                'file':         fits_path.name,
                'inj_y':        int(cy),
                'inj_x':        int(cx),
                'target_snr':   round(target_snrs[i], 3),
                'achieved_snr': round(achieved_snrs[i], 3),
                'recovered':    bool(injection_recovered[i]),
            }
            pi = int(inj_to_peak[i])
            if pi >= 0 and pi < len(file_peak_rows):
                for fam in active_families:
                    inj_row[f'{fam}_prob'] = file_peak_rows[pi].get(f'{fam}_prob', 0.0)
                inj_row['ensemble_prob']      = file_peak_rows[pi].get('ensemble_prob', 0.0)
                inj_row['peak_cutout_path']   = file_peak_rows[pi].get('cutout_path', '')
            else:
                for fam in active_families:
                    inj_row[f'{fam}_prob'] = 0.0
                inj_row['ensemble_prob']    = 0.0
                inj_row['peak_cutout_path'] = ''
            all_inj_rows.append(inj_row)

    # ── Aggregate & save CSVs ─────────────────────────────────────────────────
    if not all_peak_rows:
        print("\nNo peaks to evaluate."); return

    df     = pd.DataFrame(all_peak_rows)
    inj_df = pd.DataFrame(all_inj_rows)
    df.to_csv(    OUTPUT_DIR / 'peak_results.csv',       index=False)
    inj_df.to_csv(OUTPUT_DIR / 'injection_results.csv',  index=False)

    y_true          = df['label'].values
    ensemble_scores = df['ensemble_prob'].values
    family_scores   = [df[f'{n}_prob'].values for n in active_families]

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    print(f"\n  Classification dataset: {n_pos} positives, {n_neg} negatives")

    if n_pos == 0:
        print("  WARNING: no positive peaks found — increase MATCH_RADIUS or check injections.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating plots...")
    auc_vals = plot_roc_curves(y_true, active_families, family_scores,
                               ensemble_scores, OUTPUT_DIR / 'roc_curves.png')

    # Baseline confusion matrices at 0.5
    plot_confusion_matrices(y_true, active_families, family_scores,
                            OUTPUT_DIR / 'confusion_matrices.png')

    # Baseline SNR detection rate at 0.5
    plot_snr_detection(inj_df, active_families, OUTPUT_DIR / 'snr_detection_rate.png')

    # ── TPR-based thresholds (90%, 75%, 50% TP recovery) ─────────────────────
    print("\nComputing per-family TPR-based thresholds...")
    TARGET_TPRS = (0.90, 0.75, 0.50)
    family_thresholds = compute_family_thresholds(
        y_true, active_families, family_scores, target_tprs=TARGET_TPRS)

    for tpr_target in TARGET_TPRS:
        tpr_pct   = int(round(tpr_target * 100))
        tpr_label = f'{tpr_pct}pTP'          # e.g. '90pTP'
        tpr_title = f'{tpr_pct}% TP recovery'

        thr_dict = {fam: family_thresholds[fam][tpr_target]
                    for fam in active_families}
        print(f"\n  {tpr_title}:")
        for fam, thr in thr_dict.items():
            print(f"    {fam:<22} threshold = {thr:.4f}")

        # Confusion matrices at TPR-based thresholds
        plot_confusion_matrices_at_tpr(
            y_true, active_families, family_scores, thr_dict,
            tpr_title,
            OUTPUT_DIR / f'confusion_matrices_{tpr_label}.png')

        # SNR detection rate at TPR-based thresholds
        plot_snr_detection_tpr(
            inj_df, active_families, thr_dict, tpr_title,
            OUTPUT_DIR / f'snr_detection_rate_{tpr_label}.png')

        # TP / FP cutout grids
        print(f"\n  Saving TP/FP cutout grids ({tpr_title})...")
        save_tp_fp_grids(df, active_families, thr_dict, tpr_label, OUTPUT_DIR)

    # ── SNR bin cutout directories (use 90% TP threshold) ────────────────────
    thr_90 = {fam: family_thresholds[fam][0.90] for fam in active_families}
    snr_bins = np.arange(SNR_MIN, SNR_MAX + 0.01, 1.0)
    print(f"\n  Saving SNR-bin cutout directories (90% TP thresholds)...")
    save_snr_bin_cutouts(inj_df, active_families, thr_90, snr_bins, OUTPUT_DIR)

    # ── Text summary ──────────────────────────────────────────────────────────
    recovery_rate = (inj_df['recovered'].sum() / max(len(inj_df), 1)) * 100
    cm_ens = confusion_at_05(y_true, ensemble_scores)
    TP = cm_ens[1,1]; FP = cm_ens[0,1]; FN = cm_ens[1,0]; TN = cm_ens[0,0]
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0

    lines = [
        '='*60,
        'INJECTION-RECOVERY EVALUATION SUMMARY',
        '='*60,
        f'  Test file(s)           : {len(fits_files)}',
        f'  Injections attempted   : {N_INJECTIONS} per file',
        f'  PSFs successfully inj. : {len(inj_df)}',
        f'  Peak finder recovery   : {inj_df["recovered"].sum()} / {len(inj_df)}'
        f'  ({recovery_rate:.1f}%)',
        f'  Total peaks classified : {len(df)}',
        f'    Positives (matched)  : {n_pos}',
        f'    Negatives            : {n_neg}',
        '',
        'ENSEMBLE (mean of all families)  — threshold 0.5',
        f'  TP={TP}  FP={FP}  FN={FN}  TN={TN}',
        f'  Precision : {prec:.3f}',
        f'  Recall    : {rec:.3f}',
        f'  F1        : {f1:.3f}',
        f'  AUROC     : {auc_vals.get("Ensemble", 0.0):.3f}',
        '',
        'PER-FAMILY RECALL @ 0.5',
    ]
    for fam, scores in zip(active_families, family_scores):
        cm = confusion_at_05(y_true, scores)
        r  = cm[1,1] / (cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
        p  = cm[1,1] / (cm[1,1]+cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0
        lines.append(f'  {fam:<22} Prec={p:.3f}  Rec={r:.3f}')

    for tpr_target in TARGET_TPRS:
        tpr_pct  = int(round(tpr_target * 100))
        lines.append('')
        lines.append(f'PER-FAMILY METRICS @ {tpr_pct}% TP recovery')
        thr_dict = {fam: family_thresholds[fam][tpr_target]
                    for fam in active_families}
        for fam, scores in zip(active_families, family_scores):
            thr = thr_dict[fam]
            cm  = confusion_at_threshold(y_true, scores, thr)
            r   = cm[1,1] / (cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
            p   = cm[1,1] / (cm[1,1]+cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0
            fpr = cm[0,1] / (cm[0,1]+cm[0,0]) if (cm[0,1]+cm[0,0]) > 0 else 0
            lines.append(f'  {fam:<22} thr={thr:.3f}  Prec={p:.3f}  Rec={r:.3f}  FPR={fpr:.3f}')

    lines += ['', '='*60]

    summary_path = OUTPUT_DIR / 'summary.txt'
    summary_path.write_text('\n'.join(lines))

    for line in lines:
        print(line)

    print(f"\nAll outputs → {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
