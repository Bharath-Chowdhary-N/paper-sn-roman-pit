#!/usr/bin/env python3
"""
Ensemble Testing Script for Transient Detection

Tests all 6 model families (DenseNet169, ResNeXt50, RegNetY016,
EfficientNetB0, ConvNeXtTiny, DeiTTiny) on FITS cutouts and saves
combined predictions with per-family and ensemble-mean probabilities.

Normalization: ZScale then min-max [0,1] — matches training_script.py exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from tqdm import tqdm
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Normalization — MUST match training_script.py exactly
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
    except Exception as e:
        print(f"Warning: ZScale failed, using min-max fallback: {e}")
        if data.max() > data.min():
            normalized = (data - data.min()) / (data.max() - data.min())
        else:
            normalized = np.zeros_like(data)
    return normalized.astype(np.float32)

# ============================================================================
# Model Architectures — MUST match training_script.py exactly
# ============================================================================

# --- DenseNet169 (custom from-scratch, block_config=(6,12,32,32)) -----------

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
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
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

# --- Generic timm wrapper (ResNeXt50, RegNetY016, EfficientNetB0, ConvNeXtTiny) ---

class TimmClassifier(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=False,
                 dropout=0.3, img_size=64):
        super().__init__()
        kwargs = {}
        if 'deit' in model_name or 'vit' in model_name:
            kwargs['img_size'] = img_size
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool='avg', **kwargs)
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return torch.sigmoid(self.classifier(self.backbone(x))).squeeze(1)

def create_resnext(num_classes=1, dropout=0.3, **kwargs):
    return TimmClassifier('resnext50_32x4d', num_classes=num_classes,
                          pretrained=False, dropout=dropout)

def create_regnety(num_classes=1, dropout=0.3, **kwargs):
    return TimmClassifier('regnety_016', num_classes=num_classes,
                          pretrained=False, dropout=dropout)

def create_efficientnet(num_classes=1, dropout=0.3, **kwargs):
    return TimmClassifier('efficientnet_b0', num_classes=num_classes,
                          pretrained=False, dropout=dropout)

def create_convnext(num_classes=1, dropout=0.3, **kwargs):
    return TimmClassifier('convnext_tiny', num_classes=num_classes,
                          pretrained=False, dropout=dropout)

# --- DeiT (Data-efficient Image Transformer) ---------------------------------

class DeiTClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        self.deit = timm.create_model(
            'deit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            img_size=64,
            global_pool='avg'
        )
        feature_dim = self.deit.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return torch.sigmoid(self.classifier(self.deit(x))).squeeze(1)

def create_deit(num_classes=1, **kwargs):
    return DeiTClassifier(num_classes=num_classes, pretrained=False)

# ============================================================================
# Ensemble config — mirrors training_script.py ENSEMBLE_CONFIG
# ============================================================================

ENSEMBLE_CONFIG = [
    ('DenseNet169',    create_densenet,    False),
    ('ResNeXt50',      create_resnext,     False),
    ('RegNetY016',     create_regnety,     False),
    ('EfficientNetB0', create_efficientnet,False),
    ('ConvNeXtTiny',   create_convnext,    False),
    ('DeiTTiny',       create_deit,        True),
]

# ============================================================================
# Image loading
# ============================================================================

def load_fits_image(fits_path):
    """Load FITS cutout, apply ZScale+minmax (matches training double z-scale pipeline)."""
    with fits.open(fits_path) as hdul:
        image_data = hdul[0].data.astype(np.float32)
        if image_data.ndim > 2:
            image_data = image_data[0] if image_data.ndim == 3 else image_data.squeeze()
        if image_data.shape != (64, 64):
            from skimage.transform import resize
            image_data = resize(image_data, (64, 64), mode='constant', anti_aliasing=True)
        image_data = normalize_with_zscale(image_data)
        raw_data   = image_data.copy()
        tensor = torch.from_numpy(image_data).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        return tensor, raw_data

# ============================================================================
# Model loading and inference
# ============================================================================

def load_family_models(type_name, create_fn, device):
    """Load all checkpoint files for one model family."""
    pattern = f"{type_name}_Ensemble_Model*_best.pth"
    model_files = sorted(Path(".").glob(pattern))
    models = []
    for path in model_files:
        try:
            model = create_fn(num_classes=1)
            ckpt  = torch.load(path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device).eval()
            models.append(model)
            print(f"    Loaded {path.name}")
        except Exception as e:
            print(f"    Failed to load {path.name}: {e}")
    return models

def family_mean_prob(models, image_tensor, device):
    """Return mean probability across all members of one family."""
    image_tensor = image_tensor.to(device)
    probs = []
    with torch.no_grad():
        for model in models:
            out = model(image_tensor)
            probs.append(out.item() if out.dim() == 0 else out[0].item())
    return float(np.mean(probs)), float(np.std(probs))

def save_png(image_data, output_path):
    img = Image.fromarray((image_data * 255).astype(np.uint8), mode='L')
    img.save(output_path)

# ============================================================================
# Main
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"ENSEMBLE TESTING — TRANSIENT DETECTION (6 model families)")
    print(f"{'='*70}\n")

    test_dir   = Path("output_peaks_from_test_directory")
    output_dir = Path("ML_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    positives_dir = output_dir / "positives"
    negatives_dir = output_dir / "negatives"
    positives_dir.mkdir(parents=True, exist_ok=True)
    negatives_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    if not test_dir.exists():
        print(f"\nError: test directory not found: {test_dir}")
        return

    fits_files = sorted(test_dir.glob("*.fits"))
    if not fits_files:
        print(f"\nError: no FITS files found in {test_dir}")
        return
    print(f"Found {len(fits_files)} FITS files\n")

    # -------------------------------------------------------------------------
    # Load all model families
    # -------------------------------------------------------------------------
    loaded_families = {}  # type_name -> list of models
    for type_name, create_fn, _ in ENSEMBLE_CONFIG:
        print(f"Loading {type_name}...")
        models = load_family_models(type_name, create_fn, device)
        if models:
            loaded_families[type_name] = models
            print(f"  -> {len(models)} member(s) loaded")
        else:
            print(f"  -> No checkpoints found, skipping")

    if not loaded_families:
        print("\nError: no model checkpoints found. Train models first.")
        return

    active_families = [name for name, _, _ in ENSEMBLE_CONFIG if name in loaded_families]
    print(f"\nActive families: {active_families}\n")

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    print(f"{'='*70}")
    print(f"Running inference...")
    print(f"{'='*70}\n")

    results = []
    pos_count = neg_count = 0

    for fits_file in tqdm(fits_files, desc="Classifying"):
        try:
            image_tensor, raw_data = load_fits_image(fits_file)
        except Exception as e:
            print(f"\nSkipping {fits_file.name}: {e}")
            continue

        row = {'filename': fits_file.name, 'filepath': str(fits_file)}

        family_means = []
        for type_name in active_families:
            mean_p, std_p = family_mean_prob(loaded_families[type_name], image_tensor, device)
            row[f'{type_name}_mean_prob'] = mean_p
            row[f'{type_name}_std_prob']  = std_p
            family_means.append(mean_p)

        ensemble_mean = float(np.mean(family_means))
        row['ensemble_mean_prob'] = ensemble_mean
        row['confidence']         = max(ensemble_mean, 1 - ensemble_mean)
        row['prediction']         = 'transient' if ensemble_mean >= 0.5 else 'non-transient'
        row['num_families']       = len(active_families)
        results.append(row)

        # Save PNG under majority vote label
        if row['prediction'] == 'transient':
            save_png(raw_data, positives_dir / f"pos_{pos_count:04d}_{fits_file.stem}.png")
            pos_count += 1
        else:
            save_png(raw_data, negatives_dir / f"neg_{neg_count:04d}_{fits_file.stem}.png")
            neg_count += 1

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = output_dir / "ensemble_predictions.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Total cutouts processed : {len(df)}")
    print(f"  Predicted transients    : {(df['prediction']=='transient').sum()}")
    print(f"  Predicted non-transients: {(df['prediction']=='non-transient').sum()}")
    print(f"  Mean ensemble prob      : {df['ensemble_mean_prob'].mean():.3f}")
    print(f"  Mean confidence         : {df['confidence'].mean():.3f}")
    print(f"\n  Per-family mean probabilities:")
    for type_name in active_families:
        col = f'{type_name}_mean_prob'
        print(f"    {type_name:<22} {df[col].mean():.3f}")
    print(f"\n  Saved: {csv_path}")
    print(f"  PNGs : {positives_dir} ({pos_count} pos)  {negatives_dir} ({neg_count} neg)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
