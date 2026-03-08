import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import splitfolders
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import torch.nn.functional as F
import timm

# ============================================================================
# Normalization: ZScale first, then min-max to [0, 1]
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
# DenseNet Architecture (custom, from-scratch)
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

# ============================================================================
# Generic timm Wrapper (ResNeXt50, RegNetY016, EfficientNetB0, ConvNeXtTiny)
# ============================================================================

class TimmClassifier(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True,
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
                          pretrained=True, dropout=dropout)

def create_regnety(num_classes=1, dropout=0.3, **kwargs):
    return TimmClassifier('regnety_016', num_classes=num_classes,
                          pretrained=True, dropout=dropout)

def create_efficientnet(num_classes=1, dropout=0.3, **kwargs):
    return TimmClassifier('efficientnet_b0', num_classes=num_classes,
                          pretrained=True, dropout=dropout)

def create_convnext(num_classes=1, dropout=0.3, **kwargs):
    return TimmClassifier('convnext_tiny', num_classes=num_classes,
                          pretrained=True, dropout=dropout)

# ============================================================================
# DeiT (Data-efficient Image Transformer)
# ============================================================================

class DeiTClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
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
    return DeiTClassifier(num_classes=num_classes, pretrained=True)

# ============================================================================
# Ensemble Config: 6 families x 4 members = 24 total models
# (factory_fn, display_name, is_transformer)
# ============================================================================

ENSEMBLE_CONFIG = [
    #(create_densenet,     'DenseNet169',    False),
    #(create_resnext,      'ResNeXt50',      False),
    #(create_regnety,      'RegNetY016',     False),
    #(create_efficientnet, 'EfficientNetB0', False),
    (create_convnext,     'ConvNeXtTiny',   False)
    #(create_deit,         'DeiTTiny',       True),
]

MODELS_PER_TYPE   = 4
TRAIN_EPOCHS      = 5   # full training epochs for CNNs
DEIT_TRAIN_EPOCHS = 10   # DeiT needs more epochs

# Fixed hyperparameters for each model family (no Optuna tuning)
FIXED_HPS = {
    'DenseNet169': {
        'lr': 1e-4, 'weight_decay': 1e-4, 'dropout': 0.3, 'drop_rate': 0.1,
        'optimizer': 'adamw', 'momentum': 0.9, 'batch_size': 32, 'scheduler': 'cosine',
    },
    'ResNeXt50': {
        'lr': 1e-4, 'weight_decay': 1e-4, 'dropout': 0.3,
        'optimizer': 'adamw', 'momentum': 0.9, 'batch_size': 32, 'scheduler': 'cosine',
    },
    'RegNetY016': {
        'lr': 1e-4, 'weight_decay': 1e-4, 'dropout': 0.3,
        'optimizer': 'adamw', 'momentum': 0.9, 'batch_size': 32, 'scheduler': 'cosine',
    },
    'EfficientNetB0': {
        'lr': 1e-4, 'weight_decay': 1e-4, 'dropout': 0.3,
        'optimizer': 'adamw', 'momentum': 0.9, 'batch_size': 32, 'scheduler': 'cosine',
    },
    'ConvNeXtTiny': {
        'lr': 1e-5, 'weight_decay': 1e-4, 'dropout': 0.3,
        'optimizer': 'adamw', 'momentum': 0.9, 'batch_size': 32, 'scheduler': 'cosine',
    },
    'DeiTTiny': {
        'lr': 1e-4, 'weight_decay': 1e-4, 'dropout': 0.3,
        'optimizer': 'adamw', 'momentum': 0.9, 'batch_size': 16, 'scheduler': 'cosine',
    },
}

# ============================================================================
# FITS Dataset — ZScale then min-max [0,1], no augmentations
# ============================================================================

class FITSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = []
        self.labels = []
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.endswith('.fits'):
                        self.files.append(os.path.join(class_dir, fname))
                        self.labels.append(class_idx)
        print(f"Loading {len(self.files)} files from {root_dir} into RAM...")
        self._cache = [self._load(i) for i in range(len(self.files))]
        print(f"  Done — {len(self.files)} files cached.")

    def _load(self, idx):
        try:
            with fits.open(self.files[idx]) as hdul:
                image_data = hdul[0].data.astype(np.float32)
                if image_data.ndim > 2:
                    image_data = image_data[0] if image_data.ndim == 3 else image_data.squeeze()
                if image_data.shape != (64, 64):
                    from skimage.transform import resize
                    image_data = resize(image_data, (64, 64), mode='constant', anti_aliasing=True)
                image_data = normalize_with_zscale(image_data)
                return torch.from_numpy(image_data).float().unsqueeze(0).repeat(3, 1, 1)
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros((3, 64, 64))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self._cache[idx], self.labels[idx]

def build_optimizer(model, hp):
    name = hp['optimizer']
    lr, wd = hp['lr'], hp['weight_decay']
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                         momentum=hp['momentum'], nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, hp, num_epochs=None):
    if num_epochs is None:
        num_epochs = TRAIN_EPOCHS
    name = hp['scheduler']
    if name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    elif name == 'cosine':
        T_0 = max(10, num_epochs // 4)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2)
    else:
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)

# ============================================================================
# Optuna Objective
# ============================================================================

# ============================================================================
# Training Utilities
# ============================================================================

class TrainingProgress:
    def __init__(self, model_name):
        self.train_losses, self.train_accs = [], []
        self.val_losses,   self.val_accs   = [], []
        self.model_name = model_name

    def update(self, tl, ta, vl, va):
        self.train_losses.append(tl); self.train_accs.append(ta)
        self.val_losses.append(vl);   self.val_accs.append(va)

    def plot_progress(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.val_losses,   'r-', label='Val Loss',   linewidth=2)
        plt.title(f'{self.model_name} - Loss'); plt.xlabel('Epochs'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
        plt.plot(epochs, self.val_accs,   'r-', label='Val Acc',   linewidth=2)
        plt.title(f'{self.model_name} - Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_progress.png', dpi=300, bbox_inches='tight')
        plt.close()


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs.view(-1), labels.view(-1)).item()
            correct  += ((outputs.view(-1) >= 0.5).float() == labels.view(-1)).sum().item()
            total    += labels.size(0)
    return val_loss / len(val_loader), 100 * correct / total


def train_single_model(model, model_name, train_loader, val_loader,
                       criterion, optimizer, scheduler, scheduler_name,
                       device, num_epochs, patience=5):
    progress = TrainingProgress(model_name)
    best_val_acc, patience_counter = 0.0, 0
    model_file = f'{model_name}_best.pth'
    if os.path.exists(model_file):
        os.remove(model_file)

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.view(-1))
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            predicted = (outputs.view(-1) >= 0.5).float()
            total   += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

        train_loss = running_loss / len(train_loader)
        train_acc  = 100 * correct / total
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler_name == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        progress.update(train_loss, train_acc, val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc}, model_file)
            print(f"  Saved {model_name} — val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1

        progress.plot_progress()
        print(f'\n  {model_name} | Epoch {epoch+1}/{num_epochs}')
        print(f'    Train -> Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%')
        print(f'    Val   -> Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%')
        print(f'    LR: {optimizer.param_groups[0]["lr"]:.7f}')
        print('  ' + '-' * 58)

        if patience_counter >= patience:
            print(f"  Early stopping for {model_name} at epoch {epoch+1}")
            break

    return best_val_acc


def plot_sample_images(loader):
    pos = neg = None
    for data, labels in loader:
        for i in range(len(labels)):
            if labels[i] == 1 and pos is None: pos = data[i]
            elif labels[i] == 0 and neg is None: neg = data[i]
            if pos is not None and neg is not None: break
        if pos is not None and neg is not None: break
    if pos is None or neg is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(pos[0].cpu().numpy(), cmap='viridis')
    ax1.set_title('Positive (with PSF)\n[ZScale -> MinMax]', fontsize=14, fontweight='bold')
    ax1.axis('off'); plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(neg[0].cpu().numpy(), cmap='viridis')
    ax2.set_title('Negative (non-peak)\n[ZScale -> MinMax]', fontsize=14, fontweight='bold')
    ax2.axis('off'); plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig('sample_images_64x64_zscale.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Sample images saved as 'sample_images_64x64_zscale.png'")

# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_models = len(ENSEMBLE_CONFIG) * MODELS_PER_TYPE

    print(f"\n{'='*70}")
    print(f"ENSEMBLE TRAINING (FIXED HYPERPARAMETERS)")
    print(f"  {len(ENSEMBLE_CONFIG)} families x {MODELS_PER_TYPE} members = {total_models} total models")
    print(f"{'='*70}")
    print(f"  Device        : {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"  Normalization : ZScale then min-max [0, 1]")
    print(f"  Augmentation  : None")
    print(f"  Train epochs  : {TRAIN_EPOCHS} (CNN)  /  {DEIT_TRAIN_EPOCHS} (DeiT)")
    print(f"{'='*70}\n")

    input_folder  = 'pos_and_neg/'
    output_folder = 'split_folders/'

    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' not found!"); return

    print("Splitting dataset...")
    if not os.path.exists(output_folder):
        splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.1, 0.1))

    train_dataset = FITSDataset(root_dir=f"{output_folder}/train")
    val_dataset   = FITSDataset(root_dir=f"{output_folder}/val")
    test_dataset  = FITSDataset(root_dir=f"{output_folder}/test")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    val_loader  = DataLoader(val_dataset,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    plot_sample_images(DataLoader(train_dataset, batch_size=16, shuffle=True))

    # =========================================================================
    # Fixed Hyperparameters
    # =========================================================================
    best_hps = {type_name: FIXED_HPS[type_name].copy() for _, type_name, _ in ENSEMBLE_CONFIG}

    print(f"  {'Model':<22} {'lr':>9} {'wd':>9} {'dropout':>8} {'opt':>6} {'sched':>8} {'bs':>4}")
    print('  ' + '-' * 68)
    for _, type_name, _ in ENSEMBLE_CONFIG:
        hp = best_hps[type_name]
        print(f"  {type_name:<22} {hp['lr']:>9.2e} {hp['weight_decay']:>9.2e} "
              f"{hp['dropout']:>8.2f} {hp['optimizer']:>6} {hp['scheduler']:>8} {hp['batch_size']:>4}")
    print()

    # =========================================================================
    # Ensemble Training
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ENSEMBLE TRAINING ({MODELS_PER_TYPE} MEMBERS x {len(ENSEMBLE_CONFIG)} TYPES)")
    print(f"{'='*70}")

    results = []

    for create_fn, type_name, is_transformer in ENSEMBLE_CONFIG:
        hp = best_hps[type_name]
        num_epochs = DEIT_TRAIN_EPOCHS if is_transformer else TRAIN_EPOCHS
        patience   = 10 if is_transformer else 5

        for member_idx in range(1, MODELS_PER_TYPE + 1):
            model_name = f"{type_name}_Ensemble_Model{member_idx}"
            print(f"\n{'='*70}")
            print(f"  {model_name}  [{member_idx}/{MODELS_PER_TYPE}]")
            print(f"  lr={hp['lr']:.2e}  wd={hp['weight_decay']:.2e}  "
                  f"dropout={hp.get('dropout', 0.3):.2f}  "
                  f"opt={hp['optimizer']}  sched={hp['scheduler']}  bs={hp['batch_size']}")
            print(f"{'='*70}\n")

            mkwargs = {'num_classes': 1}
            if not is_transformer:
                mkwargs['dropout'] = hp.get('dropout', 0.3)
                if type_name == 'DenseNet169':
                    mkwargs['drop_rate'] = hp.get('drop_rate', 0.0)
            model = create_fn(**mkwargs).to(device)

            train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'],
                                      shuffle=True, num_workers=4, pin_memory=True)
            criterion = nn.BCELoss()
            optimizer = build_optimizer(model, hp)
            scheduler = build_scheduler(optimizer, hp, num_epochs=num_epochs)

            best_val_acc = train_single_model(
                model, model_name, train_loader, val_loader,
                criterion, optimizer, scheduler, hp['scheduler'],
                device, num_epochs, patience=patience
            )
            results.append((type_name, member_idx, best_val_acc))
            print(f"\n  {model_name} done — Best Val Acc: {best_val_acc:.2f}%")

            del model, optimizer, scheduler
            torch.cuda.empty_cache()

    # Summary
    all_accs = [acc for _, _, acc in results]
    print(f"\n{'='*70}")
    print(f"ALL DONE — {total_models} MODELS TRAINED")
    print(f"{'='*70}")
    print(f"\n  {'Model':<30} {'Member':>6} {'Val Acc':>10}")
    print('  ' + '-' * 50)
    for type_name, member_idx, acc in results:
        print(f"  {type_name:<30} {member_idx:>6} {acc:>9.2f}%")
    print(f"\n  Average : {np.mean(all_accs):.2f}%  |  Best: {max(all_accs):.2f}%  |  Std: {np.std(all_accs):.2f}%")
    print(f"\n  Per-type averages:")
    for _, type_name, _ in ENSEMBLE_CONFIG:
        type_accs = [acc for tn, _, acc in results if tn == type_name]
        print(f"    {type_name:<22} avg={np.mean(type_accs):.2f}%  [{', '.join(f'{a:.1f}' for a in type_accs)}]")
    print(f"\n  Best hyperparameters found:")
    for _, type_name, _ in ENSEMBLE_CONFIG:
        hp = best_hps[type_name]
        print(f"    {type_name}: lr={hp['lr']:.2e}, opt={hp['optimizer']}, "
              f"sched={hp['scheduler']}, bs={hp['batch_size']}, "
              f"dropout={hp.get('dropout', 0.3):.2f}, wd={hp['weight_decay']:.2e}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
