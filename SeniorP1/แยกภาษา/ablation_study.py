"""
ablation_study.py
=================
เปรียบเทียบผลกระทบของ augmentation ต่อ accuracy
Conditions:
  A. No augmentation
  B. SpecAugment only (freq + time mask)
  C. SpecAugment + noise
  D. SpecAugment + noise + mixup

Usage:
  python ablation_study.py --lang Thai
  python ablation_study.py --lang all
"""

import os
import sys
import argparse
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

BASE_DIR = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
FEAT_DIR = os.path.join(BASE_DIR, "extracted_features")
OUT_DIR  = os.path.join(BASE_DIR, "results", "ablation")
os.makedirs(OUT_DIR, exist_ok=True)

LANGUAGES  = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']
N_MELS     = 128
MAX_T      = 130
EPOCHS     = 50
BATCH      = 64
LR         = 3e-4


def log(msg):
    print(f"  {msg}")


# ── Model (same as pipeline) ──────────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.block(x))

class EmotionResNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.layer1 = nn.Sequential(ResBlock(32),  nn.Conv2d(32,64,3,stride=2,padding=1,bias=False),  nn.BatchNorm2d(64),  nn.ReLU(True))
        self.layer2 = nn.Sequential(ResBlock(64),  nn.Conv2d(64,128,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.layer3 = nn.Sequential(ResBlock(128), nn.Conv2d(128,256,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(256), nn.ReLU(True))
        self.pool   = nn.AdaptiveAvgPool2d((4,4))
        self.fc     = nn.Sequential(nn.Flatten(), nn.Linear(256*16,256), nn.ReLU(True), nn.Dropout(0.4), nn.Linear(256,n_classes))
    def forward(self, x):
        return self.fc(self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))))


# ── Dataset with configurable augmentation ────
class MelDataset(Dataset):
    def __init__(self, X, y, aug_config):
        self.X   = X
        self.y   = y
        self.aug = aug_config  # dict

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.aug.get('spec_augment'):
            # Freq mask
            f0 = np.random.randint(0, max(1, N_MELS - 20))
            x[f0:f0 + np.random.randint(1, 20), :] = x.min()
            # Time mask
            t0 = np.random.randint(0, max(1, MAX_T - 25))
            x[:, t0:t0 + np.random.randint(1, 25)] = x.min()
        if self.aug.get('noise'):
            x += np.random.normal(0, 0.01, x.shape).astype(np.float32)
        return torch.tensor(x[np.newaxis], dtype=torch.float32), int(self.y[idx])


def load_data(lang):
    d = os.path.join(FEAT_DIR, lang)
    if not os.path.exists(d):
        return None, None, None
    files = [f for f in os.listdir(d) if f.endswith('.npy')]
    emotions = sorted({f.split('_')[0] for f in files})
    le = {e: i for i, e in enumerate(emotions)}
    X, y = [], []
    for fname in files:
        arr = np.load(os.path.join(d, fname))
        if arr.shape[1] >= MAX_T:
            arr = arr[:, :MAX_T]
        else:
            arr = np.pad(arr, ((0,0),(0,MAX_T-arr.shape[1])), mode='edge')
        X.append(arr)
        y.append(le[fname.split('_')[0]])
    return np.array(X, dtype=np.float32), np.array(y), emotions


def train_eval(X_tr, y_tr, X_val, y_val, emotions, aug_config, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(emotions)

    class_counts = np.bincount(y_tr)
    weights  = 1.0 / class_counts[y_tr]
    sampler  = WeightedRandomSampler(weights, len(weights))
    tr_load  = DataLoader(MelDataset(X_tr, y_tr, aug_config), BATCH, sampler=sampler)
    val_load = DataLoader(MelDataset(X_val, y_val, {}),        BATCH, shuffle=False)

    model    = EmotionResNet(n_classes).to(device)
    cls_w    = torch.tensor(1.0/(class_counts+1e-6), dtype=torch.float32).to(device)
    cls_w    = cls_w / cls_w.sum() * n_classes
    crit     = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.1)
    optim    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    scaler   = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_acc, best_ep = 0.0, 0
    hist = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_load:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                crit(model(xb), yb).backward()
                optim.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_load:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total   += len(yb)
        acc = correct / total
        hist.append(acc)
        if acc > best_acc:
            best_acc, best_ep = acc, ep

    return best_acc * 100, hist


def run_ablation(lang):
    from sklearn.model_selection import train_test_split
    log(f"\n{'='*50}")
    log(f"Ablation Study: {lang}")
    log(f"{'='*50}")

    X, y, emotions = load_data(lang)
    if X is None:
        return {}
    log(f"  {len(X)} samples | {len(emotions)} classes")

    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.15, stratify=y, random_state=999)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    conditions = {
        'A — No Augment':        {},
        'B — SpecAugment':       {'spec_augment': True},
        'C — Spec + Noise':      {'spec_augment': True, 'noise': True},
    }

    results = {}
    for name, cfg in conditions.items():
        log(f"\n  [{name}]")
        acc, hist = train_eval(X_tr, y_tr, X_te, y_te, emotions, cfg)
        log(f"  → Val Acc: {acc:.2f}%")
        results[name] = {'acc': acc, 'hist': hist}

    return results


def plot_ablation(all_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    langs = [l for l in LANGUAGES if l in all_results]
    conditions = ['A — No Augment', 'B — SpecAugment', 'C — Spec + Noise']
    colors     = ['#95A5A6', '#3498DB', '#E74C3C']

    # ── Bar chart per language ──
    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(langs)), 7))
    x = np.arange(len(langs))
    w = 0.25

    for i, (cond, color) in enumerate(zip(conditions, colors)):
        accs = [all_results[l].get(cond, {}).get('acc', 0) for l in langs]
        bars = ax.bar(x + (i-1)*w, accs, w, label=cond.split(' — ')[1],
                      color=color, edgecolor='white', alpha=0.9)
        for bar, v in zip(bars, accs):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=12)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Ablation Study — Effect of Augmentation on Accuracy',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out_path = os.path.join(OUT_DIR, 'ablation_bar.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    log(f"บันทึก {out_path}")

    # ── Training curves per language ──
    for lang in langs:
        if lang not in all_results:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for cond, color in zip(conditions, colors):
            hist = all_results[lang].get(cond, {}).get('hist', [])
            if hist:
                ax.plot(range(1, len(hist)+1), [h*100 for h in hist],
                        label=cond.split(' — ')[1], color=color, linewidth=1.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Accuracy (%)')
        ax.set_title(f'{lang} — Ablation Training Curves', fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        out_path = os.path.join(OUT_DIR, f'ablation_curve_{lang.lower()}.png')
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        log(f"บันทึก {out_path}")


def write_report(all_results):
    lines = ["="*60, "  ABLATION STUDY REPORT", "="*60]
    lines.append(f"\n{'Language':<12} {'No Aug':>10} {'SpecAug':>10} {'Spec+Noise':>12}")
    lines.append("-"*50)
    for lang in LANGUAGES:
        if lang not in all_results:
            continue
        r = all_results[lang]
        a = r.get('A — No Augment',    {}).get('acc', 0)
        b = r.get('B — SpecAugment',   {}).get('acc', 0)
        c = r.get('C — Spec + Noise',  {}).get('acc', 0)
        gain = max(b,c) - a
        lines.append(f"{lang:<12} {a:>9.2f}% {b:>9.2f}% {c:>11.2f}%  (gain={gain:+.2f}%)")

    out_path = os.path.join(OUT_DIR, 'ablation_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"บันทึก {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='all')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  ABLATION STUDY — EFFECT OF AUGMENTATION")
    print("="*60)
    device_str = 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'
    print(f"  Device: {device_str}")

    langs = LANGUAGES if args.lang == 'all' else [args.lang]
    all_results = {}
    for lang in langs:
        res = run_ablation(lang)
        if res:
            all_results[lang] = res

    plot_ablation(all_results)
    write_report(all_results)

    print("\n" + "="*60)
    print(f"  ผลลัพธ์บันทึกที่: {OUT_DIR}")
