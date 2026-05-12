"""
ensemble_model.py
=================
Train N models ด้วย random seed ต่างกัน แล้วทำ soft-voting ensemble
เพิ่ม accuracy โดยไม่ต้องเพิ่ม data

Usage:
  python ensemble_model.py --lang Thai       # train 3-seed ensemble สำหรับ Thai
  python ensemble_model.py --lang all        # ทุกภาษา
  python ensemble_model.py --lang Thai --eval # train + evaluate + save report

ผลลัพธ์: models/{lang}_ensemble_seed{N}.pt (N = 0,1,2)
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

BASE_DIR  = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
FEAT_DIR  = os.path.join(BASE_DIR, "extracted_features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)

LANGUAGES = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']
N_SEEDS   = 3
N_MELS    = 128
MAX_T     = 130
EPOCHS    = 60
BATCH     = 64
LR        = 3e-4


def log(msg):
    print(f"  {msg}")


# ──────────────────────────────────────────────
# MODEL  (ต้องตรงกับ data_pipeline.py)
# ──────────────────────────────────────────────
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
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.fc(self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))))


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            # Freq mask
            f0 = np.random.randint(0, max(1, N_MELS - 20))
            x[f0:f0 + np.random.randint(1, 20), :] = x.min()
            # Time mask
            t0 = np.random.randint(0, max(1, MAX_T - 25))
            x[:, t0:t0 + np.random.randint(1, 25)] = x.min()
        return torch.tensor(x[np.newaxis], dtype=torch.float32), int(self.y[idx])


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
def load_lang_data(lang):
    d = os.path.join(FEAT_DIR, lang)
    if not os.path.exists(d):
        return None, None, None

    files = [f for f in os.listdir(d) if f.endswith('.npy')]
    emotions = sorted({f.split('_')[0] for f in files})
    le = {e: i for i, e in enumerate(emotions)}

    X, y = [], []
    for fname in files:
        emo = fname.split('_')[0]
        arr = np.load(os.path.join(d, fname))
        if arr.shape[1] >= MAX_T:
            arr = arr[:, :MAX_T]
        else:
            arr = np.pad(arr, ((0, 0), (0, MAX_T - arr.shape[1])), mode='edge')
        X.append(arr)
        y.append(le[emo])

    return np.array(X, dtype=np.float32), np.array(y), emotions


# ──────────────────────────────────────────────
# TRAIN ONE SEED
# ──────────────────────────────────────────────
def train_one(lang, X, y, emotions, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(emotions)

    # Split
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    tr_idx, val_idx = train_test_split(idx, test_size=0.15, stratify=y,
                                        random_state=seed)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Weighted sampler
    class_counts = np.bincount(y_tr)
    weights = 1.0 / class_counts[y_tr]
    sampler = WeightedRandomSampler(weights, len(weights))

    tr_loader  = DataLoader(MelDataset(X_tr, y_tr, augment=True),
                             batch_size=BATCH, sampler=sampler)
    val_loader = DataLoader(MelDataset(X_val, y_val, augment=False),
                             batch_size=BATCH, shuffle=False)

    model = EmotionResNet(n_classes).to(device)
    cls_w = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(device)
    cls_w = cls_w / cls_w.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_acc, best_state = 0.0, None
    patience, no_improve = 12, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                criterion(model(xb), yb).backward()
                optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total   += len(yb)
        val_acc = correct / total

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if epoch % 10 == 0:
            log(f"    Seed {seed} | Epoch {epoch:3d}/{EPOCHS} | val_acc={val_acc:.4f} (best={best_acc:.4f})")

    return best_state, best_acc


# ──────────────────────────────────────────────
# ENSEMBLE EVALUATE
# ──────────────────────────────────────────────
def evaluate_ensemble(lang, emotions, states, X_test, y_test):
    from sklearn.metrics import accuracy_score, f1_score

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(emotions)

    models = []
    for state in states:
        m = EmotionResNet(n_classes).to(device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    loader = DataLoader(MelDataset(X_test, y_test, augment=False),
                        batch_size=128, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            avg_prob = torch.zeros(xb.shape[0], n_classes, device=device)
            for m in models:
                avg_prob += torch.softmax(m(xb), dim=1)
            avg_prob /= len(models)
            all_probs.append(avg_prob.cpu().numpy())

    all_probs = np.vstack(all_probs)
    preds     = all_probs.argmax(1)
    acc  = accuracy_score(y_test, preds) * 100
    f1   = f1_score(y_test, preds, average='weighted') * 100
    return acc, f1


# ──────────────────────────────────────────────
# MAIN TRAIN FUNCTION
# ──────────────────────────────────────────────
def train_ensemble(lang, do_eval=False):
    log(f"\n{'='*50}")
    log(f"Ensemble Training: {lang}  ({N_SEEDS} seeds)")
    log(f"{'='*50}")

    X, y, emotions = load_lang_data(lang)
    if X is None:
        log(f"  ไม่พบข้อมูล {lang}")
        return

    log(f"  ข้อมูล: {len(X)} ตัวอย่าง | {len(emotions)} classes: {emotions}")

    # Split test set (fixed seed=999 เพื่อให้ตรงกับ data_pipeline)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.15, stratify=y, random_state=999)
    X_test, y_test = X[te_idx], y[te_idx]

    states = []
    seed_accs = []
    for seed in range(N_SEEDS):
        log(f"\n  Training seed {seed} ...")
        state, acc = train_one(lang, X[tr_idx], y[tr_idx], emotions, seed=seed * 100 + 42)
        states.append(state)
        seed_accs.append(acc * 100)
        log(f"  Seed {seed} best val_acc: {acc*100:.2f}%")

        # บันทึก individual model
        out_path = os.path.join(MODEL_DIR, f"{lang}_ensemble_seed{seed}.pt")
        torch.save({'state_dict': state, 'n_classes': len(emotions),
                    'emotions': emotions, 'seed': seed}, out_path)

    # บันทึก label encoder ถ้ายังไม่มี
    enc_path = os.path.join(MODEL_DIR, f"{lang}_label_encoder.pkl")
    if not os.path.exists(enc_path):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(emotions)
        with open(enc_path, 'wb') as f:
            pickle.dump(le, f)

    # Ensemble evaluation
    if do_eval:
        ens_acc, ens_f1 = evaluate_ensemble(lang, emotions, states, X_test, y_test)
        avg_single = np.mean(seed_accs)
        log(f"\n  Individual seeds avg: {avg_single:.2f}%")
        log(f"  Ensemble (soft-vote): {ens_acc:.2f}%  F1={ens_f1:.2f}%")
        gain = ens_acc - avg_single
        log(f"  Ensemble gain: +{gain:.2f}%")
        return ens_acc, seed_accs
    return None, seed_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='all',
                        help='ภาษา: Chinese/Japan/Korean/Thai/English/all')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate ensemble after training')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ENSEMBLE MODEL TRAINING (Soft-Voting, 3 Seeds)")
    print("=" * 60)

    device_str = 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'
    print(f"  Device: {device_str}")

    langs = LANGUAGES if args.lang == 'all' else [args.lang]
    results = {}

    for lang in langs:
        ens_acc, seed_accs = train_ensemble(lang, do_eval=args.eval)
        results[lang] = {'ensemble': ens_acc, 'seeds': seed_accs}

    print("\n" + "=" * 60)
    print("  ENSEMBLE RESULTS SUMMARY")
    print("=" * 60)
    for lang, r in results.items():
        seeds_str = ', '.join(f'{a:.1f}%' for a in r['seeds'])
        ens_str   = f"{r['ensemble']:.2f}%" if r['ensemble'] else 'N/A (run with --eval)'
        print(f"  {lang:<10} seeds=[{seeds_str}]  ensemble={ens_str}")

    print(f"\n  Models บันทึกที่: {MODEL_DIR}")
    print("  ใช้งาน ensemble ใน demo: แก้ demo_live.py ให้โหลด 3 models แล้ว average probs")
