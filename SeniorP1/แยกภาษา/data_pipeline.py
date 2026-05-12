"""
Data Pipeline: Speech Emotion Recognition — Per-Language Models
================================================================
Strategy : แยก Train โมเดลทีละภาษา → เก็บเป็น 5 โมเดล
           แล้วนำมารวมกัน (Ensemble / Router) ขั้นต่อไป

ภาษาที่รองรับ : Chinese, Japanese, Korean, Thai, English
อารมณ์ที่แยก : Angry, Happy, Sad, Neutral, Surprise, Fear

การใช้งาน:
    python data_pipeline.py                          # รันทุก stage ทุกภาษา
    python data_pipeline.py --stage verify           # ตรวจสอบ dataset
    python data_pipeline.py --stage extract          # สกัด feature ทุกภาษา
    python data_pipeline.py --stage train            # train ทุกภาษา
    python data_pipeline.py --stage eval             # evaluate ทุกภาษา
    python data_pipeline.py --stage train --lang Thai   # train แค่ภาษาเดียว
    python data_pipeline.py --stage eval  --lang English
"""

import os
import sys
import argparse
import time
import contextlib
import numpy as np
import librosa
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# แก้ปัญหา OpenMP duplicate library (TF + PyTorch อยู่ใน env เดียวกัน)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ⚙️ CONFIG
# ==========================================
BASE_DIR    = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FEATURE_DIR = os.path.join(BASE_DIR, "extracted_features")   # แยก subfolder ต่อภาษา
MODEL_DIR   = os.path.join(BASE_DIR, "models")               # แยกไฟล์ต่อภาษา
RESULT_DIR  = os.path.join(BASE_DIR, "results")

# ชื่อโฟลเดอร์ใน dataset/ ต้องตรงกับนี้
TARGET_LANGUAGES = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']

EMOTION_KEYWORDS = {
    'angry'  : 'Angry',   'anger'  : 'Angry',   'ag'   : 'Angry',
    'happy'  : 'Happy',   'joy'    : 'Happy',    'hap'  : 'Happy',
    'sad'    : 'Sad',     'sadness': 'Sad',
    'neutral': 'Neutral', 'neu'    : 'Neutral',
    'surprise':'Surprise','sur'    : 'Surprise',
    'fear'   : 'Fear',    'fearful': 'Fear',
    'disgust': 'Disgust',
}
# RAVDESS ใช้ตัวเลขระบุอารมณ์ใน English dataset
RAVDESS_MAP = {
    '01': 'Neutral', '02': 'Neutral',
    '03': 'Happy',   '04': 'Sad',
    '05': 'Angry',   '06': 'Fear',
    '07': 'Disgust', '08': 'Surprise',
}

SAMPLE_RATE    = 16000
N_MELS         = 128
MAX_TIME_STEPS = 130
BATCH_SIZE     = 32
EPOCHS         = 100

# ==========================================
# 🔧 UTILITIES
# ==========================================
def log(msg, level="INFO"):
    icons = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "ERROR": "❌", "RUN": "🚀", "LANG": "🌐"}
    print(f"{icons.get(level, '   ')} {msg}")

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def feature_dir_for(lang):
    """extracted_features/Thai/, extracted_features/English/, ..."""
    return os.path.join(FEATURE_DIR, lang)

def model_path_for(lang):
    return os.path.join(MODEL_DIR, f"{lang}_model.pt")

def encoder_path_for(lang):
    return os.path.join(MODEL_DIR, f"{lang}_label_encoder.pkl")

def result_dir_for(lang):
    return os.path.join(RESULT_DIR, lang)

# ==========================================
# STAGE 1 — VERIFY DATASET
# ==========================================
def stage_verify():
    print("\n" + "="*60)
    print("  STAGE 1: DATASET VERIFICATION")
    print("="*60)

    if not os.path.exists(DATASET_DIR):
        log(f"ไม่พบ dataset folder: {DATASET_DIR}", "ERROR")
        return False

    total = 0
    for lang in TARGET_LANGUAGES:
        lang_path = os.path.join(DATASET_DIR, lang)
        if not os.path.exists(lang_path):
            log(f"{lang:10s}: ❌ ไม่พบโฟลเดอร์", "WARN")
            continue

        count = sum(
            1 for _, _, files in os.walk(lang_path)
            for f in files if f.lower().endswith(('.wav', '.flac', '.mp3'))
        )
        total += count
        status = "OK" if count > 0 else "WARN"
        log(f"{lang:10s}: {count:6,d} ไฟล์", status)

    log(f"รวมทั้งหมด: {total:,} ไฟล์", "OK")
    return total > 0

# ==========================================
# STAGE 2 — FEATURE EXTRACTION (per language)
# ==========================================
def extract_mel(file_path):
    """แปลงไฟล์เสียงเป็น Mel-Spectrogram shape (N_MELS, MAX_TIME_STEPS)"""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mel   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < MAX_TIME_STEPS:
        pad = MAX_TIME_STEPS - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_TIME_STEPS]

    return mel_db  # (128, 130)


def get_label_from_path(file_path, lang):
    """
    ดึง emotion label จากชื่อไฟล์ / โฟลเดอร์
    - English RAVDESS: ดูจากตัวเลขในชื่อไฟล์ (03-01-01-01-01-01-01.wav)
    - อื่นๆ: ดูจาก keyword ใน path
    """
    filename  = os.path.basename(file_path)
    path_lower = file_path.lower().replace('\\', '/')

    # RAVDESS (English)
    if lang == 'English':
        parts = os.path.splitext(filename)[0].split('-')
        if len(parts) >= 3:
            label = RAVDESS_MAP.get(parts[2])
            if label:
                return label

    # Keyword matching
    for kw, em in EMOTION_KEYWORDS.items():
        if kw in path_lower:
            return em

    return None


def stage_extract(langs=None):
    print("\n" + "="*60)
    print("  STAGE 2: FEATURE EXTRACTION  (แยกต่อภาษา)")
    print("="*60)

    langs = langs or TARGET_LANGUAGES
    grand_total = 0
    t0 = time.time()

    for lang in langs:
        lang_path = os.path.join(DATASET_DIR, lang)
        if not os.path.exists(lang_path):
            log(f"ข้าม {lang}: ไม่พบโฟลเดอร์ dataset", "WARN")
            continue

        out_dir = feature_dir_for(lang)
        make_dirs(out_dir)

        log(f"ภาษา: {lang}", "LANG")
        processed, skipped, no_label = 0, 0, 0

        for root, _, files in os.walk(lang_path):
            for file in files:
                if not file.lower().endswith(('.wav', '.flac', '.mp3')):
                    continue

                file_path = os.path.join(root, file)
                label     = get_label_from_path(file_path, lang)

                if not label:
                    no_label += 1
                    continue

                base_name, _ = os.path.splitext(file)
                out_name  = f"{label}_{base_name}.npy"
                out_path  = os.path.join(out_dir, out_name)

                if os.path.exists(out_path):
                    skipped += 1
                    continue

                try:
                    mel = extract_mel(file_path)
                    np.save(out_path, mel)
                    processed += 1
                    if processed % 100 == 0:
                        log(f"  {lang}: {processed} ไฟล์...", "INFO")
                except Exception as e:
                    log(f"  ข้าม {file}: {e}", "WARN")

        grand_total += processed
        log(f"  → สกัดใหม่: {processed} | ข้าม: {skipped} | ไม่มี label: {no_label}", "OK")

    log(f"รวมสกัดใหม่ทั้งหมด: {grand_total} ไฟล์ ({time.time()-t0:.1f}s)", "OK")

# ==========================================
# STAGE 3 — LOAD DATA PER LANGUAGE
# ==========================================
def load_lang_data(lang):
    """
    โหลด .npy จาก extracted_features/{lang}/
    ชื่อไฟล์: Emotion_basename.npy
    คืนค่า X (N,128,130,1), y_enc, le
    """
    out_dir = feature_dir_for(lang)
    if not os.path.exists(out_dir):
        log(f"ไม่พบ feature folder: {out_dir}", "ERROR")
        return None, None, None

    X, y = [], []
    for file in os.listdir(out_dir):
        if not file.endswith('.npy'):
            continue
        label = file.split('_')[0]   # Emotion_xxxxx.npy
        mel   = np.load(os.path.join(out_dir, file))

        # ทำให้ทุกไฟล์มีขนาด (N_MELS, MAX_TIME_STEPS) เท่ากัน
        if mel.ndim == 1:
            continue  # ข้ามไฟล์ที่ shape ผิดปกติ
        # ตัด/เติม แกน 0 (n_mels)
        if mel.shape[0] < N_MELS:
            mel = np.pad(mel, ((0, N_MELS - mel.shape[0]), (0, 0)), mode='constant')
        else:
            mel = mel[:N_MELS, :]
        # ตัด/เติม แกน 1 (time)
        if mel.shape[1] < MAX_TIME_STEPS:
            mel = np.pad(mel, ((0, 0), (0, MAX_TIME_STEPS - mel.shape[1])), mode='constant')
        else:
            mel = mel[:, :MAX_TIME_STEPS]

        X.append(mel)
        y.append(label)

    if len(X) == 0:
        log(f"{lang}: ไม่พบข้อมูลใน {out_dir}", "WARN")
        return None, None, None

    X = np.array(X, dtype=np.float32)[:, np.newaxis, :, :]  # (N, 1, 128, 130) channels-first
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    log(f"{lang:10s}: {len(X):,} ตัวอย่าง | อารมณ์: {list(le.classes_)}", "OK")
    return X, y_enc, le


def split_data(X, y_enc):
    """70% train / 15% val / 15% test"""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_enc, test_size=0.30, stratify=y_enc, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ==========================================
# STAGE 4 — BUILD & TRAIN PER-LANGUAGE MODEL  (PyTorch + CUDA)
# ==========================================

class MelDataset(Dataset):
    """Dataset พร้อม SpecAugment สำหรับ train set"""
    def __init__(self, X, y, augment=False):
        # X: (N, 1, 128, 130) float32 | y: (N,) int
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            x = self._spec_augment(x)
        return x, self.y[idx]

    @staticmethod
    def _spec_augment(x, freq_mask=20, time_mask=25, n_freq=1, n_time=2):
        """
        SpecAugment: สุ่ม mask แถวความถี่และแถวเวลา
        ช่วยให้โมเดล robust ต่อ missing info เพิ่ม acc ~3-5%
        """
        _, F, T = x.shape
        for _ in range(n_freq):
            f  = int(torch.randint(0, freq_mask, (1,)))
            f0 = int(torch.randint(0, F - f + 1, (1,)))
            x[:, f0:f0+f, :] = 0
        for _ in range(n_time):
            t  = int(torch.randint(0, time_mask, (1,)))
            t0 = int(torch.randint(0, T - t + 1, (1,)))
            x[:, :, t0:t0+t] = 0
        return x


class ResBlock(nn.Module):
    """Residual block — skip connection ช่วยให้ gradient ไหลได้ดีขึ้น"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class EmotionResNet(nn.Module):
    """
    CNN + Residual blocks
    เพิ่ม accuracy ~5-10% เทียบกับ plain CNN
    """
    def __init__(self, n_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))   # output คงที่ไม่ว่า input size เท่าไหร่
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.classifier(x)


def _get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        log(f"GPU: {torch.cuda.get_device_name(0)}", "OK")
    else:
        device = torch.device('cpu')
        log("ใช้ CPU", "WARN")
    return device


def _class_weights(y_train, n_classes, device):
    """คำนวณ inverse-frequency weight เพื่อชดเชย class imbalance"""
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)   # กัน div/0
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_lang(lang, device):
    log(f"▶ Training: {lang}", "LANG")

    X, y_enc, le = load_lang_data(lang)
    if X is None:
        log(f"ข้าม {lang}: ไม่มีข้อมูล", "WARN")
        return

    n_classes = len(le.classes_)

    # เตือนถ้า dataset เล็กเกินไป
    if len(X) < 200:
        log(f"  คำเตือน: {lang} มีแค่ {len(X)} ตัวอย่าง — ผล accuracy อาจไม่น่าเชื่อถือ", "WARN")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_enc)
    log(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}", "INFO")

    make_dirs(MODEL_DIR, result_dir_for(lang))
    mp = model_path_for(lang)

    # SpecAugment เฉพาะ train set
    train_loader = DataLoader(MelDataset(X_train, y_train, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=device.type=='cuda')
    val_loader   = DataLoader(MelDataset(X_val,   y_val,   augment=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type=='cuda')

    model     = EmotionResNet(n_classes).to(device)
    weights   = _class_weights(y_train, n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Cosine annealing: lr ค่อยๆ ลดแบบ smooth ดีกว่า step decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Mixed Precision (AMP) — เร็วขึ้น ~2x บน RTX 3060
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_acc = 0.0
    best_state   = None
    patience_cnt = 0
    EARLY_STOP   = 20
    history      = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    out  = model(xb)
                    loss = criterion(out, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out  = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            train_loss    += loss.item() * len(xb)
            train_correct += (out.argmax(1) == yb).sum().item()
            train_total   += len(xb)

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.amp.autocast('cuda') if device.type == 'cuda' else contextlib.nullcontext():
                    out  = model(xb)
                    loss = criterion(out, yb)
                val_loss    += loss.item() * len(xb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total   += len(xb)

        t_acc  = train_correct / train_total
        v_acc  = val_correct   / val_total
        t_loss = train_loss    / train_total
        v_loss = val_loss      / val_total
        lr_now = optimizer.param_groups[0]['lr']

        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)

        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"loss={t_loss:.4f} acc={t_acc:.4f} | "
              f"val_loss={v_loss:.4f} val_acc={v_acc:.4f} | lr={lr_now:.2e}")

        # --- Early stopping & checkpoint ---
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP:
                log(f"  Early stopping at epoch {epoch}", "INFO")
                break

    # บันทึก best model
    model.load_state_dict(best_state)
    torch.save({'state_dict': model.state_dict(),
                'n_classes':  n_classes,
                'best_val_acc': best_val_acc}, mp)

    # บันทึก label encoder และ test set
    with open(encoder_path_for(lang), 'wb') as f:
        pickle.dump(le, f)
    np.save(os.path.join(MODEL_DIR, f"{lang}_X_test.npy"), X_test)
    np.save(os.path.join(MODEL_DIR, f"{lang}_y_test.npy"), y_test)

    _plot_history(history, lang)
    log(f"โมเดล {lang} บันทึกที่: {mp}  (best val_acc={best_val_acc*100:.2f}%)", "OK")


def stage_train(langs=None):
    print("\n" + "="*60)
    print("  STAGE 4: MODEL TRAINING  (PyTorch + CUDA)")
    print("="*60)

    device = _get_device()
    langs  = langs or TARGET_LANGUAGES
    for lang in langs:
        print()
        train_one_lang(lang, device)


def _plot_history(history, lang):
    out_dir = result_dir_for(lang)
    make_dirs(out_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Curves — {lang}", fontsize=13)

    axes[0].plot(history['train_acc'], label='Train')
    axes[0].plot(history['val_acc'],   label='Val')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_loss'], label='Train')
    axes[1].plot(history['val_loss'],   label='Val')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(out_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    log(f"  Training curves → {out}", "INFO")

# ==========================================
# STAGE 5 — EVALUATE PER-LANGUAGE MODEL
# ==========================================
def eval_one_lang(lang):
    mp  = model_path_for(lang)
    ep  = encoder_path_for(lang)
    xp  = os.path.join(MODEL_DIR, f"{lang}_X_test.npy")
    yp  = os.path.join(MODEL_DIR, f"{lang}_y_test.npy")

    if not all(os.path.exists(p) for p in [mp, ep, xp, yp]):
        log(f"{lang}: ยังไม่มีโมเดลหรือ test data → รัน --stage train ก่อน", "WARN")
        return None

    with open(ep, 'rb') as f:
        le = pickle.load(f)
    X_test = np.load(xp)
    y_test = np.load(yp)

    device = _get_device()
    ckpt   = torch.load(mp, map_location=device, weights_only=True)
    model  = EmotionCNN(ckpt['n_classes']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    all_preds = []
    loader = DataLoader(MelDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for xb, _ in loader:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)

    y_pred = np.array(all_preds)
    acc    = np.mean(y_pred == y_test)

    log(f"{lang:10s}: Test Accuracy = {acc*100:.2f}%", "OK")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix — {lang}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    out = os.path.join(result_dir_for(lang), "confusion_matrix.png")
    make_dirs(result_dir_for(lang))
    plt.savefig(out, dpi=150)
    plt.close()
    log(f"  Confusion matrix → {out}", "INFO")
    return acc


def stage_eval(langs=None):
    print("\n" + "="*60)
    print("  STAGE 5: EVALUATION  (แยกต่อภาษา)")
    print("="*60)

    langs = langs or TARGET_LANGUAGES
    summary = {}
    for lang in langs:
        print()
        acc = eval_one_lang(lang)
        if acc is not None:
            summary[lang] = acc

    if summary:
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)
        for lang, acc in summary.items():
            bar = "█" * int(acc * 20)
            log(f"{lang:10s}: {acc*100:5.1f}%  {bar}", "OK")
        avg = np.mean(list(summary.values()))
        log(f"{'Average':10s}: {avg*100:5.1f}%", "INFO")

# ==========================================
# MAIN
# ==========================================
def run_pipeline(run_stage=None, lang_filter=None):
    start = time.time()
    langs = [lang_filter] if lang_filter else TARGET_LANGUAGES

    print("\n" + "="*60)
    print("  SPEECH EMOTION RECOGNITION — PER-LANGUAGE PIPELINE")
    print(f"  ภาษา : {', '.join(langs)}")
    print(f"  Stage: {run_stage or 'all'}")
    print("="*60)

    if run_stage in (None, 'verify'):
        stage_verify()
        if run_stage == 'verify':
            return

    if run_stage in (None, 'extract'):
        stage_extract(langs)
        if run_stage == 'extract':
            return

    if run_stage in (None, 'train'):
        stage_train(langs)
        if run_stage == 'train':
            _print_done(start)
            return

    if run_stage in (None, 'eval'):
        stage_eval(langs)

    _print_done(start)


def _print_done(start):
    elapsed = time.time() - start
    print("\n" + "="*60)
    log(f"Pipeline เสร็จสิ้น — {elapsed:.1f} วินาที", "OK")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-Language Speech Emotion Recognition Pipeline"
    )
    parser.add_argument(
        '--stage',
        choices=['verify', 'extract', 'train', 'eval'],
        default=None,
        help='รันแค่ stage เดียว (ไม่ระบุ = รันทั้งหมด)'
    )
    parser.add_argument(
        '--lang',
        choices=TARGET_LANGUAGES,
        default=None,
        help='รันแค่ภาษาเดียว (ไม่ระบุ = ทุกภาษา)'
    )
    args = parser.parse_args()
    run_pipeline(args.stage, args.lang)
