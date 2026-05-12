"""
wav2vec2_pipeline.py
====================
Feature extraction + Training ด้วย Wav2Vec2 (Facebook XLSR-53)
แทนที่ mel-spectrogram ธรรมดาด้วย pre-trained speech representation

Model: facebook/wav2vec2-large-xlsr-53 (รองรับ 53 ภาษา รวม zh, ja, ko, th, en)
Approach:
  1. โหลด raw audio
  2. ส่งผ่าน Wav2Vec2 → hidden states (T × 1024)
  3. Mean pool → 1024-dim vector ต่อไฟล์
  4. Train MLP classifier

Usage:
  python wav2vec2_pipeline.py --stage extract            # extract ทุกภาษา
  python wav2vec2_pipeline.py --stage extract --lang Thai
  python wav2vec2_pipeline.py --stage train              # train ทุกภาษา
  python wav2vec2_pipeline.py --stage all                # extract + train + eval
  python wav2vec2_pipeline.py --stage all --lang English
"""

import os
import sys
import argparse
import pickle
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR   = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
W2V_FEAT   = os.path.join(BASE_DIR, "extracted_features_w2v")   # แยกจาก mel-spec
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results", "wav2vec2")
os.makedirs(W2V_FEAT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

LANGUAGES = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']
SR        = 16000    # Wav2Vec2 ต้องการ 16kHz
MAX_SEC   = 6        # ตัดที่ 6 วินาที
BATCH     = 32
EPOCHS    = 80
LR        = 1e-3
FEAT_DIM  = 1024     # xlsr-53 hidden size

EMOTION_KEYWORDS = {
    'angry':'Angry','anger':'Angry','분노':'Angry',
    'happy':'Happy','happiness':'Happy','joy':'Happy','기쁨':'Happy','행복':'Happy',
    'sad':'Sad','sadness':'Sad','슬픔':'Sad',
    'neutral':'Neutral','중립':'Neutral',
    'fear':'Fear','fearful':'Fear','공포':'Fear',
    'surprise':'Surprise','surprised':'Surprise','놀람':'Surprise',
    'disgust':'Disgust','혐오':'Disgust',
}
RAVDESS_MAP = {'01':'Neutral','02':'Neutral','03':'Happy','04':'Sad',
               '05':'Angry','06':'Fear','07':'Disgust','08':'Surprise'}


def log(msg):
    print(f"  {msg}")


# ─────────────────────────────────────────────
#  WAV2VEC2 FEATURE EXTRACTOR
# ─────────────────────────────────────────────
_w2v_model     = None
_w2v_extractor = None

def get_w2v():
    """โหลด Wav2Vec2FeatureExtractor + Wav2Vec2Model (ครั้งเดียว)"""
    global _w2v_model, _w2v_extractor
    if _w2v_model is None:
        # ใช้ FeatureExtractor ไม่ใช่ Processor
        # (wav2vec2-large-xlsr-53 ไม่มี vocab.json — ไม่ใช่ ASR model)
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        model_name = "facebook/wav2vec2-large-xlsr-53"
        log(f"โหลด {model_name} ...")
        _w2v_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        _w2v_model     = Wav2Vec2Model.from_pretrained(model_name)
        _w2v_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _w2v_model = _w2v_model.to(device)
        log(f"โหลดสำเร็จ — Device: {device}")
    return _w2v_extractor, _w2v_model


def extract_w2v_feature(audio_path):
    """โหลด audio → Wav2Vec2 hidden states → mean pool → (1024,)"""
    import librosa
    extractor, model = get_w2v()
    device = next(model.parameters()).device

    y, _ = librosa.load(audio_path, sr=SR)
    max_len = SR * MAX_SEC
    if len(y) > max_len:
        y = y[:max_len]
    elif len(y) < SR * 0.3:   # สั้นกว่า 0.3 วินาที — skip
        return None

    inputs = extractor(y, sampling_rate=SR, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
    hidden = outputs.last_hidden_state      # (1, T, 1024)
    feat   = hidden.mean(dim=1).squeeze(0)  # (1024,)
    return feat.cpu().numpy().astype(np.float32)


def get_label(file_path, lang):
    fname      = os.path.basename(file_path)
    path_lower = file_path.lower().replace('\\', '/')

    if lang == 'English':
        parts = os.path.splitext(fname)[0].split('-')
        if len(parts) >= 3:
            label = RAVDESS_MAP.get(parts[2])
            if label:
                return label

    for kw, em in EMOTION_KEYWORDS.items():
        if kw in path_lower:
            return em
    return None


# ─────────────────────────────────────────────
#  STAGE: EXTRACT
# ─────────────────────────────────────────────
def stage_extract(langs=None):
    print("\n" + "=" * 60)
    print("  WAV2VEC2 FEATURE EXTRACTION")
    print("=" * 60)
    langs = langs or LANGUAGES

    # ตรวจว่ามีไฟล์ audio จริงก่อนโหลด model
    has_audio = False
    for lang in langs:
        lang_path = os.path.join(DATASET_DIR, lang)
        if not os.path.exists(lang_path):
            continue
        for root, _, files in os.walk(lang_path):
            if any(f.lower().endswith(('.wav','.flac','.mp3')) for f in files):
                has_audio = True
                break
        if has_audio:
            break

    if not has_audio:
        log("ไม่มีไฟล์ audio raw ใน dataset/ — ข้าม W2V extraction")
        log("(Chinese/Japan มีเฉพาะ .npy features ไม่มี raw audio)")
        return

    # โหลด model ครั้งเดียวก่อน loop ทั้งหมด
    log("โหลด Wav2Vec2 model (ครั้งเดียว)...")
    try:
        processor, model = get_w2v()
    except Exception as e:
        log(f"โหลด model ล้มเหลว: {e}")
        log("ตรวจสอบ internet connection หรือ HuggingFace token")
        return

    device = next(model.parameters()).device
    log(f"Model พร้อม — Device: {device}")

    for lang in langs:
        lang_path = os.path.join(DATASET_DIR, lang)
        if not os.path.exists(lang_path):
            log(f"ข้าม {lang}: ไม่พบ dataset")
            continue

        # นับ audio ไฟล์ก่อน — ถ้าไม่มีเลยก็ข้าม
        audio_files = []
        for root, _, files in os.walk(lang_path):
            for fname in files:
                if fname.lower().endswith(('.wav', '.flac', '.mp3')):
                    audio_files.append(os.path.join(root, fname))

        if not audio_files:
            log(f"ข้าม {lang}: ไม่มี audio files (มีแค่ .npy features)")
            continue

        out_dir = os.path.join(W2V_FEAT, lang)
        os.makedirs(out_dir, exist_ok=True)

        log(f"\n[{lang}] — {len(audio_files)} ไฟล์")
        processed = skipped = no_label = errors = 0
        t0 = time.time()

        for fpath in audio_files:
            label = get_label(fpath, lang)
            if not label:
                no_label += 1
                continue

            base     = os.path.splitext(os.path.basename(fpath))[0]
            npy_path = os.path.join(out_dir, f"{label}_{base}.npy")
            if os.path.exists(npy_path):
                skipped += 1
                continue
            try:
                feat = extract_w2v_feature(fpath)
                if feat is None:
                    errors += 1
                    continue
                np.save(npy_path, feat)
                processed += 1
                if processed % 100 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed
                    log(f"  {processed}/{len(audio_files)} ไฟล์  ({rate:.1f} ไฟล์/วินาที)")
            except Exception as e:
                errors += 1
                if errors <= 3:
                    log(f"  Error: {os.path.basename(fpath)}: {str(e)[:60]}")

        log(f"  → สกัดใหม่: {processed} | ข้าม: {skipped} | ไม่มี label: {no_label} | error: {errors}")

    log("Feature extraction เสร็จสิ้น")


# ─────────────────────────────────────────────
#  MLP CLASSIFIER
# ─────────────────────────────────────────────
class EmotionMLP(nn.Module):
    def __init__(self, n_classes, input_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ─────────────────────────────────────────────
#  STAGE: TRAIN
# ─────────────────────────────────────────────
def load_features(lang):
    d = os.path.join(W2V_FEAT, lang)
    if not os.path.exists(d):
        return None, None, None
    files = [f for f in os.listdir(d) if f.endswith('.npy')]
    if not files:
        return None, None, None

    emotions = sorted({f.split('_')[0] for f in files})
    le = LabelEncoder()
    le.fit(emotions)

    X, y = [], []
    for fname in files:
        emo = fname.split('_')[0]
        arr = np.load(os.path.join(d, fname))
        X.append(arr)
        y.append(le.transform([emo])[0])

    return np.array(X, dtype=np.float32), np.array(y), le


def train_lang(lang):
    log(f"\n{'='*50}")
    log(f"Training: {lang} (Wav2Vec2 + MLP)")
    log(f"{'='*50}")

    X, y, le = load_features(lang)
    if X is None:
        log(f"  ไม่พบ features ใน {os.path.join(W2V_FEAT, lang)}")
        log(f"  รัน --stage extract --lang {lang} ก่อน")
        return

    log(f"  {len(X)} ตัวอย่าง | {len(le.classes_)} classes: {list(le.classes_)}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15,
                                               stratify=y, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15,
                                                  stratify=y_tr, random_state=42)
    log(f"  Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(le.classes_)

    counts  = np.bincount(y_tr)
    weights = 1.0 / counts[y_tr]
    sampler = WeightedRandomSampler(weights, len(weights))

    tr_load  = DataLoader(FeatDataset(X_tr,  y_tr),  BATCH, sampler=sampler)
    val_load = DataLoader(FeatDataset(X_val, y_val), BATCH, shuffle=False)
    te_load  = DataLoader(FeatDataset(X_te,  y_te),  BATCH, shuffle=False)

    model  = EmotionMLP(n_classes).to(device)
    cls_w  = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    cls_w  = cls_w / cls_w.sum() * n_classes
    crit   = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.1)
    optim  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_acc, best_state, patience, no_imp = 0.0, None, 15, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_load:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
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
        val_acc = correct / total

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                log(f"  Early stop @ epoch {ep}")
                break

        if ep % 10 == 0:
            log(f"  Epoch {ep:3d}/{EPOCHS} | val_acc={val_acc:.4f} (best={best_acc:.4f})")

    # Test evaluation
    model.load_state_dict(best_state)
    model.to(device).eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in te_load:
            xb = xb.to(device)
            all_pred.extend(model(xb).argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    test_acc = accuracy_score(all_true, all_pred) * 100
    test_f1  = f1_score(all_true, all_pred, average='weighted') * 100
    report   = classification_report(all_true, all_pred,
                                     target_names=le.classes_, zero_division=0)

    log(f"\n  Test Accuracy : {test_acc:.2f}%")
    log(f"  F1 Weighted   : {test_f1:.2f}%")
    log(f"\n{report}")

    # บันทึก model
    out_path = os.path.join(MODEL_DIR, f"{lang}_w2v_model.pt")
    torch.save({
        'state_dict':   best_state,
        'n_classes':    n_classes,
        'best_val_acc': best_acc,
        'test_acc':     test_acc,
        'feat_dim':     FEAT_DIM,
    }, out_path)

    # บันทึก label encoder
    enc_path = os.path.join(MODEL_DIR, f"{lang}_w2v_label_encoder.pkl")
    with open(enc_path, 'wb') as f:
        pickle.dump(le, f)

    # บันทึก test set
    np.save(os.path.join(MODEL_DIR, f"{lang}_w2v_X_test.npy"), X_te)
    np.save(os.path.join(MODEL_DIR, f"{lang}_w2v_y_test.npy"), np.array(all_true))

    log(f"  บันทึก {out_path}  (val_acc={best_acc*100:.2f}%  test_acc={test_acc:.2f}%)")
    return test_acc


# ─────────────────────────────────────────────
#  COMPARE: Wav2Vec2 vs Mel-Spectrogram
# ─────────────────────────────────────────────
def compare_results():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # โหลดผล mel-spec จาก evaluation_report
    mel_acc = {
        'Chinese': 83.83,
        'Japan':   88.48,
        'Korean':  66.22,
        'Thai':    88.45,
        'English': 82.89,
    }

    w2v_acc = {}
    for lang in LANGUAGES:
        pt = os.path.join(MODEL_DIR, f"{lang}_w2v_model.pt")
        if os.path.exists(pt):
            ckpt = torch.load(pt, map_location='cpu', weights_only=False)
            w2v_acc[lang] = ckpt.get('test_acc', 0)

    if not w2v_acc:
        return

    langs  = [l for l in LANGUAGES if l in w2v_acc]
    x      = np.arange(len(langs))
    w      = 0.35
    mel_v  = [mel_acc.get(l, 0) for l in langs]
    w2v_v  = [w2v_acc.get(l, 0) for l in langs]

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, mel_v, w, label='Mel-Spectrogram ResNet', color='#3498DB', alpha=0.85)
    b2 = ax.bar(x + w/2, w2v_v, w, label='Wav2Vec2 + MLP',         color='#E74C3C', alpha=0.85)

    for bar, v in zip(list(b1) + list(b2), mel_v + w2v_v):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Mel-Spectrogram ResNet vs Wav2Vec2 MLP', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axhline(80, color='gray', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = os.path.join(RESULT_DIR, 'comparison_melspec_vs_wav2vec2.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    log(f"บันทึก comparison graph: {out}")

    # Print table
    print(f"\n  {'ภาษา':<12} {'Mel-Spec':>10} {'Wav2Vec2':>10} {'Delta':>8}")
    print("  " + "-" * 44)
    for l in langs:
        m = mel_acc.get(l, 0)
        w = w2v_acc.get(l, 0)
        d = w - m
        sign = '+' if d >= 0 else ''
        print(f"  {l:<12} {m:>9.2f}% {w:>9.2f}% {sign}{d:>6.2f}%")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='all',
                        choices=['extract', 'train', 'compare', 'all'])
    parser.add_argument('--lang', default=None)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  WAV2VEC2 SPEECH EMOTION PIPELINE")
    print(f"  Stage: {args.stage}  |  Lang: {args.lang or 'all'}")
    print("=" * 60)
    device_str = 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'
    print(f"  Device: {device_str}")

    langs = [args.lang] if args.lang else LANGUAGES

    if args.stage in ('extract', 'all'):
        stage_extract(langs)

    if args.stage in ('train', 'all'):
        results = {}
        for lang in langs:
            acc = train_lang(lang)
            if acc:
                results[lang] = acc

        print("\n" + "=" * 60)
        print("  SUMMARY")
        for lang, acc in results.items():
            print(f"  {lang:<12}: {acc:.2f}%")

    if args.stage in ('compare', 'all'):
        compare_results()

    print("\n  เสร็จสิ้น")
