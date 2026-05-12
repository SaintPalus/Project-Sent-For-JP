"""
language_detector.py
====================
Train language classifier จาก mel-spectrogram features
ใช้สำหรับ auto-detect ภาษาใน demo_live.py --lang auto

Approach:
  - โหลด sample จาก extracted_features/{lang}/
  - ลดมิติด้วย PCA → train SVM classifier
  - บันทึกเป็น models/lang_detector.pkl

Usage:
  python language_detector.py          # train + save
  python language_detector.py --test   # train + test + visualize
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

BASE_DIR  = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
FEAT_DIR  = os.path.join(BASE_DIR, "extracted_features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LANGUAGES = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']

N_MELS = 128
T      = 130
MAX_PER_LANG = 800    # จำกัดเพื่อสมดุล


def log(msg):
    print(f"  {msg}")


def load_lang_features():
    """โหลด mel-spec features พร้อม lang label"""
    from collections import defaultdict

    X, y = [], []
    rng = np.random.RandomState(42)

    for lang_idx, lang in enumerate(LANGUAGES):
        d = os.path.join(FEAT_DIR, lang)
        if not os.path.exists(d):
            log(f"ข้าม {lang} (ไม่พบ directory)")
            continue
        files = [f for f in os.listdir(d) if f.endswith('.npy')]
        if not files:
            continue
        chosen = rng.choice(files, min(MAX_PER_LANG, len(files)), replace=False)
        log(f"{lang}: ใช้ {len(chosen)}/{len(files)} ไฟล์")
        for fname in chosen:
            arr = np.load(os.path.join(d, fname))
            if arr.shape[1] >= T:
                arr = arr[:, :T]
            else:
                arr = np.pad(arr, ((0, 0), (0, T - arr.shape[1])), mode='edge')
            X.append(arr.flatten())
            y.append(lang_idx)

    return np.array(X, dtype=np.float32), np.array(y)


def train_detector(X, y):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    log(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    # Pipeline: StandardScaler → PCA → SVM
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=80, random_state=42)),
        ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                       probability=True, random_state=42)),
    ])

    log("Training language detector ...")
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    acc  = accuracy_score(y_te, pred) * 100
    log(f"Test Accuracy: {acc:.2f}%")
    log("\n" + classification_report(y_te, pred,
                                      target_names=LANGUAGES, zero_division=0))
    return pipe, acc


def save_detector(pipe):
    out_path = os.path.join(MODEL_DIR, "lang_detector.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(pipe, f)
    log(f"บันทึก {out_path}")
    return out_path


def plot_confusion(pipe, X_te, y_te):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    pred = pipe.predict(X_te)
    cm   = confusion_matrix(y_te, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',    'Language Detection — Count'),
        (axes[1], cm_norm, '.1f',  'Language Detection — Normalized (%)'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=LANGUAGES, yticklabels=LANGUAGES,
                    ax=ax, linewidths=0.5)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title, fontweight='bold')

    out_dir = os.path.join(BASE_DIR, "results", "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'lang_detector_confusion.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    log(f"บันทึก {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='plot confusion matrix')
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  LANGUAGE DETECTOR TRAINING")
    print("=" * 55)

    log("โหลด features ...")
    X, y = load_lang_features()
    log(f"รวม: {len(X)} ตัวอย่าง, {len(set(y))} ภาษา")

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)

    pipe, acc = train_detector(X, y)
    save_detector(pipe)

    if args.test:
        plot_confusion(pipe, X_te, y_te)

    print(f"\n  Language Detector Accuracy: {acc:.2f}%")
    print("  ใช้งานใน demo: python demo_live.py --lang auto")
