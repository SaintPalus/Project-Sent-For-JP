"""
evaluate.py — Auto Evaluation for Speech Emotion Recognition
=============================================================
ทดสอบ accuracy จริงของทุกโมเดลอัตโนมัติ พร้อม:
  - Per-class accuracy (Precision / Recall / F1)
  - Confusion Matrix (บันทึกเป็น .png)
  - Summary table ทุกภาษา
  - บันทึกผลเป็น results/evaluation_report.txt

การใช้งาน:
    python evaluate.py               # ทดสอบทุกภาษา
    python evaluate.py --lang Thai   # ทดสอบแค่ภาษาเดียว
    python evaluate.py --lang Thai --lang Chinese   # หลายภาษา
"""

import os
import sys
import argparse
import pickle
import time
import contextlib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ==========================================
# CONFIG — ต้องตรงกับ data_pipeline.py
# ==========================================
BASE_DIR   = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

TARGET_LANGUAGES = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']
BATCH_SIZE = 64

# ==========================================
# MODEL (ต้องเหมือนกับใน data_pipeline.py)
# ==========================================
class ResBlock(nn.Module):
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
    def __init__(self, n_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
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
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))))


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# HELPERS
# ==========================================
def log(msg, level="INFO"):
    icons = {"INFO": "  ", "OK": "OK", "WARN": "!!", "ERROR": "XX", "HEAD": ">>"}
    print(f"[{icons.get(level,'  ')}] {msg}")


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(lang, device):
    mp = os.path.join(MODEL_DIR, f"{lang}_model.pt")
    ep = os.path.join(MODEL_DIR, f"{lang}_label_encoder.pkl")

    if not os.path.exists(mp):
        return None, None, None, None

    ckpt = torch.load(mp, map_location=device, weights_only=True)
    model = EmotionResNet(ckpt['n_classes']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    with open(ep, 'rb') as f:
        le = pickle.load(f)

    X_test = np.load(os.path.join(MODEL_DIR, f"{lang}_X_test.npy"))
    y_test = np.load(os.path.join(MODEL_DIR, f"{lang}_y_test.npy"))

    return model, le, X_test, y_test


def predict(model, X_test, device):
    loader = DataLoader(NumpyDataset(X_test, np.zeros(len(X_test), dtype=int)),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds, all_probs = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            with torch.amp.autocast('cuda') if device.type == 'cuda' else contextlib.nullcontext():
                logits = model(xb)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
    return np.array(all_preds), np.array(all_probs)

# ==========================================
# EVALUATION CORE
# ==========================================
def evaluate_lang(lang, device):
    """
    ประเมิน 1 ภาษา → คืน dict ผลลัพธ์
    """
    model, le, X_test, y_test = load_model(lang, device)

    if model is None:
        log(f"{lang}: ไม่พบโมเดล — ข้าม", "WARN")
        return None

    n_samples = len(X_test)
    n_classes = len(le.classes_)
    classes   = [str(c) for c in le.classes_]   # กัน numpy str

    y_pred, y_probs = predict(model, X_test, device)

    acc     = accuracy_score(y_test, y_pred)
    f1_mac  = f1_score(y_test, y_pred, average='macro',    zero_division=0)
    f1_wt   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report  = classification_report(y_test, y_pred, target_names=classes,
                                    digits=4, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    # Per-class accuracy
    per_class = {}
    for i, cls in enumerate(classes):
        mask = y_test == i
        if mask.sum() > 0:
            per_class[cls] = (y_pred[mask] == i).mean()
        else:
            per_class[cls] = 0.0

    return {
        'lang'      : lang,
        'n_samples' : n_samples,
        'n_classes' : n_classes,
        'classes'   : classes,
        'acc'       : acc,
        'f1_macro'  : f1_mac,
        'f1_weighted': f1_wt,
        'report'    : report,
        'cm'        : cm,
        'per_class' : per_class,
        'y_test'    : y_test,
        'y_pred'    : y_pred,
        'y_probs'   : y_probs,
    }

# ==========================================
# OUTPUT: PRINT + PLOTS + REPORT FILE
# ==========================================
def print_lang_result(res):
    lang = res['lang']
    sep  = "=" * 60

    print(f"\n{sep}")
    print(f"  {lang.upper()} — Test Set ({res['n_samples']} samples, {res['n_classes']} classes)")
    print(sep)
    print(f"  Accuracy        : {res['acc']*100:6.2f}%")
    print(f"  F1 Macro        : {res['f1_macro']*100:6.2f}%")
    print(f"  F1 Weighted     : {res['f1_weighted']*100:6.2f}%")
    print()
    print("  Per-class Accuracy:")
    for cls, acc in res['per_class'].items():
        bar = '#' * int(acc * 20)
        print(f"    {cls:12s}: {acc*100:5.1f}%  [{bar:<20s}]")
    print()
    print("  Classification Report:")
    for line in res['report'].splitlines():
        print("  " + line)


def save_confusion_matrix(res):
    out_dir = os.path.join(RESULT_DIR, res['lang'])
    os.makedirs(out_dir, exist_ok=True)

    cm        = res['cm']
    classes   = res['classes']
    n_classes = res['n_classes']

    # Absolute counts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {res['lang']}  (Acc={res['acc']*100:.1f}%)", fontsize=13)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title('Count')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized (row = true class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=classes, yticklabels=classes, ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_title('Normalized (Recall)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    out = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Confusion matrix -> {out}", "OK")


def save_per_class_bar(res):
    out_dir = os.path.join(RESULT_DIR, res['lang'])
    os.makedirs(out_dir, exist_ok=True)

    classes = list(res['per_class'].keys())
    accs    = [res['per_class'][c] * 100 for c in classes]
    colors  = ['#2ecc71' if a >= 70 else '#e67e22' if a >= 50 else '#e74c3c' for a in accs]

    fig, ax = plt.subplots(figsize=(max(6, len(classes)*1.2), 4))
    bars = ax.bar(classes, accs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f"Per-class Accuracy — {res['lang']}")
    ax.axhline(res['acc']*100, color='navy', linestyle='--', linewidth=1, label=f"Overall {res['acc']*100:.1f}%")
    ax.legend()

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out = os.path.join(out_dir, "per_class_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Per-class chart  -> {out}", "OK")


def save_confidence_distribution(res):
    """
    กราฟ 2 แบบ:
      1. Histogram ของ max-confidence ที่ถูก vs ผิด
      2. Calibration curve (confidence vs actual accuracy)
    """
    out_dir = os.path.join(RESULT_DIR, res['lang'])
    os.makedirs(out_dir, exist_ok=True)

    y_test  = res['y_test']
    y_pred  = res['y_pred']
    y_probs = res['y_probs']         # (N, n_classes)
    classes = res['classes']

    max_conf   = y_probs.max(axis=1) * 100    # confidence ของ class ที่ predict
    correct_mask = (y_pred == y_test)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{res['lang']} — Confidence Score Analysis", fontsize=13, fontweight='bold')

    # ── 1. Histogram: correct vs wrong ──
    ax = axes[0]
    bins = np.linspace(0, 100, 26)
    ax.hist(max_conf[correct_mask],  bins=bins, alpha=0.7, color='#2ECC71',
            label=f'Correct ({correct_mask.sum()})', edgecolor='white', linewidth=0.5)
    ax.hist(max_conf[~correct_mask], bins=bins, alpha=0.7, color='#E74C3C',
            label=f'Wrong ({(~correct_mask).sum()})', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Max Confidence (%)')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution: Correct vs Wrong')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    mean_c  = max_conf[correct_mask].mean()
    mean_w  = max_conf[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
    ax.axvline(mean_c, color='#27AE60', linestyle='--', linewidth=1.5,
               label=f'Correct mean={mean_c:.1f}%')
    ax.axvline(mean_w, color='#C0392B', linestyle='--', linewidth=1.5,
               label=f'Wrong mean={mean_w:.1f}%')
    ax.legend(fontsize=8)

    # ── 2. Calibration curve ──
    ax2 = axes[1]
    bin_edges = np.linspace(0, 1, 11)
    bin_centers, bin_accs, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_probs.max(axis=1) >= lo) & (y_probs.max(axis=1) < hi)
        if mask.sum() >= 5:
            bin_centers.append((lo + hi) / 2 * 100)
            bin_accs.append(correct_mask[mask].mean() * 100)
            bin_counts.append(mask.sum())

    if bin_centers:
        ax2.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')
        sc = ax2.scatter(bin_centers, bin_accs, c=bin_counts, cmap='Blues',
                         s=80, zorder=3, edgecolors='#2C3E50', linewidths=0.8)
        ax2.plot(bin_centers, bin_accs, '-', color='#3498DB', linewidth=1.5, alpha=0.7)
        plt.colorbar(sc, ax=ax2, label='Sample count')
    ax2.set_xlabel('Mean Confidence (%)')
    ax2.set_ylabel('Actual Accuracy (%)')
    ax2.set_title('Calibration Curve')
    ax2.set_xlim(0, 105)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(out_dir, "confidence_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Confidence chart -> {out}", "OK")


def save_summary_chart(all_results):
    """Bar chart รวมทุกภาษาเทียบกัน"""
    os.makedirs(RESULT_DIR, exist_ok=True)

    langs = [r['lang'] for r in all_results]
    accs  = [r['acc'] * 100 for r in all_results]
    f1s   = [r['f1_weighted'] * 100 for r in all_results]

    x   = np.arange(len(langs))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(langs)*1.5), 5))

    b1 = ax.bar(x - w/2, accs, w, label='Accuracy',    color='#3498db', edgecolor='white')
    b2 = ax.bar(x + w/2, f1s,  w, label='F1 Weighted', color='#2ecc71', edgecolor='white')

    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance by Language')
    ax.legend()
    ax.axhline(70, color='gray', linestyle=':', linewidth=0.8)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "summary_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Summary chart -> {out}", "OK")


def save_report_txt(all_results):
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = os.path.join(RESULT_DIR, "evaluation_report.txt")

    lines = []
    lines.append("=" * 60)
    lines.append("  SPEECH EMOTION RECOGNITION — EVALUATION REPORT")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Summary table
    lines.append("\nSUMMARY")
    lines.append("-" * 60)
    lines.append(f"{'Language':<12} {'Samples':>8} {'Classes':>8} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>12}")
    lines.append("-" * 60)
    for r in all_results:
        lines.append(
            f"{r['lang']:<12} {r['n_samples']:>8} {r['n_classes']:>8} "
            f"{r['acc']*100:>9.2f}% {r['f1_macro']*100:>9.2f}% {r['f1_weighted']*100:>11.2f}%"
        )
    if all_results:
        avg_acc = np.mean([r['acc'] for r in all_results])
        avg_f1  = np.mean([r['f1_weighted'] for r in all_results])
        lines.append("-" * 60)
        lines.append(f"{'AVERAGE':<12} {'':>8} {'':>8} {avg_acc*100:>9.2f}% {'':>9} {avg_f1*100:>11.2f}%")

    # Detail per language
    for r in all_results:
        lines.append(f"\n\n{'=' * 60}")
        lines.append(f"  {r['lang'].upper()}")
        lines.append(f"{'=' * 60}")
        lines.append(r['report'])

    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    log(f"Report saved  -> {out}", "OK")
    return out

# ==========================================
# MAIN
# ==========================================
def run_evaluation(langs):
    start = time.time()

    print("\n" + "=" * 60)
    print("  SPEECH EMOTION RECOGNITION — AUTO EVALUATION")
    print(f"  Languages: {', '.join(langs)}")
    print("=" * 60)

    device = get_device()
    log(f"Device: {'GPU ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", "OK")

    all_results = []

    for lang in langs:
        print(f"\n[>>] Evaluating: {lang} ...")
        res = evaluate_lang(lang, device)
        if res is None:
            continue

        print_lang_result(res)
        save_confusion_matrix(res)
        save_per_class_bar(res)
        save_confidence_distribution(res)
        all_results.append(res)

    if not all_results:
        print("\n[XX] ไม่พบโมเดลใดเลย — รัน data_pipeline.py --stage train ก่อน")
        return

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Language':<12} {'Acc':>8} {'F1-Mac':>8} {'F1-Wt':>8} {'Samples':>8}")
    print("  " + "-" * 48)
    for r in all_results:
        print(f"  {r['lang']:<12} {r['acc']*100:>7.2f}% {r['f1_macro']*100:>7.2f}% {r['f1_weighted']*100:>7.2f}% {r['n_samples']:>8}")

    avg_acc = np.mean([r['acc'] for r in all_results])
    avg_f1  = np.mean([r['f1_weighted'] for r in all_results])
    print("  " + "-" * 48)
    print(f"  {'AVERAGE':<12} {avg_acc*100:>7.2f}%          {avg_f1*100:>7.2f}%")

    save_summary_chart(all_results)
    report_path = save_report_txt(all_results)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Evaluation for Speech Emotion Recognition")
    parser.add_argument('--lang', action='append', dest='langs',
                        choices=TARGET_LANGUAGES,
                        help='ระบุภาษาที่ต้องการทดสอบ (ใช้ได้หลายครั้ง, ไม่ระบุ = ทุกภาษา)')
    args = parser.parse_args()

    langs = args.langs if args.langs else TARGET_LANGUAGES
    run_evaluation(langs)
