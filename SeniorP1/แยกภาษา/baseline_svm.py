"""
baseline_svm.py
===============
เปรียบเทียบ Classical SVM + MFCC กับ ResNet (Deep Learning)
ผลลัพธ์บันทึกใน: results/baseline_comparison/
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
FEAT_DIR = os.path.join(BASE_DIR, "extracted_features")
OUT_DIR  = os.path.join(BASE_DIR, "results", "baseline_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

LANGUAGES = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']
LANG_LABEL = {'Chinese': 'Chinese', 'Japan': 'Japanese', 'Korean': 'Korean',
               'Thai': 'Thai', 'English': 'English'}

# Accuracy จาก ResNet (deep learning) ที่ train ไว้แล้ว
RESNET_ACC = {
    'Chinese': 83.83,
    'Japan':   88.48,
    'Korean':  66.22,
    'Thai':    88.45,
    'English': 73.56,
}

DPI = 200
plt.rcParams.update({'font.size': 11, 'figure.dpi': DPI, 'savefig.dpi': DPI})


def log(msg):
    print(f"  {msg}")


def load_features(lang, T=130, max_samples=5000):
    """โหลด mel-spectrogram npy, flatten เป็น feature vector"""
    d = os.path.join(FEAT_DIR, lang)
    if not os.path.exists(d):
        return None, None, None

    files = [f for f in os.listdir(d) if f.endswith('.npy')]
    grouped = defaultdict(list)
    for f in files:
        grouped[f.split('_')[0]].append(os.path.join(d, f))

    X, y = [], []
    rng = np.random.RandomState(42)
    emotions = sorted(grouped.keys())
    label_map = {e: i for i, e in enumerate(emotions)}

    for emo, paths in grouped.items():
        n = min(len(paths), max_samples // len(emotions) + 1)
        chosen = rng.choice(paths, n, replace=False)
        for p in chosen:
            arr = np.load(p)
            if arr.shape[1] >= T:
                arr = arr[:, :T]
            else:
                arr = np.pad(arr, ((0, 0), (0, T - arr.shape[1])), mode='edge')
            X.append(arr.flatten())
            y.append(label_map[emo])

    return np.array(X, dtype=np.float32), np.array(y), emotions


def extract_mfcc_stats(X_flat, n_mels=128, T=130):
    """
    จาก flat mel-spec (128×130), คำนวณ MFCC statistics:
    mean, std, max, min ต่อ mel-bin → 4×128 = 512 features
    """
    X_mel = X_flat.reshape(-1, n_mels, T)
    feat_mean = X_mel.mean(axis=2)       # (N, 128)
    feat_std  = X_mel.std(axis=2)        # (N, 128)
    feat_max  = X_mel.max(axis=2)        # (N, 128)
    feat_delta = np.diff(X_mel, axis=2).mean(axis=2)  # (N, 128) temporal delta
    return np.hstack([feat_mean, feat_std, feat_max, feat_delta])  # (N, 512)


def train_and_eval(lang):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.decomposition import PCA

    log(f"โหลด {lang} ...")
    X_flat, y, emotions = load_features(lang, max_samples=6000)
    if X_flat is None:
        return None

    # MFCC-style feature extraction
    X_feat = extract_mfcc_stats(X_flat)

    X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y,
                                               test_size=0.15,
                                               random_state=42,
                                               stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # PCA ลดมิติเพื่อความเร็ว
    pca = PCA(n_components=min(100, X_tr.shape[0] - 1, X_tr.shape[1]),
              random_state=42)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)

    results = {}

    # ── 1. SVM (RBF) ──
    log(f"  SVM ...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_tr_pca, y_tr)
    pred = svm.predict(X_te_pca)
    acc = accuracy_score(y_te, pred) * 100
    f1  = f1_score(y_te, pred, average='weighted') * 100
    results['SVM (RBF)'] = {'acc': acc, 'f1': f1,
                             'report': classification_report(y_te, pred,
                                        target_names=emotions, zero_division=0)}
    log(f"    SVM Accuracy: {acc:.2f}%")

    # ── 2. Random Forest ──
    log(f"  Random Forest ...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr_pca, y_tr)
    pred = rf.predict(X_te_pca)
    acc = accuracy_score(y_te, pred) * 100
    f1  = f1_score(y_te, pred, average='weighted') * 100
    results['Random Forest'] = {'acc': acc, 'f1': f1,
                                 'report': classification_report(y_te, pred,
                                            target_names=emotions, zero_division=0)}
    log(f"    RF Accuracy: {acc:.2f}%")

    # ── 3. Gradient Boosting ──
    log(f"  Gradient Boosting ...")
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=5, random_state=42)
    gb.fit(X_tr_pca, y_tr)
    pred = gb.predict(X_te_pca)
    acc = accuracy_score(y_te, pred) * 100
    f1  = f1_score(y_te, pred, average='weighted') * 100
    results['Gradient Boost'] = {'acc': acc, 'f1': f1,
                                  'report': classification_report(y_te, pred,
                                             target_names=emotions, zero_division=0)}
    log(f"    GB Accuracy: {acc:.2f}%")

    return results


def plot_comparison(all_results):
    """Bar chart เปรียบเทียบทุก model ทุกภาษา"""
    models = ['SVM (RBF)', 'Random Forest', 'Gradient Boost', 'ResNet (ours)']
    colors = ['#3498DB', '#2ECC71', '#E67E22', '#E74C3C']

    langs_done = [l for l in LANGUAGES if l in all_results]
    n = len(langs_done)
    x = np.arange(n)
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n), 7))

    for i, (model, color) in enumerate(zip(models, colors)):
        accs = []
        for lang in langs_done:
            if model == 'ResNet (ours)':
                accs.append(RESNET_ACC.get(lang, 0))
            else:
                accs.append(all_results[lang].get(model, {}).get('acc', 0))
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, accs, width, label=model, color=color,
                      edgecolor='white', linewidth=0.8, alpha=0.9)
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABEL[l] for l in langs_done], fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title('Model Comparison: Classical ML vs Deep Learning (ResNet)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=70, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.4, linewidth=1)

    out_path = os.path.join(OUT_DIR, 'comparison_accuracy.png')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    log(f"บันทึก {out_path}")


def write_report(all_results):
    lines = []
    lines.append("=" * 65)
    lines.append("  BASELINE COMPARISON REPORT")
    lines.append("  Classical ML vs Deep Learning (ResNet)")
    lines.append("=" * 65)

    header = f"{'Language':<12} {'SVM':>8} {'RF':>8} {'GB':>8} {'ResNet':>8}"
    lines.append(header)
    lines.append("-" * 65)

    for lang in LANGUAGES:
        if lang not in all_results:
            continue
        r = all_results[lang]
        svm = r.get('SVM (RBF)', {}).get('acc', 0)
        rf  = r.get('Random Forest', {}).get('acc', 0)
        gb  = r.get('Gradient Boost', {}).get('acc', 0)
        dl  = RESNET_ACC.get(lang, 0)
        best_tag = ' ← best' if dl >= max(svm, rf, gb) else ''
        lines.append(f"{LANG_LABEL[lang]:<12} {svm:>7.1f}% {rf:>7.1f}% {gb:>7.1f}% {dl:>7.1f}%{best_tag}")

    lines.append("-" * 65)
    lines.append("\nDetailed Classification Reports:")
    for lang in LANGUAGES:
        if lang not in all_results:
            continue
        lines.append(f"\n{'='*40}")
        lines.append(f"  {LANG_LABEL[lang].upper()}")
        lines.append(f"{'='*40}")
        for model, res in all_results[lang].items():
            lines.append(f"\n--- {model} (Acc={res['acc']:.2f}%) ---")
            lines.append(res['report'])

    report_path = os.path.join(OUT_DIR, 'baseline_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"บันทึก {report_path}")


if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("  BASELINE COMPARISON: Classical ML vs ResNet (Deep Learning)")
    print("=" * 65)

    all_results = {}
    for lang in LANGUAGES:
        log(f"\n[{lang}]")
        res = train_and_eval(lang)
        if res:
            all_results[lang] = res

    print("\n" + "=" * 65)
    plot_comparison(all_results)
    write_report(all_results)

    # Print summary table
    print(f"\n{'Language':<12} {'SVM':>8} {'RF':>8} {'GB':>8} {'ResNet':>8}")
    print("-" * 50)
    for lang in LANGUAGES:
        if lang not in all_results:
            continue
        r = all_results[lang]
        svm = r.get('SVM (RBF)', {}).get('acc', 0)
        rf  = r.get('Random Forest', {}).get('acc', 0)
        gb  = r.get('Gradient Boost', {}).get('acc', 0)
        dl  = RESNET_ACC.get(lang, 0)
        print(f"{LANG_LABEL[lang]:<12} {svm:>7.1f}% {rf:>7.1f}% {gb:>7.1f}% {dl:>7.1f}%")

    print(f"\n  ผลลัพธ์บันทึกที่: {OUT_DIR}")
