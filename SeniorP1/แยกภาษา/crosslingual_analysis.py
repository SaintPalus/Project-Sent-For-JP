"""
crosslingual_analysis.py
========================
วิเคราะห์ความเหมือน/แตกต่างของอารมณ์ข้ามภาษา
สำหรับงานวิจัยระดับนานาชาติ

Outputs:
  1. Cross-language emotion similarity heatmap
  2. Emotion centroid distance matrix
  3. Per-emotion acoustic profile comparison (radar chart)
  4. Statistical significance test (ANOVA)
  5. Correlation matrix ระหว่างภาษา
"""

import os
import sys
import warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.spatial.distance import cosine, cdist
from scipy.stats import f_oneway

BASE_DIR = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
FEAT_DIR = os.path.join(BASE_DIR, "extracted_features")
OUT_DIR  = os.path.join(BASE_DIR, "results", "crosslingual")
os.makedirs(OUT_DIR, exist_ok=True)

LANGUAGES   = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']
LANG_LABEL  = {'Chinese':'Chinese','Japan':'Japanese','Korean':'Korean',
                'Thai':'Thai','English':'English'}
TARGET_EMOS = ['Angry', 'Happy', 'Sad', 'Neutral']

LANG_COLOR  = {'Chinese':'#E74C3C','Japan':'#3498DB','Korean':'#2ECC71',
                'Thai':'#F39C12','English':'#9B59B6'}
EMO_COLOR   = {'Angry':'#E74C3C','Happy':'#F39C12','Sad':'#3498DB',
                'Neutral':'#95A5A6','Fear':'#8E44AD','Surprise':'#1ABC9C'}

N_MELS  = 128
MAX_T   = 130
N_FEATS = 8       # acoustic features per file
MAX_PER = 200     # samples per emotion per language


def log(msg): print(f"  {msg}")


# ─────────────────────────────────────────────
#  LOAD & COMPUTE ACOUSTIC FEATURES
# ─────────────────────────────────────────────
def mel_to_acoustic(arr, T=MAX_T):
    """คำนวณ acoustic summary features จาก mel-spectrogram"""
    if arr.shape[1] >= T:
        arr = arr[:, :T]
    else:
        arr = np.pad(arr, ((0,0),(0,T-arr.shape[1])), mode='edge')

    energy      = arr.mean()
    energy_std  = arr.std()
    low_energy  = arr[:N_MELS//3, :].mean()
    mid_energy  = arr[N_MELS//3:2*N_MELS//3, :].mean()
    high_energy = arr[2*N_MELS//3:, :].mean()
    temporal_var = arr.var(axis=1).mean()
    spectral_centroid = (np.arange(N_MELS) * arr.mean(axis=1)).sum() / (arr.mean(axis=1).sum() + 1e-8)
    delta = np.abs(np.diff(arr, axis=1)).mean()
    return np.array([energy, energy_std, low_energy, mid_energy,
                     high_energy, temporal_var, spectral_centroid, delta],
                    dtype=np.float32)


FEATURE_NAMES = ['Mean Energy','Energy Std','Low Freq','Mid Freq',
                  'High Freq','Temporal Var','Spectral Centroid','Delta']


def load_emotion_features(lang, emotion, max_n=MAX_PER):
    d = os.path.join(FEAT_DIR, lang)
    if not os.path.exists(d):
        return None
    files = [f for f in os.listdir(d)
             if f.endswith('.npy') and f.split('_')[0] == emotion]
    if not files:
        return None
    rng = np.random.RandomState(42)
    chosen = rng.choice(files, min(max_n, len(files)), replace=False)
    feats = []
    for fname in chosen:
        arr = np.load(os.path.join(d, fname))
        feats.append(mel_to_acoustic(arr))
    return np.array(feats)


def load_all_features():
    """โหลด features ทุกภาษา/อารมณ์ → dict[lang][emo] = (N, 8)"""
    data = {}
    for lang in LANGUAGES:
        data[lang] = {}
        for emo in TARGET_EMOS:
            f = load_emotion_features(lang, emo)
            if f is not None and len(f) > 0:
                data[lang][emo] = f
    return data


# ─────────────────────────────────────────────
#  FIGURE 1: Cross-Language Emotion Similarity
# ─────────────────────────────────────────────
def fig_emotion_similarity(data):
    log("Figure 1: Cross-language emotion similarity ...")

    for emo in TARGET_EMOS:
        # เอา centroid ของแต่ละภาษา
        langs_avail = [l for l in LANGUAGES if emo in data[l]]
        if len(langs_avail) < 2:
            continue

        centroids = {l: data[l][emo].mean(axis=0) for l in langs_avail}

        # Cosine similarity matrix
        n = len(langs_avail)
        sim = np.zeros((n, n))
        for i, l1 in enumerate(langs_avail):
            for j, l2 in enumerate(langs_avail):
                if i == j:
                    sim[i, j] = 1.0
                else:
                    sim[i, j] = 1 - cosine(centroids[l1], centroids[l2])

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(sim, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        labels = [LANG_LABEL[l] for l in langs_avail]
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{sim[i,j]:.2f}', ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        color='white' if sim[i,j] < 0.5 else 'black')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        ax.set_title(f'"{emo}" Emotion — Cross-Language Feature Similarity',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f'similarity_{emo.lower()}.png')
        fig.savefig(out, bbox_inches='tight', dpi=200)
        plt.close(fig)
        log(f"  บันทึก {os.path.basename(out)}")


# ─────────────────────────────────────────────
#  FIGURE 2: Radar Chart (acoustic profile per emotion per language)
# ─────────────────────────────────────────────
def fig_radar_chart(data):
    log("Figure 2: Radar charts per emotion ...")
    from matplotlib.patches import Polygon

    N = N_FEATS
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for emo in TARGET_EMOS:
        langs_avail = [l for l in LANGUAGES if emo in data[l]]
        if not langs_avail:
            continue

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        # Normalize across all languages for this emotion
        all_vals = np.vstack([data[l][emo].mean(axis=0) for l in langs_avail])
        vmin, vmax = all_vals.min(axis=0), all_vals.max(axis=0)
        vrange = vmax - vmin + 1e-8

        for lang in langs_avail:
            vals = data[lang][emo].mean(axis=0)
            vals_norm = (vals - vmin) / vrange
            vals_plot = vals_norm.tolist() + [vals_norm[0]]
            color = LANG_COLOR[lang]
            ax.plot(angles, vals_plot, 'o-', linewidth=2, color=color,
                    label=LANG_LABEL[lang], alpha=0.85)
            ax.fill(angles, vals_plot, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(FEATURE_NAMES, size=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'"{emo}" Acoustic Profile per Language',
                     fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='lower right', bbox_to_anchor=(1.3, -0.1), fontsize=9)

        out = os.path.join(OUT_DIR, f'radar_{emo.lower()}.png')
        fig.savefig(out, bbox_inches='tight', dpi=200)
        plt.close(fig)
        log(f"  บันทึก {os.path.basename(out)}")


# ─────────────────────────────────────────────
#  FIGURE 3: ANOVA Statistical Significance
# ─────────────────────────────────────────────
def fig_anova(data):
    log("Figure 3: ANOVA statistical test ...")

    results = {}
    for feat_i, feat_name in enumerate(FEATURE_NAMES):
        for emo in TARGET_EMOS:
            groups = [data[l][emo][:, feat_i] for l in LANGUAGES if emo in data[l]]
            if len(groups) >= 2:
                stat, pval = f_oneway(*groups)
                results[(feat_name, emo)] = pval

    if not results:
        return

    feat_names = FEATURE_NAMES
    emo_names  = TARGET_EMOS
    matrix = np.ones((len(feat_names), len(emo_names)))
    for i, fn in enumerate(feat_names):
        for j, en in enumerate(emo_names):
            if (fn, en) in results:
                matrix[i, j] = results[(fn, en)]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(np.log10(matrix + 1e-10), cmap='RdYlGn_r', vmin=-5, vmax=0)
    ax.set_xticks(range(len(emo_names)))
    ax.set_yticks(range(len(feat_names)))
    ax.set_xticklabels(emo_names, fontsize=11)
    ax.set_yticklabels(feat_names, fontsize=10)
    for i in range(len(feat_names)):
        for j in range(len(emo_names)):
            p = matrix[i, j]
            txt = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            ax.text(j, i, txt, ha='center', va='center', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='log10(p-value)')
    ax.set_title('ANOVA: Cross-Language Differences in Acoustic Features\n'
                 '(* p<0.05, ** p<0.01, *** p<0.001)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'anova_significance.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    log(f"  บันทึก {os.path.basename(out)}")


# ─────────────────────────────────────────────
#  FIGURE 4: Within vs Between Language Distance
# ─────────────────────────────────────────────
def fig_within_between(data):
    log("Figure 4: Within-language vs between-language emotion distance ...")

    within  = defaultdict(list)
    between = defaultdict(list)

    for lang in LANGUAGES:
        emos = [e for e in TARGET_EMOS if e in data[lang]]
        for i, e1 in enumerate(emos):
            for j, e2 in enumerate(emos):
                if i >= j:
                    continue
                c1 = data[lang][e1].mean(axis=0)
                c2 = data[lang][e2].mean(axis=0)
                within[lang].append(cosine(c1, c2))

    for emo in TARGET_EMOS:
        langs_avail = [l for l in LANGUAGES if emo in data[l]]
        for i, l1 in enumerate(langs_avail):
            for j, l2 in enumerate(langs_avail):
                if i >= j:
                    continue
                c1 = data[l1][emo].mean(axis=0)
                c2 = data[l2][emo].mean(axis=0)
                between[emo].append(cosine(c1, c2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cross-Lingual Emotion Analysis', fontsize=14, fontweight='bold')

    # Within-language (emotion separation per language)
    ax = axes[0]
    langs = [l for l in LANGUAGES if within[l]]
    vals  = [np.mean(within[l]) for l in langs]
    colors = [LANG_COLOR[l] for l in langs]
    bars = ax.bar(range(len(langs)), vals, color=colors, edgecolor='white', alpha=0.9)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([LANG_LABEL[l] for l in langs], rotation=20, ha='right')
    ax.set_ylabel('Mean Cosine Distance')
    ax.set_title('Within-Language Emotion Separation\n(↑ = emotions more distinct)', fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Between-language (same emotion, different languages)
    ax2 = axes[1]
    emos = [e for e in TARGET_EMOS if between[e]]
    vals2 = [np.mean(between[e]) for e in emos]
    colors2 = [EMO_COLOR.get(e,'#AAB7B8') for e in emos]
    bars2 = ax2.bar(range(len(emos)), vals2, color=colors2, edgecolor='white', alpha=0.9)
    ax2.set_xticks(range(len(emos)))
    ax2.set_xticklabels(emos, fontsize=11)
    ax2.set_ylabel('Mean Cosine Distance')
    ax2.set_title('Cross-Language Emotion Consistency\n(↓ = emotion sounds similar across languages)', fontweight='bold')
    for bar, v in zip(bars2, vals2):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'within_between_distance.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    log(f"  บันทึก {os.path.basename(out)}")


# ─────────────────────────────────────────────
#  FIGURE 5: Summary Research Table
# ─────────────────────────────────────────────
def fig_research_summary():
    log("Figure 5: Research summary table ...")

    # Accuracy values
    mel_acc = {'Chinese':83.83,'Japan':88.48,'Korean':66.22,'Thai':88.45,'English':82.89}
    svm_acc = {'Chinese':81.78,'Japan':75.31,'Korean':79.73,'Thai':63.93,'English':60.34}

    rows = []
    for lang in LANGUAGES:
        d  = os.path.join(FEAT_DIR, lang)
        n  = len([f for f in os.listdir(d) if f.endswith('.npy')]) if os.path.exists(d) else 0
        from collections import Counter
        emos = Counter()
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.npy'):
                    emos[f.split('_')[0]] += 1
        n_emo = len(emos)
        rows.append([LANG_LABEL[lang], str(n), str(n_emo),
                     f"{mel_acc[lang]:.1f}%", f"{svm_acc[lang]:.1f}%",
                     '✓' if mel_acc[lang] > svm_acc[lang] else '✗ SVM wins'])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    cols = ['Language', 'Samples', 'Emotions', 'ResNet Acc', 'SVM Acc', 'DL Better?']
    tbl  = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 2.2)

    header_color = '#2C3E50'
    for j in range(len(cols)):
        tbl[0, j].set_facecolor(header_color)
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(rows)+1):
        bg = '#ECF0F1' if i % 2 == 0 else 'white'
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(bg)
        # highlight DL Better column
        val = rows[i-1][-1]
        if 'SVM wins' in val:
            tbl[i, len(cols)-1].set_facecolor('#FADBD8')
        else:
            tbl[i, len(cols)-1].set_facecolor('#D5F5E3')

    ax.set_title('Multilingual SER — Research Summary',
                 fontsize=14, fontweight='bold', pad=20)
    out = os.path.join(OUT_DIR, 'research_summary_table.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    log(f"  บันทึก {os.path.basename(out)}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  CROSS-LINGUAL EMOTION ANALYSIS")
    print("=" * 60)

    log("โหลด features ...")
    data = load_all_features()

    avail = {l: list(data[l].keys()) for l in LANGUAGES if data[l]}
    for lang, emos in avail.items():
        log(f"  {lang}: {emos}")

    fig_emotion_similarity(data)
    fig_radar_chart(data)
    fig_anova(data)
    fig_within_between(data)
    fig_research_summary()

    pngs = [f for f in os.listdir(OUT_DIR) if f.endswith('.png')]
    print(f"\n  สร้างกราฟ {len(pngs)} ไฟล์ → {OUT_DIR}")
    for f in sorted(pngs):
        print(f"    {f}")
