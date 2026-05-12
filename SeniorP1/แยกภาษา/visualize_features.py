"""
visualize_features.py
=====================
สร้าง Graphs สำหรับงานวิจัย Speech Emotion Recognition
แสดง: Mel-Spectrogram, Waveform, MFCC, Emotion Distribution
ผลลัพธ์บันทึกใน: results/visualizations/
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from collections import defaultdict

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR   = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
FEAT_DIR   = os.path.join(BASE_DIR, "extracted_features")
OUT_DIR    = os.path.join(BASE_DIR, "results", "visualizations")
os.makedirs(OUT_DIR, exist_ok=True)

LANGUAGES  = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']
LANG_LABEL = {'Chinese': 'Chinese', 'Japan': 'Japanese', 'Korean': 'Korean',
               'Thai': 'Thai', 'English': 'English'}

EMOTION_COLORS = {
    'Angry':    '#E74C3C',
    'Happy':    '#F39C12',
    'Sad':      '#3498DB',
    'Neutral':  '#95A5A6',
    'Fear':     '#8E44AD',
    'Surprise': '#1ABC9C',
    'Disgust':  '#27AE60',
}

DPI = 200
FONT_SIZE = 11
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
})

N_MELS = 128
SR     = 22050
HOP    = 512


def log(msg):
    print(f"  {msg}")


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
def load_samples(lang, n_per_emotion=5):
    """โหลด npy mel-spectrogram โดยเลือก n ตัวอย่างต่ออารมณ์"""
    d = os.path.join(FEAT_DIR, lang)
    if not os.path.exists(d):
        return {}
    files = [f for f in os.listdir(d) if f.endswith('.npy')]
    grouped = defaultdict(list)
    for f in files:
        emo = f.split('_')[0]
        grouped[emo].append(os.path.join(d, f))
    samples = {}
    for emo, paths in grouped.items():
        rng = np.random.RandomState(42)
        chosen = rng.choice(paths, min(n_per_emotion, len(paths)), replace=False)
        arrays = []
        for p in chosen:
            arr = np.load(p)
            arrays.append(arr)
        samples[emo] = arrays
    return samples


def load_mean_spectrogram(lang):
    """คำนวณ mean mel-spectrogram ต่ออารมณ์ (pad/crop to 130 cols)"""
    d = os.path.join(FEAT_DIR, lang)
    if not os.path.exists(d):
        return {}
    files = [f for f in os.listdir(d) if f.endswith('.npy')]
    grouped = defaultdict(list)
    for f in files:
        emo = f.split('_')[0]
        grouped[emo].append(os.path.join(d, f))

    T = 130
    means = {}
    for emo, paths in grouped.items():
        stack = []
        for p in paths:
            arr = np.load(p)
            if arr.shape[1] >= T:
                arr = arr[:, :T]
            else:
                arr = np.pad(arr, ((0, 0), (0, T - arr.shape[1])), mode='constant',
                             constant_values=arr.min())
            stack.append(arr)
        means[emo] = np.mean(stack, axis=0)
    return means


def get_raw_audio(lang, emotion):
    """พยายามหา raw audio สำหรับภาษา/อารมณ์ที่กำหนด"""
    try:
        import librosa
    except ImportError:
        return None, None

    dataset = os.path.join(BASE_DIR, 'dataset')

    if lang == 'English':
        # RAVDESS: 03-01-{emo}-... emo: 03=happy,04=sad,05=angry,06=fearful,07=disgust,08=surprised
        emo_map = {'Happy': '03', 'Sad': '04', 'Angry': '05',
                   'Fear': '06', 'Disgust': '07', 'Surprise': '08', 'Neutral': '01'}
        code = emo_map.get(emotion)
        if code:
            import glob
            pat = os.path.join(dataset, 'English', 'ravdess', '**', f'03-01-{code}-*.wav')
            found = glob.glob(pat, recursive=True)
            if found:
                y, sr = librosa.load(found[0], sr=SR)
                return y, sr

    elif lang == 'Korean':
        import glob
        pat = os.path.join(dataset, 'Korean', 'hi_kia', 'labeled', f'hikia_{emotion}_*.wav')
        found = glob.glob(pat)
        if found:
            y, sr = librosa.load(found[0], sr=SR)
            return y, sr

    elif lang == 'Thai':
        import glob, json
        label_file = os.path.join(dataset, 'Thai', 'emotion_label.json')
        if os.path.exists(label_file):
            with open(label_file, encoding='utf-8') as f:
                labels = json.load(f)
            # values are list of dicts: [{'assigned_emo': 'Neutral', ...}]
            # filename uses 'con' but actual file uses 'clip'
            for fname, emo_list in labels.items():
                if not isinstance(emo_list, list) or not emo_list:
                    continue
                assigned = emo_list[0].get('assigned_emo', '')
                if assigned != emotion:
                    continue
                # file on disk uses 'clip' not 'con'
                disk_name = fname.replace('_con_', '_clip_')
                search = os.path.join(dataset, 'Thai', 'studio1-10', '**', disk_name)
                found = glob.glob(search, recursive=True)
                if found:
                    y, sr = librosa.load(found[0], sr=SR)
                    return y, sr

    return None, None


# ──────────────────────────────────────────────
# FIGURE 1: Sample Mel-Spectrogram per Emotion (per language)
# ──────────────────────────────────────────────
def fig1_sample_melspec_per_language():
    log("Figure 1: Sample Mel-Spectrograms per Language ...")
    for lang in LANGUAGES:
        samples = load_samples(lang, n_per_emotion=1)
        if not samples:
            continue
        emotions = sorted(samples.keys())
        n_emo = len(emotions)
        if n_emo == 0:
            continue

        fig, axes = plt.subplots(1, n_emo, figsize=(3.5 * n_emo, 4))
        if n_emo == 1:
            axes = [axes]

        fig.suptitle(f'{LANG_LABEL[lang]} — Sample Mel-Spectrograms per Emotion',
                     fontsize=14, fontweight='bold', y=1.02)

        for ax, emo in zip(axes, emotions):
            arr = samples[emo][0]
            im = ax.imshow(arr, aspect='auto', origin='lower',
                           cmap='magma', interpolation='nearest',
                           vmin=-80, vmax=0)
            ax.set_title(emo, color=EMOTION_COLORS.get(emo, 'black'), fontweight='bold')
            ax.set_xlabel('Time Frames', fontsize=9)
            ax.set_ylabel('Mel Bin', fontsize=9) if ax == axes[0] else ax.set_yticks([])

        plt.colorbar(im, ax=axes[-1], label='dB', shrink=0.8)
        out_path = os.path.join(OUT_DIR, f'fig1_{lang.lower()}_sample_melspec.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        log(f"  บันทึก {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# FIGURE 2: Mean Mel-Spectrogram per Emotion (per language)
# ──────────────────────────────────────────────
def fig2_mean_melspec_per_language():
    log("Figure 2: Mean Mel-Spectrograms per Language ...")
    for lang in LANGUAGES:
        means = load_mean_spectrogram(lang)
        if not means:
            continue
        emotions = sorted(means.keys())
        n_emo = len(emotions)

        fig, axes = plt.subplots(1, n_emo, figsize=(3.5 * n_emo, 4))
        if n_emo == 1:
            axes = [axes]

        fig.suptitle(f'{LANG_LABEL[lang]} — Mean Mel-Spectrogram per Emotion',
                     fontsize=14, fontweight='bold', y=1.02)

        for ax, emo in zip(axes, emotions):
            arr = means[emo]
            im = ax.imshow(arr, aspect='auto', origin='lower',
                           cmap='viridis', interpolation='nearest',
                           vmin=-80, vmax=0)
            ax.set_title(emo, color=EMOTION_COLORS.get(emo, 'black'), fontweight='bold')
            ax.set_xlabel('Time Frames', fontsize=9)
            ax.set_ylabel('Mel Bin', fontsize=9) if ax == axes[0] else ax.set_yticks([])

        plt.colorbar(im, ax=axes[-1], label='dB (mean)', shrink=0.8)
        out_path = os.path.join(OUT_DIR, f'fig2_{lang.lower()}_mean_melspec.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        log(f"  บันทึก {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# FIGURE 3: Cross-Language Comparison (same emotion, all 5 languages)
# ──────────────────────────────────────────────
def fig3_crosslang_comparison():
    log("Figure 3: Cross-Language Emotion Comparison ...")
    common_emotions = ['Angry', 'Happy', 'Sad']
    all_means = {lang: load_mean_spectrogram(lang) for lang in LANGUAGES}

    for emo in common_emotions:
        rows = [(lang, all_means[lang][emo]) for lang in LANGUAGES if emo in all_means[lang]]
        if len(rows) < 2:
            continue

        n = len(rows)
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
        if n == 1:
            axes = [axes]

        fig.suptitle(f'Cross-Language Comparison — {emo} Emotion (Mean Mel-Spectrogram)',
                     fontsize=13, fontweight='bold', y=1.02)

        # Shared colorbar range
        vmin = min(arr.min() for _, arr in rows)
        vmax = 0

        for ax, (lang, arr) in zip(axes, rows):
            im = ax.imshow(arr, aspect='auto', origin='lower',
                           cmap='inferno', interpolation='nearest',
                           vmin=vmin, vmax=vmax)
            ax.set_title(LANG_LABEL[lang], fontweight='bold')
            ax.set_xlabel('Time Frames', fontsize=9)
            ax.set_ylabel('Mel Bin', fontsize=9) if ax == axes[0] else ax.set_yticks([])

        plt.colorbar(im, ax=axes[-1], label='dB (mean)', shrink=0.8)
        safe_emo = emo.lower()
        out_path = os.path.join(OUT_DIR, f'fig3_crosslang_{safe_emo}.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        log(f"  บันทึก {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# FIGURE 4: Emotion Distribution (bar chart per language)
# ──────────────────────────────────────────────
def fig4_emotion_distribution():
    log("Figure 4: Emotion Distribution per Language ...")
    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(4.5 * len(LANGUAGES), 5))
    fig.suptitle('Emotion Distribution per Language', fontsize=14, fontweight='bold')

    for ax, lang in zip(axes, LANGUAGES):
        d = os.path.join(FEAT_DIR, lang)
        if not os.path.exists(d):
            ax.set_visible(False)
            continue
        files = [f for f in os.listdir(d) if f.endswith('.npy')]
        from collections import Counter
        counts = Counter(f.split('_')[0] for f in files)
        emotions = sorted(counts.keys())
        vals     = [counts[e] for e in emotions]
        colors   = [EMOTION_COLORS.get(e, '#AAB7B8') for e in emotions]

        bars = ax.bar(range(len(emotions)), vals, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_xticks(range(len(emotions)))
        ax.set_xticklabels(emotions, rotation=30, ha='right', fontsize=9)
        ax.set_title(LANG_LABEL[lang], fontweight='bold')
        ax.set_ylabel('Number of Samples')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(val), ha='center', va='bottom', fontsize=8, fontweight='bold')

    out_path = os.path.join(OUT_DIR, 'fig4_emotion_distribution.png')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    log(f"  บันทึก {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# FIGURE 5: Mean Mel Energy per Frequency Band (per language)
# ──────────────────────────────────────────────
def fig5_mean_energy_profile():
    log("Figure 5: Mean Mel-Band Energy per Emotion per Language ...")

    for lang in LANGUAGES:
        means = load_mean_spectrogram(lang)
        if not means:
            continue
        emotions = sorted(means.keys())

        fig, ax = plt.subplots(figsize=(8, 5))
        freq_bins = np.arange(N_MELS)

        for emo in emotions:
            arr = means[emo]                  # (128, T)
            energy_per_bin = arr.mean(axis=1) # avg across time
            color = EMOTION_COLORS.get(emo, '#AAB7B8')
            ax.plot(freq_bins, energy_per_bin, label=emo, color=color, linewidth=1.8, alpha=0.85)

        ax.set_title(f'{LANG_LABEL[lang]} — Mean Mel-Band Energy per Emotion',
                     fontweight='bold', fontsize=13)
        ax.set_xlabel('Mel Frequency Bin', fontsize=11)
        ax.set_ylabel('Mean Energy (dB)', fontsize=11)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        out_path = os.path.join(OUT_DIR, f'fig5_{lang.lower()}_energy_profile.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        log(f"  บันทึก {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# FIGURE 6: Waveform + Mel-Spectrogram (raw audio, where available)
# ──────────────────────────────────────────────
def fig6_waveform_melspec():
    log("Figure 6: Waveform + Mel-Spectrogram (raw audio) ...")
    try:
        import librosa
        import librosa.display
    except ImportError:
        log("  librosa ไม่พบ — ข้าม Figure 6")
        return

    lang_emotions = {
        'English': ['Angry', 'Happy', 'Sad', 'Neutral'],
        'Korean':  ['Angry', 'Happy', 'Sad', 'Neutral'],
        'Thai':    ['Angry', 'Happy', 'Sad', 'Neutral'],
    }

    for lang, emotions in lang_emotions.items():
        fig = plt.figure(figsize=(4 * len(emotions), 7))
        fig.suptitle(f'{LANG_LABEL[lang]} — Waveform & Mel-Spectrogram',
                     fontsize=14, fontweight='bold')
        outer = gridspec.GridSpec(2, len(emotions), figure=fig, hspace=0.45, wspace=0.3)

        any_plotted = False
        for col, emo in enumerate(emotions):
            y, sr = get_raw_audio(lang, emo)
            if y is None:
                continue
            any_plotted = True

            # ── Waveform ──
            ax_wave = fig.add_subplot(outer[0, col])
            t = np.linspace(0, len(y) / sr, len(y))
            ax_wave.plot(t, y, color=EMOTION_COLORS.get(emo, '#2C3E50'), linewidth=0.5, alpha=0.8)
            ax_wave.set_title(emo, color=EMOTION_COLORS.get(emo, 'black'), fontweight='bold')
            ax_wave.set_xlabel('Time (s)', fontsize=8)
            ax_wave.set_ylabel('Amplitude' if col == 0 else '', fontsize=8)
            ax_wave.spines['top'].set_visible(False)
            ax_wave.spines['right'].set_visible(False)
            ax_wave.set_xlim(0, t[-1])

            # ── Mel-Spectrogram ──
            ax_mel = fig.add_subplot(outer[1, col])
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                                  hop_length=HOP, fmax=sr // 2)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            img = librosa.display.specshow(mel_db, sr=sr, hop_length=HOP,
                                           x_axis='time', y_axis='mel',
                                           ax=ax_mel, cmap='magma')
            ax_mel.set_xlabel('Time (s)', fontsize=8)
            ax_mel.set_ylabel('Hz' if col == 0 else '', fontsize=8)

        if any_plotted:
            out_path = os.path.join(OUT_DIR, f'fig6_{lang.lower()}_waveform_melspec.png')
            fig.savefig(out_path, bbox_inches='tight')
            log(f"  บันทึก {os.path.basename(out_path)}")
        plt.close(fig)


# ──────────────────────────────────────────────
# FIGURE 7: All-Language Summary Grid
#   5 rows (languages) × 3 cols (Angry / Happy / Sad) — Mean Mel-Spec
# ──────────────────────────────────────────────
def fig7_summary_grid():
    log("Figure 7: Summary Grid (5 languages × 3 emotions) ...")

    emotions = ['Angry', 'Happy', 'Sad']
    all_means = {lang: load_mean_spectrogram(lang) for lang in LANGUAGES}

    n_row = len(LANGUAGES)
    n_col = len(emotions)
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.5 * n_col, 3.2 * n_row))
    fig.suptitle('Mean Mel-Spectrogram — 5 Languages × 3 Emotions',
                 fontsize=15, fontweight='bold', y=1.01)

    # Column titles
    for col, emo in enumerate(emotions):
        axes[0, col].set_title(emo,
                               color=EMOTION_COLORS.get(emo, 'black'),
                               fontsize=13, fontweight='bold', pad=10)

    # Row labels
    for row, lang in enumerate(LANGUAGES):
        axes[row, 0].set_ylabel(LANG_LABEL[lang], fontsize=11, fontweight='bold',
                                rotation=90, labelpad=8)

    for row, lang in enumerate(LANGUAGES):
        for col, emo in enumerate(emotions):
            ax = axes[row, col]
            means = all_means[lang]
            if emo in means:
                arr = means[emo]
                ax.imshow(arr, aspect='auto', origin='lower',
                          cmap='magma', interpolation='bilinear',
                          vmin=-80, vmax=0)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_facecolor('#F0F0F0')
                ax.set_xticks([])
                ax.set_yticks([])

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap='magma', norm=Normalize(vmin=-80, vmax=0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, -1], shrink=0.6, pad=0.02)
    cbar.set_label('dB (mean)', fontsize=10)

    out_path = os.path.join(OUT_DIR, 'fig7_summary_grid.png')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    log(f"  บันทึก {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# FIGURE 8: Mel-Spectrogram Variance per Language
#   แสดงความแปรปรวนของ mel-spec ในแต่ละอารมณ์
# ──────────────────────────────────────────────
def fig8_variance_heatmap():
    log("Figure 8: Mel-Spectrogram Variance per Language ...")

    T = 130
    for lang in LANGUAGES:
        d = os.path.join(FEAT_DIR, lang)
        if not os.path.exists(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith('.npy')]
        grouped = defaultdict(list)
        for f in files:
            grouped[f.split('_')[0]].append(os.path.join(d, f))

        emotions = sorted(grouped.keys())
        n_emo = len(emotions)
        if n_emo == 0:
            continue

        fig, axes = plt.subplots(1, n_emo, figsize=(3.5 * n_emo, 4))
        if n_emo == 1:
            axes = [axes]
        fig.suptitle(f'{LANG_LABEL[lang]} — Mel-Spectrogram Variance per Emotion',
                     fontsize=13, fontweight='bold', y=1.02)

        for ax, emo in zip(axes, emotions):
            stack = []
            rng = np.random.RandomState(42)
            paths = rng.choice(grouped[emo], min(200, len(grouped[emo])), replace=False)
            for p in paths:
                arr = np.load(p)
                if arr.shape[1] >= T:
                    arr = arr[:, :T]
                else:
                    arr = np.pad(arr, ((0, 0), (0, T - arr.shape[1])),
                                 mode='constant', constant_values=arr.min())
                stack.append(arr)
            var_map = np.std(stack, axis=0)
            im = ax.imshow(var_map, aspect='auto', origin='lower',
                           cmap='hot', interpolation='nearest')
            ax.set_title(emo, color=EMOTION_COLORS.get(emo, 'black'), fontweight='bold')
            ax.set_xlabel('Time Frames', fontsize=9)
            ax.set_ylabel('Mel Bin', fontsize=9) if ax == axes[0] else ax.set_yticks([])

        plt.colorbar(im, ax=axes[-1], label='Std Dev', shrink=0.8)
        out_path = os.path.join(OUT_DIR, f'fig8_{lang.lower()}_variance.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        log(f"  บันทึก {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# FIGURE 9: MFCC Profile per Emotion (computed from npy approximation)
# ──────────────────────────────────────────────
def fig9_mfcc_per_language():
    log("Figure 9: MFCC-like Profile per Language ...")
    try:
        from scipy.fft import dct
    except ImportError:
        log("  scipy ไม่พบ — ข้าม Figure 9")
        return

    T = 130
    N_MFCC = 20

    for lang in LANGUAGES:
        means = load_mean_spectrogram(lang)
        if not means:
            continue
        emotions = sorted(means.keys())

        # Compute DCT of mean mel-spectrogram (approximates MFCC)
        fig, axes = plt.subplots(1, len(emotions), figsize=(3.5 * len(emotions), 4))
        if len(emotions) == 1:
            axes = [axes]
        fig.suptitle(f'{LANG_LABEL[lang]} — MFCC-Equivalent Coefficients per Emotion',
                     fontsize=13, fontweight='bold', y=1.02)

        for ax, emo in zip(axes, emotions):
            mel = means[emo]  # (128, T)
            # Pad/crop
            if mel.shape[1] > T:
                mel = mel[:, :T]
            else:
                mel = np.pad(mel, ((0, 0), (0, T - mel.shape[1])), mode='edge')
            # Convert dB back to power, then DCT along mel axis → MFCC
            power = 10 ** (mel / 10.0)
            mfcc = dct(power, type=2, axis=0, norm='ortho')[:N_MFCC, :]  # (20, T)
            im = ax.imshow(mfcc, aspect='auto', origin='lower',
                           cmap='RdBu_r', interpolation='nearest')
            ax.set_title(emo, color=EMOTION_COLORS.get(emo, 'black'), fontweight='bold')
            ax.set_xlabel('Time Frames', fontsize=9)
            ax.set_ylabel('MFCC Coeff', fontsize=9) if ax == axes[0] else ax.set_yticks([])

        plt.colorbar(im, ax=axes[-1], label='Coefficient Value', shrink=0.8)
        out_path = os.path.join(OUT_DIR, f'fig9_{lang.lower()}_mfcc.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        log(f"  บันทึก {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# FIGURE 10: Dataset Overview Summary Table
# ──────────────────────────────────────────────
def fig10_dataset_summary():
    log("Figure 10: Dataset Summary Table ...")
    from collections import Counter

    rows = []
    for lang in LANGUAGES:
        d = os.path.join(FEAT_DIR, lang)
        if not os.path.exists(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith('.npy')]
        counts = Counter(f.split('_')[0] for f in files)
        rows.append((LANG_LABEL[lang], counts, sum(counts.values())))

    all_emos = sorted({e for _, c, _ in rows for e in c.keys()})

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(all_emos)), 0.5 * len(rows) + 3))
    ax.axis('off')

    # Build table data
    col_labels = ['Language'] + all_emos + ['Total']
    table_data = []
    for lang_name, counts, total in rows:
        row = [lang_name] + [str(counts.get(e, '-')) for e in all_emos] + [str(total)]
        table_data.append(row)

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 2.0)

    # Style header
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')

    # Style emotion columns
    for j, emo in enumerate(all_emos, start=1):
        color = EMOTION_COLORS.get(emo, '#BDC3C7')
        tbl[0, j].set_facecolor(color)
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        bg = '#ECF0F1' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
        tbl[i, len(col_labels) - 1].set_text_props(fontweight='bold')

    ax.set_title('Dataset Summary — Number of Samples per Language and Emotion',
                 fontsize=13, fontweight='bold', pad=20)

    out_path = os.path.join(OUT_DIR, 'fig10_dataset_summary.png')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    log(f"  บันทึก {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# FIGURE 11: t-SNE Feature Embedding (per language + cross-language)
# ──────────────────────────────────────────────
def fig11_tsne():
    log("Figure 11: t-SNE Feature Embeddings ...")
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log("  scikit-learn ไม่พบ — ข้าม Figure 11")
        return

    T = 130
    MAX_PER_EMO = 150   # จำกัดเพื่อความเร็ว

    # ─── 11a: t-SNE per language ───
    for lang in LANGUAGES:
        d = os.path.join(FEAT_DIR, lang)
        if not os.path.exists(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith('.npy')]
        grouped = defaultdict(list)
        for f in files:
            grouped[f.split('_')[0]].append(os.path.join(d, f))

        X, y_labels = [], []
        rng = np.random.RandomState(42)
        emotions = sorted(grouped.keys())
        for emo in emotions:
            paths = rng.choice(grouped[emo], min(MAX_PER_EMO, len(grouped[emo])), replace=False)
            for p in paths:
                arr = np.load(p)
                if arr.shape[1] >= T:
                    arr = arr[:, :T]
                else:
                    arr = np.pad(arr, ((0,0),(0,T-arr.shape[1])), mode='edge')
                X.append(arr.flatten())
                y_labels.append(emo)

        if len(set(y_labels)) < 2:
            continue

        X = np.array(X, dtype=np.float32)
        X = StandardScaler().fit_transform(X)
        tsne = TSNE(n_components=2, perplexity=min(30, len(X)//4),
                    random_state=42, max_iter=1000, verbose=0)
        coords = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 7))
        for emo in emotions:
            idx = [i for i, l in enumerate(y_labels) if l == emo]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       c=EMOTION_COLORS.get(emo, '#AAB7B8'),
                       label=emo, alpha=0.7, s=25, edgecolors='white', linewidths=0.3)
        ax.set_title(f'{LANG_LABEL[lang]} — t-SNE Feature Embedding',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')
        ax.legend(loc='best', fontsize=9, framealpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        out_path = os.path.join(OUT_DIR, f'fig11a_{lang.lower()}_tsne.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        log(f"  บันทึก {os.path.basename(out_path)}")

    # ─── 11b: t-SNE cross-language (Angry/Happy/Sad only) ───
    MAX_CROSS = 100
    X_cross, emo_cross, lang_cross = [], [], []
    target_emos = {'Angry', 'Happy', 'Sad'}
    rng = np.random.RandomState(42)

    for lang in LANGUAGES:
        d = os.path.join(FEAT_DIR, lang)
        if not os.path.exists(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith('.npy')]
        grouped = defaultdict(list)
        for f in files:
            emo = f.split('_')[0]
            if emo in target_emos:
                grouped[emo].append(os.path.join(d, f))
        for emo, paths in grouped.items():
            chosen = rng.choice(paths, min(MAX_CROSS, len(paths)), replace=False)
            for p in chosen:
                arr = np.load(p)
                if arr.shape[1] >= T:
                    arr = arr[:, :T]
                else:
                    arr = np.pad(arr, ((0,0),(0,T-arr.shape[1])), mode='edge')
                X_cross.append(arr.flatten())
                emo_cross.append(emo)
                lang_cross.append(lang)

    if len(X_cross) > 50:
        X_cross = np.array(X_cross, dtype=np.float32)
        X_cross = StandardScaler().fit_transform(X_cross)
        tsne2 = TSNE(n_components=2, perplexity=min(30, len(X_cross)//5),
                     random_state=42, max_iter=1000, verbose=0)
        coords2 = tsne2.fit_transform(X_cross)

        # Plot colored by emotion
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Cross-Language t-SNE Feature Embedding', fontsize=14, fontweight='bold')

        for emo in sorted(target_emos):
            idx = [i for i, e in enumerate(emo_cross) if e == emo]
            axes[0].scatter(coords2[idx, 0], coords2[idx, 1],
                            c=EMOTION_COLORS.get(emo, '#AAB7B8'),
                            label=emo, alpha=0.6, s=20, edgecolors='none')
        axes[0].set_title('Colored by Emotion', fontweight='bold')
        axes[0].set_xlabel('t-SNE Dim 1')
        axes[0].set_ylabel('t-SNE Dim 2')
        axes[0].legend(fontsize=9)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # Plot colored by language
        lang_palette = {'Chinese':'#E74C3C','Japan':'#3498DB','Korean':'#2ECC71',
                        'Thai':'#F39C12','English':'#9B59B6'}
        for lang in LANGUAGES:
            idx = [i for i, l in enumerate(lang_cross) if l == lang]
            if idx:
                axes[1].scatter(coords2[idx, 0], coords2[idx, 1],
                                c=lang_palette.get(lang, '#AAB7B8'),
                                label=LANG_LABEL[lang], alpha=0.6, s=20, edgecolors='none')
        axes[1].set_title('Colored by Language', fontweight='bold')
        axes[1].set_xlabel('t-SNE Dim 1')
        axes[1].set_ylabel('t-SNE Dim 2')
        axes[1].legend(fontsize=9)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        out_path = os.path.join(OUT_DIR, 'fig11b_crosslang_tsne.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        log(f"  บันทึก {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SPEECH EMOTION — FEATURE VISUALIZATION")
    print("=" * 60)
    print(f"  Output directory: {OUT_DIR}\n")

    fig1_sample_melspec_per_language()
    fig2_mean_melspec_per_language()
    fig3_crosslang_comparison()
    fig4_emotion_distribution()
    fig5_mean_energy_profile()
    fig6_waveform_melspec()
    fig7_summary_grid()
    fig8_variance_heatmap()
    fig9_mfcc_per_language()
    fig10_dataset_summary()
    fig11_tsne()

    print("\n" + "=" * 60)
    png_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.png')]
    print(f"  สร้างกราฟทั้งหมด: {len(png_files)} ไฟล์")
    for f in sorted(png_files):
        print(f"    {f}")
    print(f"\n  บันทึกทั้งหมดที่: {OUT_DIR}")
    print("=" * 60)
