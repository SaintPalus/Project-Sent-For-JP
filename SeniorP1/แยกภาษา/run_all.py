"""
run_all.py
==========
รันทุกขั้นตอนอัตโนมัติ ตั้งแต่ต้นจนจบ (International Competition Edition)

Usage:
  python run_all.py                      # รันทุกขั้นตอน (11 steps)
  python run_all.py --skip-download      # ข้ามดาวน์โหลด (มี dataset อยู่แล้ว)
  python run_all.py --only-train         # extract + train + evaluate อย่างเดียว
  python run_all.py --skip-w2v           # ข้าม Wav2Vec2 pipeline
  python run_all.py --skip-extra-data    # ข้ามดาวน์โหลด extra data
  python run_all.py --lang Thai          # รันเฉพาะภาษาไทย
  python run_all.py --web-demo           # เปิด web demo หลังเสร็จ

Pipeline Steps:
  1.  Download Core Datasets (Korean hi_kia + English CREMA-D)
  2.  Download Extra Datasets (Korean/Japanese HuggingFace)
  3.  Feature Extraction — Mel-Spectrogram
  4.  Train ResNet Models (GPU)
  5.  Evaluate ResNet + Confidence Analysis
  6.  Train Language Detector (98%+ accuracy)
  7.  Visualization — 33+ Research Figures
  8.  Baseline SVM Comparison
  9.  Wav2Vec2 XLSR-53 Feature Extraction
  10. Wav2Vec2 MLP Training + Comparison
  11. Cross-Lingual Analysis (Similarity, Radar, ANOVA)
  [Optional] Launch Gradio Web Demo
"""

import os
import sys
import time
import argparse
import subprocess

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR   = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
SCRIPT_DIR = os.path.join(BASE_DIR, "แยกภาษา")
PYTHON     = os.path.join(os.path.dirname(BASE_DIR), "Anaconda", "python.exe")

if not os.path.exists(PYTHON):
    PYTHON = sys.executable


def header(title):
    w = 60
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def step(n, total, desc):
    print(f"\n[{n:>2}/{total}] {desc}")
    print("-" * 50)


def run(script, *args, cwd=SCRIPT_DIR, allow_fail=False):
    """รัน Python script แล้วรอจนจบ แสดง output แบบ real-time"""
    cmd = [PYTHON, script] + list(args)
    print(f"  $ python {os.path.basename(script)} {' '.join(args)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        if allow_fail:
            print(f"  [!!] Script failed (exit {result.returncode}) — ข้ามต่อ")
            return elapsed
        print(f"\n[!!] Script failed (exit {result.returncode}) — หยุดทำงาน")
        sys.exit(result.returncode)
    print(f"  [OK] เสร็จใน {elapsed:.0f}s")
    return elapsed


def check_dataset_exists():
    status = {}
    checks = {
        'Korean':  os.path.join(BASE_DIR, "dataset", "Korean", "hi_kia", "labeled"),
        'English': os.path.join(BASE_DIR, "dataset", "English", "crema_d"),
        'Thai':    os.path.join(BASE_DIR, "dataset", "Thai"),
        'Chinese': os.path.join(BASE_DIR, "dataset", "Chinese"),
        'Japan':   os.path.join(BASE_DIR, "dataset", "Japan"),
    }
    for lang, path in checks.items():
        if os.path.exists(path):
            count = sum(1 for _, _, fs in os.walk(path)
                        for f in fs if f.endswith(('.wav', '.flac', '.mp3')))
            status[lang] = count
        else:
            status[lang] = 0
    return status


def check_extra_data_exists():
    result = {}
    for lang, subdir in [('Korean', 'Korean/hf_extra'), ('Japan', 'Japan/hf_extra')]:
        d = os.path.join(BASE_DIR, "dataset", subdir)
        if os.path.exists(d):
            result[lang] = sum(1 for f in os.listdir(d) if f.endswith('.wav'))
        else:
            result[lang] = 0
    return result


def check_features_exist():
    feat_dir = os.path.join(BASE_DIR, "extracted_features")
    result = {}
    for lang in ['Chinese', 'Japan', 'Korean', 'Thai', 'English']:
        d = os.path.join(feat_dir, lang)
        result[lang] = sum(1 for f in os.listdir(d) if f.endswith('.npy')) \
            if os.path.exists(d) else 0
    return result


def check_w2v_features_exist():
    feat_dir = os.path.join(BASE_DIR, "extracted_features_w2v")
    result = {}
    for lang in ['Chinese', 'Japan', 'Korean', 'Thai', 'English']:
        d = os.path.join(feat_dir, lang)
        result[lang] = sum(1 for f in os.listdir(d) if f.endswith('.npy')) \
            if os.path.exists(d) else 0
    return result


def check_models_exist():
    model_dir = os.path.join(BASE_DIR, "models")
    result = {}
    for lang in ['Chinese', 'Japan', 'Korean', 'Thai', 'English']:
        pt = os.path.join(model_dir, f"{lang}_model.pt")
        result[lang] = os.path.exists(pt)
    return result


def check_w2v_models_exist():
    model_dir = os.path.join(BASE_DIR, "models")
    result = {}
    for lang in ['Chinese', 'Japan', 'Korean', 'Thai', 'English']:
        pt = os.path.join(model_dir, f"{lang}_w2v_model.pt")
        result[lang] = os.path.exists(pt)
    return result


def print_status_table(ds, ft, md, w2v_ft=None, w2v_md=None):
    print(f"\n  {'ภาษา':<10} {'Dataset':>10} {'MelFeat':>9} {'ResNet':>8}", end="")
    if w2v_ft:
        print(f" {'W2VFeat':>9} {'W2VModel':>9}", end="")
    print()
    print(f"  {'-'*55}")
    for lang in ['Chinese', 'Japan', 'Korean', 'Thai', 'English']:
        m = '✓' if md[lang] else '✗'
        line = f"  {lang:<10} {ds.get(lang,0):>9}f {ft[lang]:>8}f {m:>8}"
        if w2v_ft:
            wm = '✓' if (w2v_md or {}).get(lang) else '✗'
            line += f" {w2v_ft[lang]:>8}f {wm:>9}"
        print(line)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition — Full Auto Pipeline")
    parser.add_argument('--skip-download',    action='store_true',
                        help='ข้ามขั้น download core datasets')
    parser.add_argument('--skip-extra-data',  action='store_true',
                        help='ข้ามขั้น download extra data (Korean/Japanese HuggingFace)')
    parser.add_argument('--skip-extract',     action='store_true',
                        help='ข้ามขั้น mel-spectrogram feature extraction')
    parser.add_argument('--skip-train',       action='store_true',
                        help='ข้ามขั้น ResNet training')
    parser.add_argument('--skip-w2v',         action='store_true',
                        help='ข้าม Wav2Vec2 pipeline ทั้งหมด')
    parser.add_argument('--skip-crosslingual',action='store_true',
                        help='ข้าม cross-lingual analysis')
    parser.add_argument('--only-train',       action='store_true',
                        help='รันแค่ extract + train + evaluate (ข้าม download/viz/w2v)')
    parser.add_argument('--lang',             default='all',
                        help='ระบุภาษา: Thai/Chinese/Japan/Korean/English/all')
    parser.add_argument('--web-demo',         action='store_true',
                        help='เปิด Gradio web demo หลังเสร็จ pipeline')
    parser.add_argument('--share',            action='store_true',
                        help='เปิด web demo แบบ public URL (ใช้กับ --web-demo)')
    args = parser.parse_args()

    if args.only_train:
        args.skip_download   = True
        args.skip_extra_data = True
        args.skip_w2v        = True
        args.skip_crosslingual = True

    lang_args = [] if args.lang == 'all' else ['--lang', args.lang]

    # นับจำนวน steps จริง
    TOTAL = 11
    if args.skip_download:    TOTAL -= 1
    if args.skip_extra_data:  TOTAL -= 1
    if args.skip_w2v:         TOTAL -= 2
    if args.skip_crosslingual:TOTAL -= 1

    header("SPEECH EMOTION RECOGNITION — AUTO PIPELINE (v2 International)")
    print(f"  Python    : {PYTHON}")
    print(f"  BaseDir   : {BASE_DIR}")
    print(f"  ภาษา      : {args.lang}")
    print(f"  Steps     : {TOTAL} ขั้นตอน")

    t_start = time.time()
    timings = {}
    step_n  = [0]

    def next_step(desc, skip_label=None):
        step_n[0] += 1
        if skip_label:
            step(step_n[0], TOTAL, f"{desc} [ข้าม]")
            print(f"  → {skip_label}")
        else:
            step(step_n[0], TOTAL, desc)

    # ─── ตรวจสถานะปัจจุบัน ───────────────────
    print("\n  กำลังตรวจสอบสถานะ...")
    ds     = check_dataset_exists()
    ft     = check_features_exist()
    md     = check_models_exist()
    w2v_ft = check_w2v_features_exist()
    w2v_md = check_w2v_models_exist()
    extra  = check_extra_data_exists()

    print_status_table(ds, ft, md, w2v_ft, w2v_md)
    print(f"\n  Extra data: Korean={extra.get('Korean',0)} | Japan={extra.get('Japan',0)} ไฟล์")

    # ════════════════════════════════════════
    # STEP 1 — Download Core Datasets
    # ════════════════════════════════════════
    if not args.skip_download:
        next_step("Download Core Datasets (Korean hi_kia + CREMA-D)")

        if ds['Korean'] < 400:
            print("  → Korean hi_kia dataset ...")
            timings['download_korean'] = run(
                os.path.join(SCRIPT_DIR, "download_korean_data.py"))
        else:
            print(f"  → Korean: มีอยู่แล้ว {ds['Korean']} ไฟล์ ข้าม")

        if ds['English'] < 7000:
            print("  → English CREMA-D dataset ...")
            timings['download_crema'] = run(
                os.path.join(SCRIPT_DIR, "download_crema_d.py"))
        else:
            print(f"  → English CREMA-D: มีอยู่แล้ว {ds['English']} ไฟล์ ข้าม")
    else:
        next_step("Download Core Datasets", "--skip-download ข้ามขั้นนี้")

    # ════════════════════════════════════════
    # STEP 2 — Download Extra Data (Korean/Japanese HuggingFace)
    # ════════════════════════════════════════
    if not args.skip_extra_data:
        next_step("Download Extra Datasets (Korean/Japanese HuggingFace)")
        need_ko = extra.get('Korean', 0) < 500
        need_ja = extra.get('Japan',  0) < 500
        if need_ko or need_ja:
            timings['extra_data'] = run(
                os.path.join(SCRIPT_DIR, "download_extra_data.py"),
                allow_fail=True)
        else:
            print(f"  → Extra data: Korean={extra['Korean']} Japan={extra['Japan']} — มีแล้ว ข้าม")
    else:
        next_step("Download Extra Datasets", "--skip-extra-data ข้ามขั้นนี้")

    # ════════════════════════════════════════
    # STEP 3 — Mel-Spectrogram Feature Extraction
    # ════════════════════════════════════════
    if not args.skip_extract:
        next_step("Feature Extraction — Mel-Spectrogram (128×130)")
        timings['extract'] = run(
            os.path.join(SCRIPT_DIR, "data_pipeline.py"),
            "--stage", "extract", *lang_args)
    else:
        next_step("Feature Extraction — Mel-Spectrogram", "--skip-extract ข้ามขั้นนี้")

    # ════════════════════════════════════════
    # STEP 4 — Train ResNet Models
    # ════════════════════════════════════════
    if not args.skip_train:
        next_step("Train ResNet Models (PyTorch + GPU)")
        timings['train'] = run(
            os.path.join(SCRIPT_DIR, "data_pipeline.py"),
            "--stage", "train", *lang_args)
    else:
        next_step("Train ResNet Models", "--skip-train ข้ามขั้นนี้")

    # ════════════════════════════════════════
    # STEP 5 — Evaluate + Confidence Analysis
    # ════════════════════════════════════════
    next_step("Evaluate ResNet + Confidence Analysis")
    timings['evaluate'] = run(
        os.path.join(SCRIPT_DIR, "evaluate.py"),
        *lang_args)

    if args.only_train:
        _print_summary(timings, t_start)
        return

    # ════════════════════════════════════════
    # STEP 6 — Language Detector
    # ════════════════════════════════════════
    next_step("Train Language Detector (SVM, target 98%+)")
    timings['lang_detect'] = run(
        os.path.join(SCRIPT_DIR, "language_detector.py"),
        "--test")

    # ════════════════════════════════════════
    # STEP 7 — Visualization (33+ Figures)
    # ════════════════════════════════════════
    next_step("Visualization — 33+ Research Figures (Mel/t-SNE/Heatmap)")
    timings['visualize'] = run(
        os.path.join(SCRIPT_DIR, "visualize_features.py"))

    # ════════════════════════════════════════
    # STEP 8 — Baseline SVM Comparison
    # ════════════════════════════════════════
    next_step("Baseline Comparison (SVM / RF / GradientBoosting vs ResNet)")
    timings['baseline'] = run(
        os.path.join(SCRIPT_DIR, "baseline_svm.py"))

    # ════════════════════════════════════════
    # STEP 9 — Wav2Vec2 Feature Extraction
    # ════════════════════════════════════════
    if not args.skip_w2v:
        total_w2v = sum(w2v_ft.values())
        next_step("Wav2Vec2 XLSR-53 Feature Extraction (1024-dim)")

        if total_w2v < 1000:
            print("  → กำลัง extract Wav2Vec2 features (ใช้เวลานาน ~30-60 นาที) ...")
            timings['w2v_extract'] = run(
                os.path.join(SCRIPT_DIR, "wav2vec2_pipeline.py"),
                "--stage", "extract", *lang_args)
        else:
            print(f"  → W2V features มีอยู่แล้ว {total_w2v} ไฟล์ ข้าม extract")

        # ════════════════════════════════════════
        # STEP 10 — Wav2Vec2 MLP Train + Compare
        # ════════════════════════════════════════
        next_step("Wav2Vec2 MLP Training + Mel vs W2V Comparison")
        total_w2v_models = sum(1 for v in w2v_md.values() if v)
        if total_w2v_models < 3:
            timings['w2v_train'] = run(
                os.path.join(SCRIPT_DIR, "wav2vec2_pipeline.py"),
                "--stage", "train", *lang_args)
        else:
            print(f"  → W2V models มีอยู่แล้ว {total_w2v_models}/5 ภาษา ข้าม train")

        print("  → สร้างกราฟเปรียบเทียบ Mel-ResNet vs Wav2Vec2-MLP ...")
        timings['w2v_compare'] = run(
            os.path.join(SCRIPT_DIR, "wav2vec2_pipeline.py"),
            "--stage", "compare", *lang_args,
            allow_fail=True)
    else:
        next_step("Wav2Vec2 Feature Extraction", "--skip-w2v ข้ามขั้นนี้")
        next_step("Wav2Vec2 MLP Training",       "--skip-w2v ข้ามขั้นนี้")

    # ════════════════════════════════════════
    # STEP 11 — Cross-Lingual Analysis
    # ════════════════════════════════════════
    if not args.skip_crosslingual:
        next_step("Cross-Lingual Analysis (Similarity / Radar / ANOVA)")
        timings['crosslingual'] = run(
            os.path.join(SCRIPT_DIR, "crosslingual_analysis.py"),
            allow_fail=True)
    else:
        next_step("Cross-Lingual Analysis", "--skip-crosslingual ข้ามขั้นนี้")

    _print_summary(timings, t_start)

    # ════════════════════════════════════════
    # OPTIONAL — Web Demo
    # ════════════════════════════════════════
    if args.web_demo:
        header("เปิด Gradio Web Demo")
        share_args = ['--share'] if args.share else []
        print("  → กำลังเปิด web demo ที่ http://localhost:7860")
        print("  → กด Ctrl+C เพื่อหยุด")
        subprocess.run(
            [PYTHON, os.path.join(SCRIPT_DIR, "web_demo.py")] + share_args,
            cwd=SCRIPT_DIR)


def _print_summary(timings, t_start):
    total = time.time() - t_start
    header("เสร็จสิ้นทุกขั้นตอน")
    print(f"\n  {'ขั้นตอน':<28} {'เวลา':>8}")
    print(f"  {'-'*38}")
    labels = {
        'download_korean': 'Download Korean hi_kia',
        'download_crema':  'Download CREMA-D',
        'extra_data':      'Download Extra Data',
        'extract':         'Mel Feature Extraction',
        'train':           'ResNet Training',
        'evaluate':        'Evaluation',
        'lang_detect':     'Language Detector',
        'visualize':       'Visualization (33+ fig)',
        'baseline':        'Baseline SVM',
        'w2v_extract':     'Wav2Vec2 Extract',
        'w2v_train':       'Wav2Vec2 MLP Train',
        'w2v_compare':     'Mel vs W2V Compare',
        'crosslingual':    'Cross-Lingual Analysis',
    }
    for key, label in labels.items():
        if key in timings:
            t = timings[key]
            m, s = divmod(int(t), 60)
            print(f"  {label:<28} {m:>3}m {s:02d}s")
    m, s = divmod(int(total), 60)
    print(f"  {'-'*38}")
    print(f"  {'รวมทั้งหมด':<28} {m:>3}m {s:02d}s")

    results_dir = os.path.join(BASE_DIR, "results")
    report = os.path.join(results_dir, "evaluation_report.txt")
    if os.path.exists(report):
        print(f"\n  ผลการประเมิน : {report}")

    baseline_report = os.path.join(results_dir, "baseline_comparison", "baseline_report.txt")
    if os.path.exists(baseline_report):
        print(f"  Baseline     : {baseline_report}")

    crosslingual_dir = os.path.join(results_dir, "crosslingual")
    if os.path.exists(crosslingual_dir):
        n_figs = len([f for f in os.listdir(crosslingual_dir) if f.endswith('.png')])
        print(f"  Cross-lingual: {crosslingual_dir} ({n_figs} figures)")

    print(f"\n  คำสั่งต่อไป:")
    print(f"  python web_demo.py                    # web demo (localhost)")
    print(f"  python web_demo.py --share             # public URL")
    print(f"  python demo_live.py --lang Thai        # terminal demo")
    print(f"  python demo_live.py --lang auto        # auto-detect language")
    print(f"  python wav2vec2_pipeline.py --stage all  # W2V only")
    print("=" * 60)


if __name__ == "__main__":
    main()
