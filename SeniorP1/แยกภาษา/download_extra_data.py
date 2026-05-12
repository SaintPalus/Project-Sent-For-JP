"""
download_extra_data.py
======================
ขยาย dataset สำหรับ Korean และ Japanese

Strategy (2 ขั้นตอน):
  1. ค้นหา HuggingFace Hub จริง ๆ ด้วย API (ไม่ hardcode ชื่อ)
  2. Audio augmentation เป็น fallback ที่การันตีผล
     - Pitch shift ±1, ±2 semitones
     - Time stretch 0.85×, 1.15×
     - Add Gaussian noise (SNR 20 dB)
     → Korean 488 → ~2,500+   Japanese 1,615 → ~8,000+
"""

import os
import sys
import shutil
import numpy as np
import soundfile as sf
import librosa
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

BASE_DIR  = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
KO_AUG    = os.path.join(BASE_DIR, "dataset", "Korean",  "augmented")
JA_AUG    = os.path.join(BASE_DIR, "dataset", "Japan",   "augmented")
os.makedirs(KO_AUG, exist_ok=True)
os.makedirs(JA_AUG, exist_ok=True)

EMOTION_NORM = {
    'angry':'Angry','anger':'Angry','분노':'Angry','怒り':'Angry',
    'happy':'Happy','happiness':'Happy','joy':'Happy','기쁨':'Happy',
    '喜び':'Happy','행복':'Happy','excited':'Happy',
    'sad':'Sad','sadness':'Sad','슬픔':'Sad','悲しみ':'Sad',
    'neutral':'Neutral','중립':'Neutral','普通':'Neutral','normal':'Neutral',
    'fear':'Fear','fearful':'Fear','공포':'Fear','恐れ':'Fear','anxious':'Fear',
    'surprise':'Surprise','surprised':'Surprise','놀람':'Surprise','驚き':'Surprise',
    'disgust':'Disgust','혐오':'Disgust','嫌悪':'Disgust',
}

SR_TARGET = 22050


def log(msg, level="INFO"):
    icons = {"INFO": "  ", "OK": "OK", "WARN": "!!", "ERROR": "XX", "RUN": ">>"}
    print(f"[{icons.get(level,'  ')}] {msg}", flush=True)


def save_wav(arr, sr, path):
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    peak = np.abs(arr).max()
    if peak > 0:
        arr = arr / peak * 0.9
    sf.write(path, arr, sr, subtype='PCM_16')


def load_audio(path, sr=SR_TARGET):
    try:
        arr, orig_sr = sf.read(path, always_2d=False)
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if orig_sr != sr:
            arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=sr)
        return arr, sr
    except Exception as e:
        return None, None


def norm_emotion(raw):
    raw = str(raw).lower().strip()
    if raw in EMOTION_NORM:
        return EMOTION_NORM[raw]
    for k, v in EMOTION_NORM.items():
        if k in raw:
            return v
    return None


# ════════════════════════════════════════════
#  PART 1 — ค้นหาจาก HuggingFace Hub API
# ════════════════════════════════════════════

def search_hf_datasets(language_tag, keywords=("emotion", "speech", "sentiment")):
    """ค้นหา dataset จาก HuggingFace Hub API แบบ real-time"""
    try:
        from huggingface_hub import list_datasets
        log(f"ค้นหา HuggingFace Hub สำหรับ language={language_tag} ...", "RUN")
        results = []
        for kw in keywords:
            try:
                for ds in list_datasets(search=kw, limit=30):
                    tags = getattr(ds, 'tags', []) or []
                    card = getattr(ds, 'cardData', {}) or {}
                    langs = card.get('language', [])
                    if not isinstance(langs, list):
                        langs = [langs]
                    if language_tag in langs or language_tag in tags:
                        if ds.id not in results:
                            results.append(ds.id)
            except Exception:
                pass
        log(f"  พบ {len(results)} dataset ที่น่าสนใจ: {results[:5]}", "INFO")
        return results
    except Exception as e:
        log(f"  Hub API ล้มเหลว: {e}", "WARN")
        return []


def try_hf_repos(candidates, out_dir, prefix, lang_label):
    """ลอง load HuggingFace dataset จากรายชื่อที่กำหนด"""
    try:
        from datasets import load_dataset
    except ImportError:
        log("  ไม่มี 'datasets' library — ข้าม HF download", "WARN")
        return 0

    total = 0
    for repo in candidates:
        existing = sum(1 for f in os.listdir(out_dir) if f.startswith(prefix))
        if existing > 300:
            log(f"  {prefix}: มีอยู่แล้ว {existing} ไฟล์ ข้าม", "OK")
            total += existing
            break
        try:
            log(f"  ลอง {repo} ...", "INFO")
            ds = load_dataset(repo, split="train", trust_remote_code=False)
            n = _extract_hf(ds, out_dir, prefix)
            total += n
            if n > 0:
                log(f"  {repo} → บันทึก {n} ไฟล์", "OK")
        except Exception as e:
            log(f"  {repo}: {str(e)[:80]}", "WARN")
    return total


def _extract_hf(ds, out_dir, prefix):
    audio_col   = next((c for c in ['audio','Audio','speech','wav','file']
                        if c in ds.column_names), None)
    emotion_col = next((c for c in ['emotion','Emotion','label','Label',
                                    'emotion_label','sentiment','category']
                        if c in ds.column_names), None)
    if not audio_col:
        return 0
    count = 0
    for i, row in enumerate(ds):
        emotion = norm_emotion(row.get(emotion_col, 'Unknown')) if emotion_col else 'Unknown'
        if emotion is None:
            continue
        audio = row.get(audio_col)
        if audio is None:
            continue
        arr = np.array(audio['array'], dtype=np.float32)
        sr  = audio['sampling_rate']
        fname = f"{prefix}_{i:05d}_{emotion}.wav"
        out_path = os.path.join(out_dir, fname)
        if not os.path.exists(out_path):
            save_wav(arr, sr, out_path)
        count += 1
        if count % 200 == 0:
            log(f"    {count} ไฟล์...", "INFO")
    return count


# ════════════════════════════════════════════
#  PART 2 — Audio Augmentation (Fallback)
# ════════════════════════════════════════════

AUG_OPS = [
    ("ps+1",  lambda a, sr: librosa.effects.pitch_shift(a, sr=sr, n_steps=1)),
    ("ps-1",  lambda a, sr: librosa.effects.pitch_shift(a, sr=sr, n_steps=-1)),
    ("ps+2",  lambda a, sr: librosa.effects.pitch_shift(a, sr=sr, n_steps=2)),
    ("ps-2",  lambda a, sr: librosa.effects.pitch_shift(a, sr=sr, n_steps=-2)),
    ("ts085", lambda a, sr: librosa.effects.time_stretch(a, rate=0.85)),
    ("ts115", lambda a, sr: librosa.effects.time_stretch(a, rate=1.15)),
    ("noise", lambda a, sr: _add_noise(a, snr_db=20)),
]


def _add_noise(arr, snr_db=20):
    rms = np.sqrt(np.mean(arr ** 2))
    noise_rms = rms / (10 ** (snr_db / 20))
    noise = np.random.randn(len(arr)).astype(np.float32) * noise_rms
    return arr + noise


def augment_language(src_dirs, out_dir, lang_label, target_total=3000):
    """ขยาย dataset ด้วย pitch shift / time stretch / noise"""
    # นับไฟล์ที่มีอยู่แล้วใน out_dir
    already = sum(1 for f in os.listdir(out_dir) if f.endswith('.wav'))
    if already >= target_total:
        log(f"  {lang_label}: augmented มีอยู่แล้ว {already} ไฟล์ ข้าม", "OK")
        return already

    # รวมไฟล์ต้นฉบับจากทุก src_dir
    src_files = []
    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            continue
        for root, _, files in os.walk(src_dir):
            for f in files:
                if f.lower().endswith(('.wav', '.flac', '.mp3')):
                    src_files.append(os.path.join(root, f))

    if not src_files:
        log(f"  {lang_label}: ไม่มีไฟล์ต้นฉบับ", "WARN")
        return 0

    log(f"  {lang_label}: ต้นฉบับ {len(src_files)} ไฟล์ → เป้าหมาย {target_total}", "RUN")

    count = 0
    needed = target_total - already
    ops_cycle = [op for op in AUG_OPS]

    for i, src_path in enumerate(src_files):
        if count >= needed:
            break

        # หา emotion จากชื่อไฟล์ — ลองทุก part (รองรับหลาย naming convention)
        # hikia_Angry_F5_S01_1_a → parts=['hikia','Angry','F5','S01','1','a']
        # aug_Korean_00001_ps+1_Happy → parts=[...,'Happy']
        fname = os.path.splitext(os.path.basename(src_path))[0]
        emo = None
        for part in fname.split('_'):
            emo = norm_emotion(part)
            if emo:
                break
        emo = emo or 'Unknown'

        arr, sr = load_audio(src_path, SR_TARGET)
        if arr is None or len(arr) < sr * 0.3:
            continue

        for op_name, op_fn in ops_cycle:
            if count >= needed:
                break
            try:
                aug = op_fn(arr, sr)
                aug = aug.astype(np.float32)
                out_name = f"aug_{lang_label}_{i:05d}_{op_name}_{emo}.wav"
                out_path = os.path.join(out_dir, out_name)
                if not os.path.exists(out_path):
                    save_wav(aug, sr, out_path)
                count += 1
                if count % 500 == 0:
                    log(f"    {count}/{needed} augmented ไฟล์...", "INFO")
            except Exception:
                pass

    total_now = sum(1 for f in os.listdir(out_dir) if f.endswith('.wav'))
    log(f"  {lang_label}: สร้าง {count} ไฟล์ใหม่ รวมทั้งหมด {total_now} ใน augmented/", "OK")
    return total_now


# ════════════════════════════════════════════
#  Korean
# ════════════════════════════════════════════

def download_korean():
    log("=" * 55)
    log("KOREAN — ค้นหา + Augment", "RUN")
    log("=" * 55)

    # รายชื่อที่ค้นผ่าน API + ทดลองเพิ่มเติม
    hf_extra_dir = os.path.join(BASE_DIR, "dataset", "Korean", "hf_extra")
    os.makedirs(hf_extra_dir, exist_ok=True)

    # ลองจาก HuggingFace Hub API (ค้นจริง)
    found_repos = search_hf_datasets("ko", keywords=("emotion speech korean",
                                                      "korean emotion",
                                                      "speech emotion ko"))
    # เพิ่ม fallback repos ที่น่าจะมี
    fallback = [
        "snunlp/KR-FinBERT",          # จะ fail แต่ไม่เป็นไร
        "waduda/korean_speech_emotion",
        "EzioTLive/Korean-Speech-Emotion",
        "jason9693/EmotionTTSKo",
        "seyoungsong/kss",
    ]
    all_candidates = list(dict.fromkeys(found_repos + fallback))

    hf_count = try_hf_repos(all_candidates, hf_extra_dir, "ko_hf", "Korean")

    # ─── Augmentation (guaranteed) ───────────────────
    ko_src_dirs = [
        os.path.join(BASE_DIR, "dataset", "Korean", "hi_kia", "labeled"),
        os.path.join(BASE_DIR, "dataset", "Korean", "hf_kgen"),
        os.path.join(BASE_DIR, "dataset", "Korean", "hf_extra"),
        os.path.join(BASE_DIR, "dataset", "Korean",
                     "korean-voice-emotion-dataset"),
    ]
    aug_count = augment_language(ko_src_dirs, KO_AUG, "Korean", target_total=3000)

    # สรุป
    hi_kia = _count_wav(os.path.join(BASE_DIR, "dataset", "Korean", "hi_kia", "labeled"))
    hf_kgen = _count_wav(os.path.join(BASE_DIR, "dataset", "Korean", "hf_kgen"))
    grand = hi_kia + hf_kgen + hf_count + aug_count
    log(f"Korean รวม: hi_kia={hi_kia} + hf_kgen={hf_kgen} + hf_extra={hf_count}"
        f" + augmented={aug_count} = {grand} ไฟล์", "OK")
    return aug_count + hf_count


# ════════════════════════════════════════════
#  Japanese
# ════════════════════════════════════════════

def download_japanese():
    log("=" * 55)
    log("JAPANESE — ค้นหา + Augment", "RUN")
    log("=" * 55)

    hf_extra_dir = os.path.join(BASE_DIR, "dataset", "Japan", "hf_extra")
    os.makedirs(hf_extra_dir, exist_ok=True)

    found_repos = search_hf_datasets("ja", keywords=("emotion speech japanese",
                                                      "japanese emotion",
                                                      "speech emotion ja"))
    fallback = [
        "pstroe/JTES",
        "WillWatson/JVNV",
        "ncoop57/japanese-speech-emotion",
        "jonatasgrosman/jsut-speech-corpus",
        "Audiogram/jvnv",
    ]
    all_candidates = list(dict.fromkeys(found_repos + fallback))
    hf_count = try_hf_repos(all_candidates, hf_extra_dir, "ja_hf", "Japan")

    # Augmentation
    ja_src_dirs = [
        os.path.join(BASE_DIR, "dataset", "Japan"),
        hf_extra_dir,
    ]
    aug_count = augment_language(ja_src_dirs, JA_AUG, "Japan", target_total=5000)

    all_ja = _count_wav(os.path.join(BASE_DIR, "dataset", "Japan"))
    log(f"Japanese รวม: original={all_ja} + hf_extra={hf_count}"
        f" + augmented={aug_count} ไฟล์", "OK")
    return aug_count + hf_count


def _count_wav(path):
    if not os.path.exists(path):
        return 0
    return sum(1 for _, _, fs in os.walk(path)
               for f in fs if f.lower().endswith(('.wav', '.flac', '.mp3')))


def print_final_summary():
    print("\n" + "=" * 55)
    print("  สรุป Dataset ทั้งหมด (รวม Augmented)")
    print("=" * 55)
    lang_dirs = {
        'Korean':   [
            os.path.join(BASE_DIR, "dataset", "Korean", "hi_kia", "labeled"),
            os.path.join(BASE_DIR, "dataset", "Korean", "hf_kgen"),
            os.path.join(BASE_DIR, "dataset", "Korean", "hf_extra"),
            os.path.join(BASE_DIR, "dataset", "Korean", "augmented"),
        ],
        'Japanese': [
            os.path.join(BASE_DIR, "dataset", "Japan"),
            os.path.join(BASE_DIR, "dataset", "Japan", "augmented"),
        ],
        'Chinese':  [os.path.join(BASE_DIR, "dataset", "Chinese")],
        'Thai':     [os.path.join(BASE_DIR, "dataset", "Thai")],
        'English':  [os.path.join(BASE_DIR, "dataset", "English")],
    }
    for lang, dirs in lang_dirs.items():
        total = sum(_count_wav(d) for d in dirs)
        status = ("✓ พอแล้ว" if total >= 2000
                  else ("△ ยังน้อย" if total >= 500 else "✗ น้อยมาก"))
        print(f"  {lang:<10}: {total:>6} ไฟล์  {status}")

    print("\n  หมายเหตุ:")
    print("  - Augmented ไฟล์อยู่ใน dataset/{lang}/augmented/")
    print("  - data_pipeline.py จะ scan ทุก subfolder อัตโนมัติ")
    print("  - Korean AIHub Dataset 263: aihub.or.kr (43,991 ไฟล์, ต้องสมัคร)")
    print("  - Korean KEMDy20: etri.re.kr (ต้องขอ)")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  EXTRA DATA: HuggingFace Search + Audio Augmentation")
    print("=" * 55)

    ko = download_korean()
    ja = download_japanese()
    print_final_summary()

    print(f"\n[OK] Korean เพิ่ม: {ko} | Japanese เพิ่ม: {ja} ไฟล์")
    if ko + ja > 0:
        print("\n  ขั้นตอนต่อไป — re-extract features:")
        print("  python data_pipeline.py --stage extract --lang Korean")
        print("  python data_pipeline.py --stage extract --lang Japan")
        print("  python data_pipeline.py --stage train   --lang Korean")
        print("  python data_pipeline.py --stage train   --lang Japan")
