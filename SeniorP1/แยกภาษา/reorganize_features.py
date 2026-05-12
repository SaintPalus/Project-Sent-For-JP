"""
reorganize_features.py
======================
จัดระเบียบ .npy ที่รวมกันอยู่ใน extracted_features/ แยกเป็นโฟลเดอร์ต่อภาษา
พร้อมติด emotion label แต่ละไฟล์

โครงสร้าง output:
    extracted_features/
    ├── Thai/
    │   ├── Neutral_s001_clip_actor001_impro1_1.npy
    │   ├── Happy_...npy
    │   └── unlabeled/   ← ที่หา emotion ไม่เจอ
    ├── Japan/
    ├── Korean/
    ├── Chinese/
    └── English/   ← extract ใหม่จาก dataset

การใช้งาน:
    python reorganize_features.py           # จัดระเบียบ + extract English
    python reorganize_features.py --no-extract   # จัดระเบียบอย่างเดียว ไม่ extract English
"""

import os
import sys
import shutil
import json
import argparse
import time
import numpy as np
import librosa

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# CONFIG
# ==========================================
BASE_DIR    = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
OLD_DIR     = os.path.join(BASE_DIR, "extracted_features")      # flat folder เดิม
FEATURE_DIR = os.path.join(BASE_DIR, "extracted_features")      # root เดิม (จะสร้าง subfolder)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
THAI_LABEL_JSON = os.path.join(DATASET_DIR, "Thai", "emotion_label.json")

SAMPLE_RATE    = 16000
N_MELS         = 128
MAX_TIME_STEPS = 130

# Map ชื่อ emotion ให้เป็น canonical form
EMOTION_NORM = {
    'angry'   : 'Angry',   'anger'  : 'Angry',
    'happy'   : 'Happy',   'joy'    : 'Happy',   'pleasant_surprise': 'Surprise',
    'sad'     : 'Sad',     'sadness': 'Sad',
    'neutral' : 'Neutral',
    'surprise': 'Surprise','sur'    : 'Surprise','surprised': 'Surprise',
    'fear'    : 'Fear',    'fearful': 'Fear',
    'disgust' : 'Disgust',
}
VALID_EMOTIONS = set(EMOTION_NORM.values())

# RAVDESS emotion map (ตัวเลขในชื่อไฟล์ ตำแหน่งที่ 3)
RAVDESS_MAP = {
    '01': 'Neutral', '02': 'Neutral',
    '03': 'Happy',   '04': 'Sad',
    '05': 'Angry',   '06': 'Fear',
    '07': 'Disgust', '08': 'Surprise',
}

# SAVEE emotion code (ตัวอักษรหลัง _ ในชื่อไฟล์)
SAVEE_MAP = {
    'a': 'Angry', 'd': 'Disgust', 'f': 'Fear',
    'h': 'Happy', 'n': 'Neutral', 'sa': 'Sad', 'su': 'Surprise',
}

# ==========================================
# HELPERS
# ==========================================
def log(msg, level="INFO"):
    icons = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "ERROR": "❌", "RUN": "🚀", "LANG": "🌐"}
    print(f"{icons.get(level, '   ')} {msg}")

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def normalize_emotion(raw):
    """แปลง emotion keyword ให้เป็น canonical (Angry/Happy/Sad/...)"""
    return EMOTION_NORM.get(raw.lower().strip())


def extract_emotion_from_name(basename, lang):
    """
    พยายามหา emotion จากชื่อไฟล์ (basename = ไม่มี extension)
    คืน emotion string หรือ None ถ้าหาไม่เจอ
    """
    name_lower = basename.lower()

    # --- Korean: Korean_Old_Female1_Angry ---
    # --- Japan: Japan_fear_F1_free_238 ---
    # --- Chinese: Chinese_Mel_02.mel.npy_ref_Angry_VC ---
    for kw, em in EMOTION_NORM.items():
        # ตรวจ word boundary ด้วย _ หรือ - หรือ . เพื่อกัน false positive
        for sep in ('_', '-', '.', ' '):
            if f'{sep}{kw}{sep}' in f'{sep}{name_lower}{sep}':
                return em

    # --- TESS folder pattern: OAF_Fear_word.wav → folder ชื่อ OAF_Fear ---
    # (ใช้ในส่วน extract English เท่านั้น)

    return None

# ==========================================
# STEP 1: โหลด Thai emotion_label.json
# ==========================================
def load_thai_label_map():
    """
    คืน dict: normalized_key → emotion
    key = ชื่อไฟล์ (ไม่มี extension) แบบ lowercase
    """
    if not os.path.exists(THAI_LABEL_JSON):
        log("ไม่พบ emotion_label.json ของ Thai", "WARN")
        return {}

    with open(THAI_LABEL_JSON, encoding='utf-8') as f:
        raw = json.load(f)

    label_map = {}
    for filename, entries in raw.items():
        if not entries:
            continue
        emotion = entries[0].get('assigned_emo') or entries[0].get('majority_emo')
        if not emotion:
            continue
        emotion = normalize_emotion(emotion) or emotion

        # key = ชื่อไฟล์ไม่มีนามสกุล ตัวเล็กหมด
        key = os.path.splitext(filename)[0].lower()
        label_map[key] = emotion

    log(f"Thai label map: {len(label_map):,} รายการ", "OK")
    return label_map


def thai_lookup(basename, thai_map):
    """
    basename เช่น s001_clip_actor001_impro1_1
    JSON key เช่น  s001_con_actor001_impro1_1
    ลอง exact match ก่อน ถ้าไม่เจอให้แทน clip→con
    """
    key = basename.lower()
    if key in thai_map:
        return thai_map[key]

    # แทน _clip_ → _con_  (Thai dataset มีสองรูปแบบ)
    alt = key.replace('_clip_', '_con_')
    if alt in thai_map:
        return thai_map[alt]

    return None

# ==========================================
# STEP 2: จัดระเบียบไฟล์ที่มีอยู่แล้ว
# ==========================================
def reorganize():
    print("\n" + "="*60)
    print("  STEP 1: REORGANIZE EXISTING .npy FILES")
    print("="*60)

    thai_map = load_thai_label_map()

    stats = {lang: {'ok': 0, 'unlabeled': 0} for lang in ['Thai', 'Japan', 'Korean', 'Chinese']}
    t0 = time.time()

    # รายการ .npy ทั้งหมดใน flat folder (ไม่รวม subfolder)
    all_files = [
        f for f in os.listdir(OLD_DIR)
        if f.endswith('.npy') and os.path.isfile(os.path.join(OLD_DIR, f))
    ]
    log(f"พบ .npy ทั้งหมด: {len(all_files):,} ไฟล์", "INFO")

    for filename in all_files:
        # ดึง language prefix: Thai_xxx.npy → Thai
        parts = filename.split('_', 1)
        if len(parts) < 2:
            continue
        lang = parts[0]
        if lang not in stats:
            continue  # ข้ามภาษาที่ไม่รู้จัก

        basename = os.path.splitext(filename)[0]  # ไม่มี .npy
        # basename = "Thai_s001_clip_actor001_impro1_1"
        # ตัด lang prefix ออก: "s001_clip_actor001_impro1_1"
        name_no_lang = parts[1].replace('.npy', '')  # กันกรณี double extension

        # --- หา emotion ---
        emotion = None

        if lang == 'Thai':
            emotion = thai_lookup(name_no_lang, thai_map)

        if emotion is None:
            emotion = extract_emotion_from_name(name_no_lang, lang)

        # --- ย้ายไฟล์ ---
        src = os.path.join(OLD_DIR, filename)

        if emotion and emotion in VALID_EMOTIONS:
            dest_dir = os.path.join(FEATURE_DIR, lang)
            make_dirs(dest_dir)
            dest = os.path.join(dest_dir, f"{emotion}_{name_no_lang}.npy")
            stats[lang]['ok'] += 1
        else:
            dest_dir = os.path.join(FEATURE_DIR, lang, "unlabeled")
            make_dirs(dest_dir)
            dest = os.path.join(dest_dir, filename)
            stats[lang]['unlabeled'] += 1

        # ถ้าปลายทางมีอยู่แล้วให้ข้าม (idempotent)
        if not os.path.exists(dest):
            shutil.move(src, dest)

    # สรุป
    print()
    print(f"{'ภาษา':12s} {'มี label':>10s} {'ไม่มี label':>12s}")
    print("-"*36)
    for lang, s in stats.items():
        print(f"{lang:12s} {s['ok']:>10,} {s['unlabeled']:>12,}")
    log(f"เสร็จ ({time.time()-t0:.1f}s)", "OK")

# ==========================================
# STEP 3: Extract English (ยังไม่มีไฟล์ .npy)
# ==========================================
def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mel   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_TIME_STEPS:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_TIME_STEPS - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_TIME_STEPS]
    return mel_db


def get_english_emotion(file_path):
    """
    รองรับ 3 sub-datasets ใน English/:
      1. ravdess  - 03-01-XX-...wav  (position 2 = emotion code)
      2. tess     - OAF_Fear/word.wav  (emotion ในชื่อโฟลเดอร์)
      3. SAVEE    - DC_a01.wav  (ตัวอักษรหลัง _ เป็น emotion code)
    """
    path_lower = file_path.lower().replace('\\', '/')
    filename   = os.path.basename(file_path)
    name_no_ext = os.path.splitext(filename)[0]

    # RAVDESS
    if 'ravdess' in path_lower:
        parts = name_no_ext.split('-')
        if len(parts) >= 3:
            return RAVDESS_MAP.get(parts[2])

    # TESS: โฟลเดอร์ชื่อ OAF_Fear, OAF_angry, OAF_Pleasant_surprise, YAF_...
    if 'tess' in path_lower:
        folder = os.path.basename(os.path.dirname(file_path)).lower()
        for kw, em in EMOTION_NORM.items():
            if kw in folder:
                return em

    # SAVEE: DC_a01, DC_sa01, DC_su01
    if 'savee' in path_lower:
        parts_name = name_no_ext.split('_')
        if len(parts_name) >= 2:
            code = parts_name[1].rstrip('0123456789')  # ตัดตัวเลขออก → a, sa, su
            return SAVEE_MAP.get(code)

    # Fallback: keyword ใน path
    for kw, em in EMOTION_NORM.items():
        if f'/{kw}/' in path_lower or f'_{kw}_' in path_lower or f'_{kw}.' in path_lower:
            return em

    return None


def extract_english():
    print("\n" + "="*60)
    print("  STEP 2: EXTRACT ENGLISH FEATURES")
    print("="*60)

    eng_dataset = os.path.join(DATASET_DIR, "English")
    if not os.path.exists(eng_dataset):
        log("ไม่พบ dataset/English/", "WARN")
        return

    out_dir = os.path.join(FEATURE_DIR, "English")
    make_dirs(out_dir, os.path.join(out_dir, "unlabeled"))

    processed, skipped, no_label = 0, 0, 0
    t0 = time.time()

    for root, _, files in os.walk(eng_dataset):
        for file in files:
            if not file.lower().endswith(('.wav', '.flac', '.mp3')):
                continue

            file_path = os.path.join(root, file)
            emotion   = get_english_emotion(file_path)
            base_name = os.path.splitext(file)[0]

            if emotion and emotion in VALID_EMOTIONS:
                # ป้องกันชื่อซ้ำโดยแทรก parent folder
                parent = os.path.basename(root)
                out_name = f"{emotion}_{parent}_{base_name}.npy"
                out_path = os.path.join(out_dir, out_name)
            else:
                out_path = os.path.join(out_dir, "unlabeled", f"{base_name}.npy")
                no_label += 1

            if os.path.exists(out_path):
                skipped += 1
                continue

            try:
                mel = extract_mel(file_path)
                np.save(out_path, mel)
                processed += 1
                if processed % 200 == 0:
                    log(f"  English: {processed} ไฟล์...", "INFO")
            except Exception as e:
                log(f"  ข้าม {file}: {e}", "WARN")

    log(f"English: สกัดใหม่ {processed} | ข้าม {skipped} | ไม่มี label {no_label} ({time.time()-t0:.1f}s)", "OK")

# ==========================================
# SUMMARY
# ==========================================
def print_summary():
    print("\n" + "="*60)
    print("  SUMMARY — extracted_features/")
    print("="*60)

    langs = ['Thai', 'Japan', 'Korean', 'Chinese', 'English']
    for lang in langs:
        lang_dir = os.path.join(FEATURE_DIR, lang)
        if not os.path.exists(lang_dir):
            continue

        labeled   = len([f for f in os.listdir(lang_dir) if f.endswith('.npy')])
        unlabeled_dir = os.path.join(lang_dir, 'unlabeled')
        unlabeled = len(os.listdir(unlabeled_dir)) if os.path.exists(unlabeled_dir) else 0

        # นับ emotion breakdown
        emotions = {}
        for f in os.listdir(lang_dir):
            if f.endswith('.npy'):
                emo = f.split('_')[0]
                emotions[emo] = emotions.get(emo, 0) + 1

        log(f"{lang:10s}: labeled={labeled:,} | unlabeled={unlabeled:,}", "OK")
        for emo, cnt in sorted(emotions.items()):
            print(f"             {emo:12s}: {cnt:,}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize extracted features by language")
    parser.add_argument('--no-extract', action='store_true',
                        help='ข้ามการ extract English ใหม่')
    args = parser.parse_args()

    start = time.time()

    reorganize()

    if not args.no_extract:
        extract_english()

    print_summary()

    print("\n" + "="*60)
    log(f"เสร็จสิ้น — {time.time()-start:.1f} วินาที", "OK")
    print("="*60)
    print()
    print("  ขั้นตอนต่อไป: รัน data_pipeline.py --stage train")
    print("="*60)
