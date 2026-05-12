"""
download_korean_data.py
=======================
ดาวน์โหลด Korean Emotion Speech datasets จากหลายแหล่ง:
  1. kgen-team/korean-voice-emotion-dataset  (HuggingFace, 67 files, CC BY 4.0)
  2. KESD — Korean Emotional Speech DB via HuggingFace
  3. Git LFS pull จาก repo ที่โหลดค้างอยู่

ผลลัพธ์จะเก็บใน:
  dataset/Korean/hf_kgen/        ← source 1
  dataset/Korean/hf_kesd/        ← source 2
"""

import os
import sys
import shutil
import subprocess
import soundfile as sf
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR    = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
KOREAN_DIR  = os.path.join(BASE_DIR, "dataset", "Korean")

EMOTION_NORM = {
    'angry'    : 'Angry',
    'anger'    : 'Angry',
    'happy'    : 'Happy',
    'happiness': 'Happy',
    'joy'      : 'Happy',
    'sad'      : 'Sad',
    'sadness'  : 'Sad',
    'surprised': 'Surprise',
    'surprise' : 'Surprise',
    'neutral'  : 'Neutral',
    'fear'     : 'Fear',
    'fearful'  : 'Fear',
    'disgust'  : 'Disgust',
}


def log(msg, level="INFO"):
    icons = {"INFO": "  ", "OK": "OK", "WARN": "!!", "ERROR": "XX", "RUN": ">>"}
    print(f"[{icons.get(level,'  ')}] {msg}")


def save_wav(audio_array, sr, out_path):
    """บันทึก audio array เป็น .wav"""
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=-1)
    # Normalize
    peak = np.abs(audio_array).max()
    if peak > 0:
        audio_array = audio_array / peak * 0.9
    sf.write(out_path, audio_array, sr, subtype='PCM_16')


# ==========================================
# SOURCE 1: Git LFS pull จาก repo ที่มีอยู่แล้ว
# ==========================================
def source1_git_lfs():
    log("Source 1: Git LFS pull (korean-voice-emotion-dataset)", "RUN")
    repo_dir = os.path.join(KOREAN_DIR, "korean-voice-emotion-dataset")
    out_dir  = os.path.join(KOREAN_DIR, "hf_kgen")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(repo_dir):
        log(f"  ไม่พบ repo: {repo_dir}", "WARN")
        return 0

    # Pull LFS files
    log("  กำลัง git lfs pull ...", "RUN")
    result = subprocess.run(
        ["git", "lfs", "pull"],
        cwd=repo_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"  git lfs pull failed: {result.stderr[:200]}", "WARN")

    # นับและ copy ไฟล์เสียง
    count = 0
    for root, _, files in os.walk(repo_dir):
        if '.git' in root:
            continue
        for f in files:
            if not f.lower().endswith('.wav'):
                continue
            src  = os.path.join(root, f)
            # ตรวจว่าเป็นไฟล์จริงหรือ LFS pointer (ขนาด < 200 bytes = pointer)
            if os.path.getsize(src) < 200:
                continue
            dst = os.path.join(out_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            count += 1

    log(f"  Source 1: {count} ไฟล์", "OK")
    return count


# ==========================================
# SOURCE 2: HuggingFace — kgen-team/korean-voice-emotion-dataset
# ==========================================
def source2_hf_kgen():
    log("Source 2: HuggingFace kgen-team/korean-voice-emotion-dataset", "RUN")
    out_dir = os.path.join(KOREAN_DIR, "hf_kgen")
    os.makedirs(out_dir, exist_ok=True)

    try:
        from datasets import load_dataset
        ds = load_dataset("kgen-team/korean-voice-emotion-dataset",
                          split="train", trust_remote_code=True)
        log(f"  โหลดสำเร็จ: {len(ds)} samples", "OK")
        log(f"  Columns: {ds.column_names}", "INFO")

        count = 0
        emotion_col = None
        for col in ['emotion', 'Emotion', 'label', 'Label']:
            if col in ds.column_names:
                emotion_col = col
                break

        for i, row in enumerate(ds):
            emotion_raw = str(row.get(emotion_col, 'unknown')).lower()
            emotion     = EMOTION_NORM.get(emotion_raw, emotion_raw.capitalize())

            audio = row.get('audio', None)
            if audio is None:
                continue

            arr = np.array(audio['array'])
            sr  = audio['sampling_rate']
            fname = f"kgen_{i:04d}_{emotion}.wav"
            out_path = os.path.join(out_dir, fname)

            if not os.path.exists(out_path):
                save_wav(arr, sr, out_path)
            count += 1

            if count % 20 == 0:
                log(f"  {count} ไฟล์...", "INFO")

        log(f"  Source 2: {count} ไฟล์ใน {out_dir}", "OK")
        return count

    except Exception as e:
        log(f"  Source 2 failed: {e}", "WARN")
        return 0


# ==========================================
# SOURCE 3: HuggingFace — Kratos-AI/korean-voice-emotion-dataset
# ==========================================
def source3_hf_kratos():
    log("Source 3: HuggingFace Kratos-AI/korean-voice-emotion-dataset", "RUN")
    out_dir = os.path.join(KOREAN_DIR, "hf_kratos")
    os.makedirs(out_dir, exist_ok=True)

    try:
        from datasets import load_dataset
        ds = load_dataset("Kratos-AI/korean-voice-emotion-dataset", split="train")
        log(f"  โหลดสำเร็จ: {len(ds)} samples | columns: {ds.column_names}", "OK")

        emotion_col = next((c for c in ['emotion','Emotion','label','Label'] if c in ds.column_names), None)
        count = 0
        for i, row in enumerate(ds):
            emotion_raw = str(row.get(emotion_col, 'unknown')).lower()
            emotion     = EMOTION_NORM.get(emotion_raw, emotion_raw.capitalize())
            audio = row.get('audio')
            if audio is None:
                continue
            arr, sr = np.array(audio['array']), audio['sampling_rate']
            fname   = f"kratos_{i:04d}_{emotion}.wav"
            out_path = os.path.join(out_dir, fname)
            if not os.path.exists(out_path):
                save_wav(arr, sr, out_path)
            count += 1
        log(f"  Source 3: {count} ไฟล์ใน {out_dir}", "OK")
        return count
    except Exception as e:
        log(f"  Source 3 failed: {e}", "WARN")
        return 0


# ==========================================
# SOURCE 4: Zenodo — Hi, KIA Dataset (488 files, CC-BY 4.0)
#   https://zenodo.org/records/7091465
# ==========================================
def source4_hi_kia():
    log("Source 4: Hi,KIA Dataset from Zenodo (488 files, CC-BY 4.0)", "RUN")
    out_dir  = os.path.join(KOREAN_DIR, "hi_kia")
    tmp_tar  = os.path.join(KOREAN_DIR, "hi_kia.tar.gz")
    os.makedirs(out_dir, exist_ok=True)

    # ตรวจว่าโหลดแล้วหรือยัง
    existing = sum(1 for f in os.listdir(out_dir) if f.endswith('.wav'))
    if existing >= 400:
        log(f"  มีอยู่แล้ว {existing} ไฟล์ ข้าม", "OK")
        return existing

    url = "https://zenodo.org/records/7091465/files/hi_kia_1.0.tar.gz"
    log(f"  กำลังดาวน์โหลด {url} ...", "RUN")

    try:
        import urllib.request, tarfile
        urllib.request.urlretrieve(url, tmp_tar)
        log("  แตกไฟล์...", "RUN")

        with tarfile.open(tmp_tar, 'r:gz') as tar:
            tar.extractall(out_dir)
        os.remove(tmp_tar)

        # Hi,KIA โครงสร้าง: wav/{speaker}_{scene}_{trial}_{code}.wav
        # emotion code: a=angry, h=happy, s=sad, n=neutral
        HIKIA_CODE = {'a': 'Angry', 'h': 'Happy', 's': 'Sad', 'n': 'Neutral', 'f': 'Fear'}
        wav_dir = os.path.join(out_dir, 'hi_kia', 'wav')
        if not os.path.exists(wav_dir):
            # ลอง nested folder
            for root, dirs, _ in os.walk(out_dir):
                if 'wav' in dirs:
                    wav_dir = os.path.join(root, 'wav')
                    break

        flat_dir = os.path.join(out_dir, 'labeled')
        os.makedirs(flat_dir, exist_ok=True)

        count = 0
        for root, _, files in os.walk(out_dir):
            if 'labeled' in root:
                continue
            for f in files:
                if not f.lower().endswith('.wav'):
                    continue
                # ชื่อไฟล์: F5_S01_1_a.wav → code = 'a'
                name   = os.path.splitext(f)[0]
                parts  = name.split('_')
                code   = parts[-1].lower() if parts else ''
                emotion = HIKIA_CODE.get(code)
                if emotion is None:
                    continue
                src = os.path.join(root, f)
                dst = os.path.join(flat_dir, f"hikia_{emotion}_{f}")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                count += 1

        log(f"  Source 4 (Hi,KIA): {count} ไฟล์", "OK")
        return count
    except Exception as e:
        log(f"  Source 4 failed: {e}", "WARN")
        if os.path.exists(tmp_tar):
            os.remove(tmp_tar)
        return 0


# ==========================================
# SUMMARY
# ==========================================
def print_summary():
    print("\n" + "=" * 55)
    print("  KOREAN DATASET SUMMARY")
    print("=" * 55)

    from collections import Counter
    total = 0
    for sub in ['hf_kgen', 'hf_kratos', os.path.join('hi_kia','labeled'),
                'korean-voice-emotion-dataset',
                'Korean Base Voice', 'korean_drama']:
        d = os.path.join(KOREAN_DIR, sub)
        if not os.path.exists(d):
            continue
        wavs = [f for f in os.walk(d) for f in f[2] if f.lower().endswith('.wav')]
        # flatten properly
        wavs = []
        for root, _, files in os.walk(d):
            if '.git' in root:
                continue
            for f in files:
                if f.lower().endswith('.wav') and os.path.getsize(os.path.join(root, f)) > 200:
                    wavs.append(f)
        if wavs:
            labels = Counter()
            for f in wavs:
                parts = f.lower().replace('.wav','').split('_')
                for p in parts:
                    if p in EMOTION_NORM:
                        labels[EMOTION_NORM[p]] += 1
                        break
            log(f"{sub}: {len(wavs)} ไฟล์  {dict(labels)}", "OK")
            total += len(wavs)

    print(f"\n  รวมทั้งหมด: {total} ไฟล์")
    if total >= 500:
        log("พร้อม retrain!", "OK")
    else:
        log(f"ยังน้อยอยู่ ({total}) แนะนำให้ดาวน์โหลด AIHub dataset เพิ่ม (ดูด้านล่าง)", "WARN")

    print("""
  หาก dataset ยังไม่พอ — แหล่งข้อมูลเพิ่มเติม:
  ┌─────────────────────────────────────────────────────┐
  │ 1. AIHub (ไทย-free สำหรับการศึกษา)                  │
  │    https://www.aihub.or.kr/                         │
  │    ค้นหา "감정 음성" → ได้ ~47,000 files            │
  │                                                     │
  │ 2. Kaggle - Korean Speech Emotion                   │
  │    https://www.kaggle.com/datasets/                 │
  │    ค้นหา "korean speech emotion"                    │
  │                                                     │
  │ 3. GitHub - KESD                                    │
  │    https://github.com/emotiontts/                   │
  └─────────────────────────────────────────────────────┘
""")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  KOREAN DATASET DOWNLOADER")
    print("=" * 55)

    t1 = source1_git_lfs()
    t2 = source2_hf_kgen()
    t3 = source3_hf_kratos()
    t4 = source4_hi_kia()

    print_summary()

    grand = t1 + t2 + t3 + t4
    print(f"\n[OK] ดาวน์โหลดใหม่รวม: {grand} ไฟล์")
    if grand > 0:
        print("\n  ขั้นตอนต่อไป:")
        print("  1. python reorganize_features.py --no-extract")
        print("     (pipeline จะ pick up ไฟล์ใน hf_kgen/hf_soulness ด้วย)")
        print("  2. python data_pipeline.py --stage extract --lang Korean")
        print("  3. python data_pipeline.py --stage train   --lang Korean")
        print("  4. python evaluate.py --lang Korean")
