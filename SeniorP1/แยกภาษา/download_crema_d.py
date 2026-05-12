"""
download_crema_d.py
===================
ดาวน์โหลด CREMA-D dataset สำหรับ English emotion recognition
Source: HuggingFace — sem-ens/CREMA-D (CC BY-NC-SA 3.0)
  - 7,442 ไฟล์ เสียงพูด (91 actors, 6 emotions)
  - Emotions: Angry, Disgust, Fear, Happy, Neutral, Sad

ผลลัพธ์: dataset/English/crema_d/{Emotion}_{filename}.wav
"""

import os
import sys
import shutil
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR    = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
CREMA_DIR   = os.path.join(BASE_DIR, "dataset", "English", "crema_d")
os.makedirs(CREMA_DIR, exist_ok=True)

EMOTION_MAP = {
    'ANG': 'Angry',
    'DIS': 'Disgust',
    'FEA': 'Fear',
    'HAP': 'Happy',
    'NEU': 'Neutral',
    'SAD': 'Sad',
}


def log(msg, level="INFO"):
    icons = {"INFO": "  ", "OK": "OK", "WARN": "!!", "ERROR": "XX", "RUN": ">>"}
    print(f"[{icons.get(level,'  ')}] {msg}")


def save_wav(audio_array, sr, out_path):
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=-1)
    peak = np.abs(audio_array).max()
    if peak > 0:
        audio_array = audio_array / peak * 0.9
    sf.write(out_path, audio_array, sr, subtype='PCM_16')


# ──────────────────────────────────────────────
# SOURCE 1: HuggingFace datasets API
# ──────────────────────────────────────────────
def download_hf():
    log("Source: HuggingFace sem-ens/CREMA-D", "RUN")
    try:
        from datasets import load_dataset
        # ลอง dataset ต่างๆ
        candidates = [
            ("sem-ens/CREMA-D",        {}),
            ("simonschoe/crema-d",     {}),
            ("Zahra99/CREMA-D",        {}),
            ("SamsonEu/CREMA-D",       {}),
        ]
        ds = None
        for name, kwargs in candidates:
            try:
                log(f"  ลอง {name} ...", "RUN")
                ds = load_dataset(name, split="train", **kwargs)
                log(f"  โหลดสำเร็จ: {len(ds)} samples | columns: {ds.column_names}", "OK")
                break
            except Exception as e:
                log(f"  {name}: {str(e)[:80]}", "WARN")

        if ds is None:
            return 0

        emotion_col = next((c for c in ['emotion','Emotion','label','Label'] if c in ds.column_names), None)
        audio_col   = next((c for c in ['audio','Audio'] if c in ds.column_names), None)
        if not emotion_col or not audio_col:
            log(f"  ไม่พบ column emotion/audio ใน {ds.column_names}", "ERROR")
            return 0

        count = 0
        for i, row in enumerate(ds):
            emo_raw = str(row.get(emotion_col, '')).upper()[:3]
            emotion = EMOTION_MAP.get(emo_raw)
            if emotion is None:
                # ลองแบบ full name
                full = str(row.get(emotion_col, '')).capitalize()
                if full in EMOTION_MAP.values():
                    emotion = full
                else:
                    continue

            audio = row.get(audio_col)
            if audio is None:
                continue

            arr = np.array(audio['array'])
            sr  = audio['sampling_rate']
            fname = f"cremad_{i:05d}_{emotion}.wav"
            out_path = os.path.join(CREMA_DIR, fname)
            if not os.path.exists(out_path):
                save_wav(arr, sr, out_path)
            count += 1
            if count % 200 == 0:
                log(f"  {count} ไฟล์...", "INFO")

        log(f"  โหลดสำเร็จ: {count} ไฟล์", "OK")
        return count

    except Exception as e:
        log(f"  HuggingFace failed: {e}", "WARN")
        return 0


# ──────────────────────────────────────────────
# SOURCE 2: Direct URL download (GitHub raw files list)
# ──────────────────────────────────────────────
def download_direct():
    """
    CREMA-D ไฟล์ audio อยู่บน GitHub:
    https://github.com/CheyneyComputerScience/CREMA-D/tree/master/AudioWAV
    ใช้ git clone เนื่องจากมีมากกว่า 7,000 ไฟล์
    """
    log("Source 2: Git clone CREMA-D AudioWAV folder", "RUN")
    repo_dir = os.path.join(BASE_DIR, "dataset", "English", "CREMA-D-repo")

    if not os.path.exists(repo_dir):
        log("  กำลัง sparse clone (AudioWAV only) ...", "RUN")
        import subprocess
        cmds = [
            ["git", "clone", "--filter=blob:none", "--sparse",
             "https://github.com/CheyneyComputerScience/CREMA-D.git", repo_dir],
        ]
        for cmd in cmds:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
            if r.returncode != 0:
                log(f"  clone failed: {r.stderr[:200]}", "WARN")
                return 0

        # sparse checkout AudioWAV only
        r = subprocess.run(["git", "sparse-checkout", "set", "AudioWAV"],
                           capture_output=True, text=True, cwd=repo_dir)
        if r.returncode != 0:
            log(f"  sparse-checkout failed: {r.stderr[:200]}", "WARN")

    audio_dir = os.path.join(repo_dir, "AudioWAV")
    if not os.path.exists(audio_dir):
        log(f"  ไม่พบ AudioWAV ใน {repo_dir}", "WARN")
        return 0

    # Copy + rename ไฟล์
    # CREMA-D filename: 1001_DFA_ANG_XX.wav
    # format: {actorID}_{sentence}_{emotion}_{intensity}.wav
    count = 0
    for fname in os.listdir(audio_dir):
        if not fname.lower().endswith('.wav'):
            continue
        parts = fname.replace('.wav', '').split('_')
        if len(parts) < 3:
            continue
        emo_code = parts[2].upper()
        emotion  = EMOTION_MAP.get(emo_code)
        if emotion is None:
            continue
        src = os.path.join(audio_dir, fname)
        dst = os.path.join(CREMA_DIR, f"cremad_{emotion}_{fname}")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        count += 1

    log(f"  Source 2: {count} ไฟล์", "OK")
    return count


def print_summary():
    from collections import Counter
    wavs = [f for f in os.listdir(CREMA_DIR) if f.endswith('.wav')]
    counts = Counter()
    for f in wavs:
        for emo in EMOTION_MAP.values():
            if emo in f:
                counts[emo] += 1
                break

    print(f"\n  CREMA-D Summary: {len(wavs)} ไฟล์รวม")
    for emo, n in sorted(counts.items()):
        print(f"    {emo:<10}: {n}")

    existing_eng = sum(
        1 for d in ['ravdess', 'tess', 'SAVEE']
        for root, _, files in os.walk(os.path.join(BASE_DIR, 'dataset', 'English', d))
        for f in files if f.endswith('.wav')
    )
    print(f"\n  English dataset รวมทั้งหมด: {existing_eng + len(wavs)} ไฟล์")
    if len(wavs) > 0:
        print("\n  ขั้นตอนต่อไป:")
        print("  1. python data_pipeline.py --stage extract --lang English")
        print("  2. python data_pipeline.py --stage train   --lang English")
        print("  3. python evaluate.py --lang English")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  CREMA-D DOWNLOADER (English Emotion Dataset)")
    print("=" * 55)

    # ตรวจว่าโหลดแล้วหรือยัง
    existing = [f for f in os.listdir(CREMA_DIR) if f.endswith('.wav')]
    if len(existing) >= 7000:
        log(f"มีอยู่แล้ว {len(existing)} ไฟล์ — ข้าม", "OK")
        print_summary()
    else:
        t1 = download_hf()
        if t1 == 0:
            t2 = download_direct()
        else:
            t2 = 0

        total = t1 + t2
        print_summary()
        print(f"\n[OK] ดาวน์โหลดใหม่: {total} ไฟล์")
