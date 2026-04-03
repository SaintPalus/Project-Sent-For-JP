"""
Demo: Live Emotion Detection (Single Language)
กด Enter เพื่ออัดเสียง 3 วินาที → แสดงผลอารมณ์ทันที
"""

import os, sys, pickle
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONFIG
# ============================================================
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(CURRENT_DIR, 'demo_model.keras')
SCALER_PATH  = os.path.join(CURRENT_DIR, 'demo_scaler.pkl')
LABEL_PATH   = os.path.join(CURRENT_DIR, 'demo_label_encoder.pkl')

SAMPLE_RATE  = 22050
DURATION     = 3
SAMPLES      = SAMPLE_RATE * DURATION

EMOTION_ICON = {
    'angry':   '😡 Angry   (โกรธ)',
    'happy':   '😊 Happy   (มีความสุข)',
    'sad':     '😢 Sad     (เศร้า)',
    'neutral': '😐 Neutral (เฉยๆ)',
    'surprise':'😲 Surprise(ประหลาดใจ)',
}

# ============================================================
# FEATURE EXTRACTION  (ต้องตรงกับ Demo_Train.py)
# ============================================================
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    zcr  = librosa.feature.zero_crossing_rate(y)
    rms  = librosa.feature.rms(y=y)
    sc   = librosa.feature.spectral_centroid(y=y, sr=sr)
    feat = np.concatenate([mfcc, zcr, rms, sc], axis=0)
    return feat.T   # (T, 43)

# ============================================================
# RECORD
# ============================================================
def record():
    print(f"\n🔴 กำลังอัดเสียง {DURATION} วินาที  — พูดได้เลย!")
    audio = sd.rec(SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("✅ อัดเสียงเสร็จแล้ว กำลังวิเคราะห์...")
    return audio.flatten()

# ============================================================
# PREDICT
# ============================================================
def predict(audio, model, scaler, lb):
    audio, _ = librosa.effects.trim(audio, top_db=25)
    audio = np.pad(audio, (0, max(0, SAMPLES - len(audio))), 'constant')[:SAMPLES]

    feat     = extract_features(audio, SAMPLE_RATE)
    T, F     = feat.shape
    feat     = scaler.transform(feat.reshape(1, -1)).reshape(1, T, F)

    probs    = model.predict(feat, verbose=0)[0]
    idx      = np.argmax(probs)
    return lb.classes_[idx], probs[idx] * 100, probs

def show_result(label, conf, probs, lb):
    icon = EMOTION_ICON.get(label, label)
    print("\n" + "━" * 45)
    print(f"  ผลลัพธ์  :  {icon}")
    print(f"  ความมั่นใจ: {conf:.1f}%")
    print("━" * 45)
    for i, cls in enumerate(lb.classes_):
        bar  = "█" * int(probs[i] * 100 / 5)
        tag  = EMOTION_ICON.get(cls, cls)
        print(f"  {tag:<28} {probs[i]*100:5.1f}%  {bar}")
    print()

# ============================================================
# MAIN
# ============================================================
print("\n" + "=" * 45)
print("     🎙️  Live Emotion Detector  (Demo)")
print("=" * 45)

# โหลด model
for path, name in [(MODEL_PATH, 'Model'), (SCALER_PATH, 'Scaler'), (LABEL_PATH, 'Label Encoder')]:
    if not os.path.exists(path):
        print(f"❌ ไม่พบไฟล์ {name}  →  รัน Demo_Train.py ก่อนครับ")
        sys.exit(1)

print("⏳ กำลังโหลดโมเดล...")
model  = tf.keras.models.load_model(MODEL_PATH)
scaler = pickle.load(open(SCALER_PATH, 'rb'))
lb     = pickle.load(open(LABEL_PATH, 'rb'))
print(f"✅ พร้อมใช้งาน  |  อารมณ์ที่รู้จัก: {list(lb.classes_)}\n")

while True:
    try:
        cmd = input("กด Enter เพื่ออัดเสียง   หรือพิมพ์ 'q' เพื่อออก: ").strip().lower()
        if cmd == 'q':
            print("👋 จบการสาธิต")
            break
        audio = record()
        label, conf, probs = predict(audio, model, scaler, lb)
        show_result(label, conf, probs, lb)
    except KeyboardInterrupt:
        print("\n👋 จบการสาธิต")
        break
    except Exception as e:
        print(f"❌ Error: {e}")