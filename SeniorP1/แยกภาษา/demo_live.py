"""
demo_live.py
============
Live Demo — อัดเสียงจากไมโครโฟน แล้ว predict อารมณ์ด้วย PyTorch ResNet
พร้อม confidence bar แสดงผล

Usage:
  python demo_live.py                     # ถามให้เลือกภาษา
  python demo_live.py --lang Thai         # ระบุภาษาตรงๆ
  python demo_live.py --lang auto         # ตรวจจับภาษาอัตโนมัติ (ต้องมี lang_detector.pkl)
  python demo_live.py --file audio.wav    # ทดสอบจากไฟล์ (ไม่ต้องใช้ไมโครโฟน)
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR   = r"C:\Users\Administrator\Desktop\66070131_SeniorP1\SeniorP1"
MODEL_DIR  = os.path.join(BASE_DIR, "models")

LANGUAGES  = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']
LANG_LABEL = {'Chinese': 'Chinese (จีน)', 'Japan': 'Japanese (ญี่ปุ่น)',
               'Korean': 'Korean (เกาหลี)', 'Thai': 'Thai (ไทย)', 'English': 'English (อังกฤษ)'}

# Emotion → color (ANSI)
EMO_COLOR = {
    'Angry':    '\033[91m',   # red
    'Happy':    '\033[93m',   # yellow
    'Sad':      '\033[94m',   # blue
    'Neutral':  '\033[37m',   # white
    'Fear':     '\033[95m',   # magenta
    'Surprise': '\033[96m',   # cyan
    'Disgust':  '\033[92m',   # green
}
RESET = '\033[0m'
BOLD  = '\033[1m'

N_MELS         = 128
MAX_TIME_STEPS = 130
SR             = 22050
RECORD_SECONDS = 3      # อัดเสียง 3 วินาทีต่อครั้ง


# ──────────────────────────────────────────────
# MODEL ARCHITECTURE  (ต้องตรงกับ data_pipeline.py)
# ──────────────────────────────────────────────
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class EmotionResNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.fc(x)


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
def load_model(lang):
    import pickle
    model_path   = os.path.join(MODEL_DIR, f"{lang}_model.pt")
    encoder_path = os.path.join(MODEL_DIR, f"{lang}_label_encoder.pkl")

    if not os.path.exists(model_path):
        print(f"[!!] ไม่พบ model: {model_path}")
        return None, None, None

    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(model_path, map_location=device, weights_only=False)

    n_classes = ckpt['n_classes']
    model = EmotionResNet(n_classes).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, le, device


# ──────────────────────────────────────────────
# AUDIO → MEL SPECTROGRAM
# ──────────────────────────────────────────────
def audio_to_mel(y, sr):
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          hop_length=512, fmax=sr // 2)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Pad / crop
    if mel_db.shape[1] >= MAX_TIME_STEPS:
        mel_db = mel_db[:, :MAX_TIME_STEPS]
    else:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_TIME_STEPS - mel_db.shape[1])),
                        mode='constant', constant_values=mel_db.min())
    return mel_db.astype(np.float32)


# ──────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────
def predict(model, le, device, mel):
    x = torch.tensor(mel[np.newaxis, np.newaxis, :, :]).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx       = int(np.argmax(probs))
    emotion   = le.classes_[idx]
    confidence = probs[idx] * 100
    all_probs  = {le.classes_[i]: probs[i] * 100 for i in range(len(probs))}
    return emotion, confidence, all_probs


# ──────────────────────────────────────────────
# LANGUAGE AUTO-DETECT
# ──────────────────────────────────────────────
def detect_language(mel):
    import pickle
    det_path = os.path.join(MODEL_DIR, "lang_detector.pkl")
    if not os.path.exists(det_path):
        return None
    with open(det_path, 'rb') as f:
        det = pickle.load(f)
    feat = mel.flatten().reshape(1, -1)
    pred = det.predict(feat)[0]
    return pred


# ──────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────
def print_result(lang, emotion, confidence, all_probs):
    col = EMO_COLOR.get(emotion, '')
    print("\n" + "─" * 50)
    print(f"  ภาษา    : {BOLD}{LANG_LABEL.get(lang, lang)}{RESET}")
    print(f"  อารมณ์  : {col}{BOLD}{emotion}{RESET}  ({confidence:.1f}%)")
    print("─" * 50)
    # Confidence bar
    bar_width = 30
    for emo, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        filled = int(round(bar_width * prob / 100))
        bar = '█' * filled + '░' * (bar_width - filled)
        c   = EMO_COLOR.get(emo, '')
        marker = ' ◄' if emo == emotion else ''
        print(f"  {emo:<10} {c}{bar}{RESET} {prob:5.1f}%{marker}")
    print("─" * 50)


# ──────────────────────────────────────────────
# RECORD MICROPHONE
# ──────────────────────────────────────────────
def record_audio(seconds=RECORD_SECONDS, sr=SR):
    try:
        import sounddevice as sd
    except ImportError:
        print("[!!] ไม่พบ sounddevice — ติดตั้งด้วย: pip install sounddevice")
        return None

    print(f"\n  🎙  กำลังอัด {seconds} วินาที ... (พูดได้เลย)")
    for i in range(3, 0, -1):
        print(f"      {i}...", end='\r')
        time.sleep(1)
    print("      กำลังอัด...  ", end='\r')

    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                   dtype='float32', blocking=True)
    print("      อัดเสร็จแล้ว ✓")
    return audio.flatten()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition — Live Demo')
    parser.add_argument('--lang', default=None,
                        help='ภาษา: Chinese/Japan/Korean/Thai/English/auto')
    parser.add_argument('--file', default=None,
                        help='ทดสอบจากไฟล์เสียง (wav/flac/mp3)')
    parser.add_argument('--seconds', type=int, default=RECORD_SECONDS,
                        help='ความยาวที่อัด (วินาที, default=3)')
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print(f"  {BOLD}SPEECH EMOTION RECOGNITION — LIVE DEMO{RESET}")
    print("=" * 55)
    device_str = 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'
    print(f"  Device: {device_str}")

    # ── เลือกภาษา ──
    lang = args.lang
    if lang is None:
        print("\n  เลือกภาษา:")
        for i, l in enumerate(LANGUAGES, 1):
            print(f"    {i}. {LANG_LABEL[l]}")
        print("    6. Auto-detect (ตรวจจับอัตโนมัติ)")
        choice = input("\n  กรุณาเลือก [1-6]: ").strip()
        if choice == '6':
            lang = 'auto'
        elif choice.isdigit() and 1 <= int(choice) <= 5:
            lang = LANGUAGES[int(choice) - 1]
        else:
            print("  ตัวเลือกไม่ถูกต้อง")
            return
    elif lang.lower() == 'auto':
        lang = 'auto'

    # ── โหลด model ──
    if lang != 'auto':
        model, le, device = load_model(lang)
        if model is None:
            return
        print(f"\n  โหลด model {lang} สำเร็จ (best_val_acc in checkpoint)")

    # ── Loop การทำนาย ──
    print("\n  กด Ctrl+C เพื่อหยุด\n")
    try:
        while True:
            # อ่านเสียง
            if args.file:
                import librosa
                print(f"\n  โหลดไฟล์: {args.file}")
                y, sr = librosa.load(args.file, sr=SR)
                args.file = None   # ครั้งต่อไปใช้ไมโครโฟน
            else:
                y = record_audio(seconds=args.seconds)
                if y is None:
                    break
                sr = SR

            # เสียงเงียบเกินไป?
            if np.abs(y).max() < 0.01:
                print("  [!!] เสียงเบาเกินไป — ลองพูดดังขึ้น")
                continue

            mel = audio_to_mel(y, sr)

            # ตรวจภาษาอัตโนมัติ
            actual_lang = lang
            if lang == 'auto':
                detected = detect_language(mel)
                if detected is None:
                    print("  [!!] ไม่พบ lang_detector.pkl — รัน language_detector.py ก่อน")
                    break
                actual_lang = detected
                print(f"  🌐 ตรวจพบภาษา: {LANG_LABEL.get(actual_lang, actual_lang)}")
                model, le, device = load_model(actual_lang)
                if model is None:
                    continue

            # Predict
            emotion, confidence, all_probs = predict(model, le, device, mel)
            print_result(actual_lang, emotion, confidence, all_probs)

            if args.file is None:
                inp = input("\n  กด Enter อัดต่อ, 'q' เพื่อออก: ").strip().lower()
                if inp == 'q':
                    break

    except KeyboardInterrupt:
        print("\n\n  ออกจาก Demo")

    print("\n  ขอบคุณที่ใช้งาน Speech Emotion Recognition!\n")


if __name__ == '__main__':
    main()
