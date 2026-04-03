"""
Demo: Emotion Recognition from Speech (Single Language)
Dataset  : RAVDESS (English) หรือ dataset ที่มี label อยู่ในชื่อไฟล์/โฟลเดอร์
Model    : CNN + BiLSTM + Self-Attention
Features : MFCC 40 + ZCR + RMS + Spectral Centroid = 43 features
"""

import os, sys, pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, BatchNormalization,
                                     Dropout, Bidirectional, LSTM, Dense,
                                     MultiHeadAttention, LayerNormalization, Add)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 1. CONFIG
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(os.path.dirname(CURRENT_DIR), "dataset")

SAMPLE_RATE = 22050
DURATION    = 3
SAMPLES     = SAMPLE_RATE * DURATION
BATCH_SIZE  = 32
EPOCHS      = 100

# อารมณ์ที่รองรับ
EMOTIONS = {
    'angry':   ['angry', 'anger', 'ag'],
    'happy':   ['happy', 'joy', 'hap'],
    'sad':     ['sad', 'sadness'],
    'neutral': ['neutral', 'neu'],
}

# RAVDESS naming: ส่วนที่ 3 ของชื่อไฟล์ = emotion code
RAVDESS_MAP = {'03': 'happy', '04': 'sad', '05': 'angry', '01': 'neutral'}

# ============================================================
# 2. FEATURE EXTRACTION  (43 features)
# ============================================================
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)     # (40, T)
    zcr  = librosa.feature.zero_crossing_rate(y)             # (1,  T)
    rms  = librosa.feature.rms(y=y)                          # (1,  T)
    sc   = librosa.feature.spectral_centroid(y=y, sr=sr)     # (1,  T)
    feat = np.concatenate([mfcc, zcr, rms, sc], axis=0)     # (43, T)
    return feat.T                                             # (T,  43)

def load_file(path):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
        y = np.pad(y, (0, max(0, SAMPLES - len(y))), 'constant')[:SAMPLES]
        return extract_features(y, sr)
    except Exception as e:
        print(f"  [skip] {os.path.basename(path)}: {e}")
        return None

def get_label(file_path):
    name  = os.path.basename(file_path).lower()
    parts = name.split('-')
    # RAVDESS format
    if len(parts) >= 3 and parts[2] in RAVDESS_MAP:
        return RAVDESS_MAP[parts[2]]
    # Keyword match
    path_lower = file_path.lower().replace('\\', '/')
    for emotion, keywords in EMOTIONS.items():
        if any(k in path_lower for k in keywords):
            return emotion
    return None

# ============================================================
# 3. LOAD DATA
# ============================================================
print("=" * 50)
print("  Demo: Emotion Recognition — Single Language")
print("=" * 50)
print(f"\n📂 สแกน dataset จาก: {DATA_PATH}\n")

X, y_labels = [], []
all_files = [
    os.path.join(r, f)
    for r, _, fs in os.walk(DATA_PATH)
    for f in fs if f.lower().endswith(('.wav', '.mp3', '.flac'))
]
print(f"  พบไฟล์เสียง: {len(all_files)} ไฟล์")

for i, fp in enumerate(all_files):
    label = get_label(fp)
    if label is None:
        continue
    feat = load_file(fp)
    if feat is not None:
        X.append(feat)
        y_labels.append(label)
    if (i + 1) % 100 == 0:
        print(f"  โหลดแล้ว {i+1}/{len(all_files)}...", end='\r')

print(f"\n✅ ข้อมูลที่ใช้ได้: {len(X)} samples")

if len(X) == 0:
    print("❌ ไม่พบข้อมูล! ตรวจสอบโฟลเดอร์ dataset และชื่อไฟล์")
    sys.exit(1)

# ============================================================
# 4. ENCODE & SPLIT
# ============================================================
lb = LabelEncoder()
y_enc = to_categorical(lb.fit_transform(y_labels))
print(f"🏷️  Classes: {list(lb.classes_)}\n")

X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# Normalize
scaler = StandardScaler()
N, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, F)
X_val   = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape[0], T, F)
X_test  = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape[0], T, F)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# บันทึก Scaler & Label Encoder
with open(os.path.join(CURRENT_DIR, 'demo_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(CURRENT_DIR, 'demo_label_encoder.pkl'), 'wb') as f:
    pickle.dump(lb, f)

# ============================================================
# 5. MODEL  (CNN → BiLSTM → Attention → Dense)
# ============================================================
inp = Input(shape=(T, F))

# CNN — สกัด pattern ระยะสั้น
x = Conv1D(128, 5, padding='same', activation='relu')(inp)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)

x = Conv1D(64, 5, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)

# BiLSTM — จับบริบทเวลา
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.3)(x)

# Self-Attention — โฟกัสช่วงเวลาสำคัญ
attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
x    = LayerNormalization()(Add()([x, attn]))
x    = Dropout(0.2)(x)

# สรุปผล
x   = Bidirectional(LSTM(32))(x)
x   = Dropout(0.3)(x)
x   = Dense(64, activation='relu')(x)
x   = Dropout(0.3)(x)
out = Dense(len(lb.classes_), activation='softmax', dtype='float32')(x)

model = Model(inp, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ============================================================
# 6. TRAIN
# ============================================================
callbacks = [
    ModelCheckpoint(os.path.join(CURRENT_DIR, 'demo_model.keras'),
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
]

print("\n🚀 เริ่ม Training...\n")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

# ============================================================
# 7. EVALUATE
# ============================================================
model.load_weights(os.path.join(CURRENT_DIR, 'demo_model.keras'))

# Accuracy / Loss Graph
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
y_pred  = np.argmax(model.predict(X_test), axis=1)
y_true  = np.argmax(y_test, axis=1)
cm      = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=lb.classes_))