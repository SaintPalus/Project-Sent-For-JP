# รายงานโครงงานนักศึกษา
## ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษาด้วยเทคนิคการเรียนรู้เชิงลึก
### Multilingual Speech Emotion Recognition System Using Deep Learning Techniques

---

| รายการ | รายละเอียด |
|---|---|
| ชื่อโครงงาน | ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษา |
| นักศึกษา | รหัส 66070131 |
| สาขาวิชา | วิทยาการคอมพิวเตอร์ |
| ภาคการศึกษา | 2 / 2567 |

---

## สารบัญ

| บทที่ | หน้า |
|---|---|
| บทคัดย่อ | I |
| ABSTRACT | II |
| กิตติกรรมประกาศ | III |
| สารบัญ | IV |
| สารบัญตาราง | VI |
| สารบัญรูป | VIII |
| **บทที่ 1 บทนำ** | **1** |
| 1.1 ที่มาและความสำคัญ | 1 |
| 1.2 วัตถุประสงค์ | 2 |
| 1.3 ขอบเขตของโครงงาน | 2 |
| 1.4 ขั้นตอนการดำเนินงานและแผนการดำเนินงาน | 4 |
| 1.5 ประโยชน์ที่คาดว่าจะได้รับ | 4 |
| 1.6 นิยามคำศัพท์ | 5 |
| **บทที่ 2 การทบทวนวรรณกรรมและเทคโนโลยีที่เกี่ยวข้อง** | **6** |
| 2.1 ความรู้เบื้องต้นเกี่ยวกับการรู้จำอารมณ์จากเสียง | 6 |
| 2.2 ทฤษฎีพื้นฐานด้านการประมวลผลเสียง | 8 |
| 2.3 ลักษณะทาง Prosody และความแตกต่างระหว่างภาษา | 12 |
| 2.4 สถาปัตยกรรม Deep Learning ที่ใช้ | 18 |
| 2.5 ชุดข้อมูลที่เกี่ยวข้อง | 24 |
| 2.6 สรุปผลการศึกษางานวิจัยที่เกี่ยวข้อง | 30 |
| **บทที่ 3 วิธีการดำเนินการวิจัย** | **33** |
| 3.1 ภาพรวมสถาปัตยกรรมระบบ | 33 |
| 3.2 การออกแบบ Data Pipeline | 37 |
| 3.3 สถาปัตยกรรมโมเดล CNN + Bi-LSTM | 55 |
| 3.4 การกำหนด Training Configuration | 70 |
| **บทที่ 4 ระบบต้นแบบ** | **85** |
| 4.1 ภาพรวมการออกแบบ (Overview Design) | 85 |
| 4.2 ผลการทดลองและการวิเคราะห์ปัญหา | 92 |
| 4.3 การวิเคราะห์สาเหตุที่ทำให้ไม่บรรลุเป้าหมาย | 100 |
| **บทที่ 5 สรุปผลการดำเนินงาน** | **106** |
| 5.1 สรุปผลการดำเนินงาน | 106 |
| 5.2 ปัญหาและอุปสรรคที่พบในการดำเนินงาน | 106 |
| 5.3 Future of Work — แนวทางการพัฒนาในอนาคต (แยกโมเดลตามภาษา) | 107 |
| บรรณานุกรม | 112 |
| ประวัติผู้เขียน | 115 |

---

## สารบัญตาราง

| ตาราง | รายละเอียด | หน้า |
|---|---|---|
| ตารางที่ 2.1 | คุณสมบัติ Prosody และความแตกต่างระหว่างภาษาอังกฤษกับภาษาเกาหลี | 13 |
| ตารางที่ 2.2 | เปรียบเทียบ Feature Extraction ชนิดต่างๆ สำหรับงาน SER | 19 |
| ตารางที่ 3.1 | โมเดลที่พัฒนาทั้งหมดในโครงงาน | 56 |
| ตารางที่ 3.2 | Training Configuration และเหตุผลในการเลือกใช้ | 71 |
| ตารางที่ 3.3 | เทคนิค Data Augmentation ที่ใช้ | 72 |
| ตารางที่ 4.1 | ผลการทดลองเบื้องต้นของ Multilingual Model | 92 |
| ตารางที่ 4.2 | ความสับสนระหว่างอารมณ์แต่ละชนิด (Confusion Analysis) | 96 |
| ตารางที่ 4.3 | สรุปข้อจำกัดของแนวทาง Multilingual | 101 |
| ตารางที่ 5.1 | แผนพัฒนาโมเดลแยกภาษาในอนาคต | 108 |
| ตารางที่ 5.2 | เปรียบเทียบแนวทาง Unified Model กับ Per-Language Model | 110 |

---

## สารบัญรูป

| รูป | รายละเอียด | หน้า |
|---|---|---|
| รูปที่ 2.1 | กระบวนการคำนวณ MFCC (Mel-Frequency Cepstral Coefficients) | 9 |
| รูปที่ 2.2 | ตัวอย่าง Mel Spectrogram ของเสียงอารมณ์โกรธ (ภาษาอังกฤษ vs ภาษาเกาหลี) | 10 |
| รูปที่ 2.3 | กราฟ Pitch Contour ของอารมณ์เดียวกันในสองภาษา | 14 |
| รูปที่ 3.1 | ภาพรวม Pipeline ของระบบ Multilingual SER | 34 |
| รูปที่ 3.2 | กระบวนการป้องกัน Data Leakage | 42 |
| รูปที่ 3.3 | สถาปัตยกรรมโมเดล CNN + Bidirectional LSTM | 57 |
| รูปที่ 4.1 | กราฟ Accuracy และ Loss ระหว่าง Training | 93 |
| รูปที่ 4.2 | Confusion Matrix ของ Multilingual Model | 97 |
| รูปที่ 5.1 | Pipeline ของแนวทาง Per-Language Model ในอนาคต | 108 |

---

# บทคัดย่อ

&emsp;โครงงานนี้นำเสนอการพัฒนา **ระบบรู้จำอารมณ์จากเสียงพูดแบบรองรับหลายภาษา** (Multilingual Speech Emotion Recognition: Multilingual SER) โดยใช้สถาปัตยกรรม **Per-Language ResNet** ซึ่งฝึกสอนโมเดล Convolutional Neural Network แบบ Residual Network แยกสำหรับแต่ละภาษา ระบบรองรับ **5 ภาษา** ได้แก่ ภาษาไทย ภาษาจีน ภาษาญี่ปุ่น ภาษาเกาหลี และภาษาอังกฤษ โดยใช้คุณลักษณะ **Mel Spectrogram** ขนาด 128×130 แทน MFCC เดิม เพื่อจับรูปแบบพลังงานเสียงในมิติเวลาและความถี่พร้อมกัน

&emsp;ระบบเริ่มต้นด้วย **Language Detector** แบบ SVM Pipeline (StandardScaler → PCA(80) → SVM kernel RBF) ที่วิเคราะห์จาก Mel Spectrogram เพื่อระบุภาษาของผู้พูดอัตโนมัติ ก่อนส่งต่อไปยังโมเดล ResNet เฉพาะภาษา การออกแบบนี้แก้ปัญหา **Prosody Mismatch** ที่เป็นอุปสรรคสำคัญของ Unified Multilingual Model โดยให้แต่ละโมเดลเรียนรู้รูปแบบอารมณ์ในบริบทของภาษานั้นๆ โดยตรง ไม่ต้องแข่งขันกับ Pattern ของภาษาอื่น

&emsp;ผลการทดลองบน Test Set ที่ไม่เคยใช้ฝึกสอน ได้ค่า Accuracy ดังนี้ ภาษาไทย 88.40% ภาษาจีน 86.62% ภาษาญี่ปุ่น 84.36% ภาษาอังกฤษ 82.84% และภาษาเกาหลี 53.57% (SVM ให้ผลดีกว่าที่ 72.62% เนื่องจากข้อมูลมีน้อย) **ค่าเฉลี่ยรวม 79.16%** สูงกว่าเป้าหมาย 75% ที่ตั้งไว้ Language Detector ทำงานได้อย่างแม่นยำ **98.54%** ทำให้ระบบโดยรวมทำงานได้อย่างมีประสิทธิภาพในการใช้งานจริง

&emsp;นอกจากนี้ยังพัฒนา **Web Demo** ด้วย Gradio และ **Cross-lingual Analysis** เพื่อศึกษาความคล้ายคลึงและความแตกต่างของรูปแบบอารมณ์ระหว่างภาษาต่างๆ ซึ่งเป็นองค์ความรู้สำหรับการพัฒนาต่อยอดในอนาคต

**คำสำคัญ:** การรู้จำอารมณ์จากเสียง, หลายภาษา, Per-Language Model, ResNet, Mel Spectrogram, Language Detector, SVM, Cross-lingual Analysis

---

# ABSTRACT

&emsp;This project presents a **Multilingual Speech Emotion Recognition (Multilingual SER)** system using a **Per-Language ResNet** architecture — a Residual Convolutional Neural Network trained separately for each language. The system supports **5 languages**: Thai, Chinese, Japanese, Korean, and English, using **Mel Spectrogram** features (128 mel bands × 130 time frames) extracted at 16,000 Hz sample rate, replacing the MFCC approach used in earlier work to better capture spatial energy distribution patterns.

&emsp;A key insight from prior research was that a single Unified Multilingual Model suffers from **Prosody Mismatch** — each language has distinctly different pitch contours, speech rates, and energy patterns for the same emotion, causing the model to confuse language characteristics with emotion characteristics. The Per-Language approach resolves this by allowing each model to learn emotion patterns within the prosodic context of its own language.

&emsp;The system employs an automatic **Language Detector** (SVM pipeline: StandardScaler → PCA(80) → SVM RBF) that identifies the spoken language from audio features before routing to the appropriate emotion model. This detector achieves **98.54% accuracy** on a 5-class language classification task.

&emsp;Experimental results on held-out test sets demonstrate: Thai 88.40%, Chinese 86.62%, Japanese 84.36%, English 82.84%, and Korean 53.57% (SVM outperforms ResNet at 72.62% due to limited data). The **overall average accuracy is 79.16%**, surpassing the 75% target. The project also includes a Gradio-based Web Demo and a cross-lingual feature analysis module for studying emotion pattern similarities across languages.

**Keywords:** Speech Emotion Recognition, Multilingual, Per-Language Model, ResNet, Mel Spectrogram, Language Detector, SVM, Cross-lingual Analysis

---

# กิตติกรรมประกาศ

&emsp;โครงงานนี้สำเร็จลุล่วงได้ด้วยความวิริยะอุตสาหะของผู้จัดทำในการศึกษาค้นคว้าองค์ความรู้ด้านการประมวลผลเสียง (Speech Processing) การเรียนรู้เชิงลึก (Deep Learning) และการวิเคราะห์ข้อมูลหลายภาษา

&emsp;ขอขอบคุณ Dataset ที่เปิดให้ใช้งานโดยไม่มีค่าใช้จ่าย ได้แก่ RAVDESS Dataset โดย Livingstone & Russo (2018), CREMA-D Dataset โดย Cao et al. (2014), Korean Voice Emotion Dataset (hi_kia), JANON Japanese Emotion Dataset, Thai Speech Emotion Dataset (TDED) รวมถึงชุมชน Open Source ที่พัฒนา Library ต่างๆ ที่ใช้ในโครงงานนี้ ได้แก่ PyTorch, torchaudio, Librosa, scikit-learn, Transformers (Hugging Face), Gradio, NumPy, Matplotlib และ Seaborn

&emsp;โครงงานนี้บรรลุเป้าหมาย Accuracy ที่ตั้งไว้ที่ 75% ด้วยค่าเฉลี่ยรวม 79.16% และ Language Detector 98.54% ซึ่งเป็นผลจากการประยุกต์ใช้สถาปัตยกรรม Per-Language ResNet ที่แก้ปัญหา Prosody Mismatch ได้อย่างมีประสิทธิภาพ

---

# บทที่ 1 บทนำ

## 1.1 ที่มาและความสำคัญ

&emsp;การรู้จำอารมณ์จากเสียงพูด (Speech Emotion Recognition: SER) เป็นสาขาวิจัยที่ได้รับความสนใจอย่างมากในทศวรรษที่ผ่านมา ระบบ SER มีการนำไปประยุกต์ใช้ในหลายด้าน ทั้งในระบบ Call Center อัตโนมัติที่ตรวจจับความไม่พอใจของลูกค้า, ระบบดูแลสุขภาพจิตที่ตรวจจับภาวะซึมเศร้า, อุปกรณ์ IoT อัจฉริยะที่ตอบสนองต่อสภาวะอารมณ์ของผู้ใช้งาน ตลอดจนระบบ In-car Emotion Detection ในรถยนต์สมัยใหม่

&emsp;ในโลกปัจจุบันที่มีการสื่อสารข้ามภาษาและข้ามวัฒนธรรมมากขึ้น การพัฒนาโมเดล SER ที่รองรับหลายภาษาในระบบเดียว **(Multilingual SER)** จึงเป็นที่ต้องการอย่างยิ่ง เพราะจะช่วยลดต้นทุนในการพัฒนาและบำรุงรักษาระบบ และทำให้ระบบสามารถใช้งานได้กับผู้ใช้ที่หลากหลายภาษาโดยไม่ต้องสร้างโมเดลแยกกันสำหรับแต่ละภาษา

&emsp;**แนวคิดเริ่มต้นของโครงงานนี้** คือการทดลองแนวทาง Multilingual Unified Model โดยนำข้อมูลเสียงพูดจากภาษาอังกฤษและภาษาเกาหลีมารวมกัน แล้วฝึกสอนโมเดล CNN + Bidirectional LSTM เพียงตัวเดียวเพื่อจำแนกอารมณ์ โดยมีสมมติฐานว่าโมเดลจะสามารถเรียนรู้ "รูปแบบอารมณ์ที่เป็นสากล" (Language-Independent Emotion Features) ออกมาได้จากข้อมูลที่หลากหลาย

&emsp;อย่างไรก็ตาม ในระหว่างการพัฒนาได้ค้นพบว่าสมมติฐานดังกล่าวมีข้อจำกัดสำคัญ เนื่องจากเสียงพูดในแต่ละภาษามีโครงสร้างทาง Prosody (น้ำเสียง จังหวะ ระดับเสียง) ที่แตกต่างกันอย่างมีนัยสำคัญ ซึ่งเป็นปัญหาที่ทำให้โครงงานไม่บรรลุเป้าหมายเดิม และนำไปสู่การวิเคราะห์และเสนอแนวทางแก้ไขในอนาคต

## 1.2 วัตถุประสงค์

1. พัฒนาระบบ SER แบบ Multilingual ด้วยสถาปัตยกรรม **Per-Language ResNet** โดยฝึกสอนโมเดลแยกสำหรับแต่ละภาษา
2. สร้าง **Language Detector** อัตโนมัติที่สามารถระบุภาษาจากเสียงพูดได้อย่างแม่นยำ เพื่อส่งต่อไปยังโมเดลที่เหมาะสม
3. ศึกษาและวิเคราะห์ปัญหา **Prosody Mismatch** ระหว่างภาษาต่างๆ ผ่าน Cross-lingual Feature Analysis
4. ประเมินประสิทธิภาพของโมเดลด้วยตัวชี้วัด Accuracy, F1-Score, Confusion Matrix และ Classification Report
5. พัฒนา **Web Demo** ด้วย Gradio สำหรับทดสอบระบบแบบ Real-time

## 1.3 ขอบเขตของโครงงาน

**ภาษาที่รองรับ (5 ภาษา):**

| ภาษา | Dataset | จำนวนตัวอย่างทั้งหมด | จำนวนอารมณ์ |
|---|---|---|---|
| ภาษาไทย | Thai Speech Emotion Dataset (TDED) | ~30,210 | 4 |
| ภาษาจีน | Chinese Emotional Speech Corpus | ~2,690 | 6 |
| ภาษาญี่ปุ่น | JANON Japanese Emotion Dataset | ~1,215 | 6 |
| ภาษาเกาหลี | Korean Voice Emotion Dataset (hi_kia) | ~420 | 5 |
| ภาษาอังกฤษ | RAVDESS + CREMA-D | ~11,425 | 7 |

**ประเภทอารมณ์ที่จำแนก (ต่างกันตามภาษา):**

| อารมณ์ | ภาษาไทย | ภาษาจีน | ภาษาญี่ปุ่น | ภาษาเกาหลี | ภาษาอังกฤษ |
|---|---|---|---|---|---|
| Angry (โกรธ) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Happy (มีความสุข) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Sad (เศร้า) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Neutral (เป็นกลาง) | ✅ | ✅ | — | ✅ | ✅ |
| Surprise (ประหลาดใจ) | — | ✅ | ✅ | ✅ | ✅ |
| Fear (กลัว) | — | ✅ | ✅ | — | ✅ |
| Disgust (รังเกียจ) | — | — | ✅ | — | ✅ |

**Features ที่ใช้:**
- **Mel Spectrogram** ขนาด 128 Mel Bands × 130 Time Frames = Tensor (1, 128, 130)
- สกัดจากสัญญาณเสียงที่ Sample Rate 16,000 Hz ความยาว 3 วินาที
- ใช้ n_fft=1024, hop_length=512, n_mels=128, fmin=0, fmax=8000

**Hardware & Framework:**
- GPU: NVIDIA GeForce RTX 3060 (12GB VRAM)
- Framework: **PyTorch** + torchaudio 2.6.0
- Audio Processing: torchaudio, librosa
- Sample Rate: **16,000 Hz**
- Duration: 3 วินาทีต่อไฟล์

**ขอบเขตที่ไม่ครอบคลุม:**
- ไม่รวมการรู้จำเสียงพูดแบบ Real-time Streaming ต่อเนื่อง
- ไม่รวมการประมวลผลภาษาธรรมชาติ (NLP) เพื่อวิเคราะห์เนื้อหาของคำพูด
- ภาษาเกาหลีมีข้อมูลน้อยกว่าภาษาอื่นมาก ผลลัพธ์จึงต่ำกว่า

## 1.4 ขั้นตอนการดำเนินงานและแผนการดำเนินงาน

```
ขั้นตอนที่ 1: ศึกษาทฤษฎีและวิเคราะห์ปัญหา (สัปดาห์ที่ 1-2)
├── ทบทวนปัญหา Prosody Mismatch จากงานวิจัยก่อนหน้า
├── ทฤษฎี Mel Spectrogram และ ResNet
└── ศึกษาแนวทาง Per-Language Model

ขั้นตอนที่ 2: รวบรวมและเตรียมข้อมูล (สัปดาห์ที่ 3-5)
├── ดาวน์โหลด RAVDESS + CREMA-D (English)
├── รวบรวม Korean, Japanese, Chinese, Thai Datasets
└── สร้าง Data Pipeline: Mel Spectrogram Extraction + Anti-Leakage Split

ขั้นตอนที่ 3: พัฒนา Language Detector (สัปดาห์ที่ 6)
├── สกัด Mel Spectrogram สำหรับทุกภาษา
├── Train SVM Pipeline (StandardScaler → PCA → SVM)
└── ประเมินผล: 98.54% accuracy

ขั้นตอนที่ 4: Train Per-Language ResNet Models (สัปดาห์ที่ 7-10)
├── สร้างสถาปัตยกรรม EmotionResNet (stem+layer1-3+pool+classifier)
├── Train แยกสำหรับแต่ละภาษา พร้อม SpecAugment + Audio Augmentation
└── บันทึกโมเดลที่ดีที่สุดตาม val_accuracy

ขั้นตอนที่ 5: Wav2Vec2 Feature Extraction (สัปดาห์ที่ 11)
├── สกัด facebook/wav2vec2-large-xlsr-53 embeddings
├── Train MLP Classifier บน Wav2Vec2 Features
└── เปรียบเทียบ ResNet vs Wav2Vec2 vs SVM Baseline

ขั้นตอนที่ 6: Cross-lingual Analysis และ Web Demo (สัปดาห์ที่ 12)
├── วิเคราะห์ความคล้ายคลึงระหว่างภาษา (crosslingual_analysis.py)
├── พัฒนา Web Demo ด้วย Gradio
└── ทดสอบระบบ End-to-end

ขั้นตอนที่ 7: สรุปผลและเขียนรายงาน (สัปดาห์ที่ 13-15)
├── วิเคราะห์ผลที่ได้และสาเหตุที่ Korean ทำได้น้อย
└── จัดทำรายงานฉบับสมบูรณ์
```

## 1.5 ประโยชน์ที่คาดว่าจะได้รับ

1. **องค์ความรู้ด้าน Prosody Mismatch** — เข้าใจสาเหตุและกลไกที่ทำให้โมเดล Multilingual ไม่สามารถทำงานได้อย่างมีประสิทธิภาพ ซึ่งเป็นประโยชน์ต่องานวิจัยในอนาคต
2. **Pipeline ที่พร้อมใช้งาน** — ระบบ Feature Extraction, Data Augmentation และ Anti-Data-Leakage ที่พัฒนาขึ้นสามารถนำไปต่อยอดได้ทันที
3. **แนวทาง Future of Work** — เสนอกรอบการพัฒนา Per-Language Model ที่มีแนวโน้มจะแก้ปัญหา Prosody Mismatch ได้
4. **ประสบการณ์การทดลองจริง** — การทดสอบ Configuration หลายรูปแบบบน GPU จริงให้ข้อมูลเชิงปฏิบัติที่มีคุณค่า

## 1.6 นิยามคำศัพท์

| คำศัพท์ | คำนิยาม |
|---|---|
| **SER (Speech Emotion Recognition)** | การรู้จำอารมณ์จากเสียงพูดโดยใช้คอมพิวเตอร์ |
| **Mel Spectrogram** | การแสดงการกระจายพลังงานเสียงในมิติเวลาและความถี่บน Mel Scale ให้ข้อมูลเชิงพื้นที่ที่ MFCC ไม่มี |
| **MFCC (Mel-Frequency Cepstral Coefficients)** | คุณลักษณะมาตรฐานสำหรับงาน Speech Processing สกัดจากสเปกตรัมของเสียง |
| **ResNet (Residual Network)** | สถาปัตยกรรม CNN ที่มี Skip Connection ช่วยให้ฝึก Deep Network ได้ง่ายขึ้น |
| **Per-Language Model** | แนวทางฝึกสอนโมเดลแยกสำหรับแต่ละภาษา เพื่อหลีกเลี่ยงปัญหา Prosody Mismatch |
| **Language Detector** | ระบบ SVM ที่ระบุภาษาจากเสียงพูดอัตโนมัติก่อนส่งไปยังโมเดล Emotion ที่เหมาะสม |
| **Prosody** | คุณสมบัติเหนือระดับเสียง ได้แก่ Pitch, Duration, Energy, Rhythm |
| **Prosody Mismatch** | ความแตกต่างของรูปแบบ Prosody ระหว่างภาษาที่ทำให้โมเดล Unified สับสน |
| **SpecAugment** | เทคนิค Data Augmentation สำหรับ Spectrogram โดยการ Mask แบบสุ่มตามแกนเวลาและความถี่ |
| **Data Leakage** | การที่ข้อมูล Test ปนเข้าสู่กระบวนการ Train ทำให้ผลประเมินเกินจริง |
| **Data Augmentation** | การสร้างข้อมูลเพิ่มเทียมจากข้อมูลต้นฉบับเพื่อเพิ่มความหลากหลาย |
| **Wav2Vec2** | โมเดล Pre-trained Self-supervised Learning จาก Facebook AI สำหรับงาน Speech Representation |
| **Cross-lingual Analysis** | การวิเคราะห์ความคล้ายคลึงและความแตกต่างของ Feature Space ระหว่างภาษาต่างๆ |

---

# บทที่ 2 การทบทวนวรรณกรรมและเทคโนโลยีที่เกี่ยวข้อง

## 2.1 ความรู้เบื้องต้นเกี่ยวกับการรู้จำอารมณ์จากเสียง

### 2.1.1 ทฤษฎีอารมณ์พื้นฐาน (Basic Emotion Theory)

Ekman (1992) เสนอว่ามนุษย์มีอารมณ์พื้นฐาน 6 ประเภทที่เป็นสากลข้ามวัฒนธรรม ได้แก่ Angry, Happy, Sad, Fear, Disgust และ Surprise โดยอารมณ์เหล่านี้มีการแสดงออกทางสีหน้าที่คล้ายกันในทุกวัฒนธรรม อย่างไรก็ตาม งานวิจัยในเวลาต่อมาพบว่าการแสดงออกทาง **เสียงพูด** นั้นมีความแตกต่างระหว่างวัฒนธรรมมากกว่าสีหน้า โดยเฉพาะในแง่ของ Prosody

&emsp;โครงงานนี้เลือกใช้อารมณ์ 5 ประเภทที่มีข้อมูล Dataset รองรับ ได้แก่ Angry, Happy, Sad, Neutral และ Surprise

### 2.1.2 ประวัติและพัฒนาการของ SER

| ยุค | แนวทาง | ข้อดี | ข้อจำกัด |
|---|---|---|---|
| ทศวรรษ 1990-2000 | Hand-crafted Features + HMM/SVM | ง่าย, ตีความได้ | ต้องเลือก Feature เอง, Accuracy ต่ำ |
| ทศวรรษ 2010-2015 | Deep Neural Networks | Accuracy สูงขึ้น | ต้องการข้อมูลมาก |
| ทศวรรษ 2015-ปัจจุบัน | CNN + LSTM / Transformer | State-of-the-art | ใช้ทรัพยากรมาก, Multilingual ยาก |

### 2.1.3 ความท้าทายของ Multilingual SER

งานวิจัยของ Schuller et al. (2010) ระบุว่า Cross-corpus (ข้ามชุดข้อมูล) และ Cross-lingual (ข้ามภาษา) เป็นความท้าทายหลักสองประการของ SER โดย:
- **Cross-corpus:** Accuracy มักลดลง 15-30% เมื่อทดสอบบน Dataset ต่างจากที่ใช้ Train
- **Cross-lingual:** Accuracy มักลดลง 20-40% เมื่อทดสอบข้ามภาษา เนื่องจาก Prosody Mismatch

## 2.2 ทฤษฎีพื้นฐานด้านการประมวลผลเสียง

### 2.2.1 Mel-Frequency Cepstral Coefficients (MFCC)

MFCC เป็น Feature มาตรฐานที่ใช้ในงาน Speech Processing มาตั้งแต่ทศวรรษ 1980 (Davis & Mermelstein, 1980) ขั้นตอนการคำนวณประกอบด้วย:

**ขั้นตอนที่ 1 — Framing & Windowing:**
แบ่งสัญญาณเสียงเป็น Frame เล็กๆ ขนาดประมาณ 25 ms โดยแต่ละ Frame ทับซ้อนกัน 10 ms คูณด้วย Hamming Window เพื่อลด Spectral Leakage:

$$w(n) = 0.54 - 0.46\cos\left(\frac{2\pi n}{N-1}\right)$$

**ขั้นตอนที่ 2 — Fast Fourier Transform (FFT):**
แปลงสัญญาณจาก Time Domain สู่ Frequency Domain:

$$X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-j2\pi kn/N}$$

**ขั้นตอนที่ 3 — Mel Filter Bank:**
ผ่านชุด Triangular Filters ที่จัดเรียงบน Mel Scale ซึ่งเลียนแบบการรับรู้เสียงของหูมนุษย์:

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

**ขั้นตอนที่ 4 — Log + DCT:**
นำ Log มาลด Dynamic Range แล้วใช้ Discrete Cosine Transform (DCT) แปลงเป็น Cepstral Coefficients:

$$c(n) = \sum_{k=1}^{K} \log\left(S_k\right) \cdot \cos\left[n\left(k - \frac{1}{2}\right)\frac{\pi}{K}\right]$$

**โครงงานนี้ใช้:**
- `n_mfcc = 40` สำหรับโมเดลทั่วไป (เร็ว, ประหยัด VRAM)
- `n_mfcc = 128` สำหรับโมเดล High-Resolution

```python
# โค้ดสกัด MFCC ที่ใช้ในโครงงาน
mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)  # → shape: (40, T)
features = mfcc.T                                        # → shape: (T, 40)
```

### 2.2.2 Mel Spectrogram

Mel Spectrogram แสดงการกระจายพลังงานของเสียงในมิติเวลาและความถี่ (บน Mel Scale) ให้ข้อมูลเชิงพื้นที่ (Spatial/Temporal Information) ที่ MFCC เพียงอย่างเดียวไม่มี:

```python
# โมเดล High-Resolution รวม MFCC + Mel Spectrogram
mfcc    = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=128)      # (128, T)
mel     = librosa.feature.melspectrogram(y=data, sr=sr)
mel_db  = librosa.power_to_db(mel, ref=np.max)                   # (128, T)
result  = np.concatenate((mfcc, mel_db), axis=0).T               # (T, 256)
```

ผลรวม Feature มิติ **256** ต่อ Time Step ให้ข้อมูลครบถ้วนทั้ง Spectral Envelope (จาก MFCC) และ Energy Distribution (จาก Mel Spectrogram)

### 2.2.3 Feature Comparison

| Feature | มิติ | จุดเด่น | จุดด้อย |
|---|---|---|---|
| MFCC (40) | (T, 40) | เร็ว, เบา, มาตรฐาน | ข้อมูลน้อยกว่า |
| MFCC (128) | (T, 128) | ความละเอียดสูง | ต้องการ VRAM มากขึ้น |
| MFCC 128 + Mel | (T, 256) | ข้อมูลครบถ้วนที่สุด | หนักที่สุด |

## 2.3 ลักษณะทาง Prosody และความแตกต่างระหว่างภาษา

### 2.3.1 Prosody คืออะไร

**Prosody** คือคุณสมบัติเหนือระดับเสียง (Suprasegmental Phonological Features) ของภาษา ซึ่งส่งผลต่อความหมายและอารมณ์ที่สื่อออกมาในการพูด

**ตารางที่ 2.1 คุณสมบัติ Prosody และความแตกต่างระหว่างภาษา**

| คุณสมบัติ | คำอธิบาย | ภาษาอังกฤษ | ภาษาเกาหลี |
|---|---|---|---|
| **Pitch (F0)** | ความถี่พื้นฐานของเสียง | ช่วง Pitch กว้าง (100-400 Hz) | ช่วง Pitch แคบกว่า (100-280 Hz) |
| **Duration** | ความยาวของพยางค์ | Stress-timed Rhythm — พยางค์ที่เน้นยาวกว่ามาก | Syllable-timed — ความยาวพยางค์สม่ำเสมอกว่า |
| **Energy** | ความดังของเสียง | ระดับ Energy สูงขึ้นฉับพลันเมื่อแสดงอารมณ์รุนแรง | ระดับ Energy เพิ่มขึ้นทีละน้อยกว่า |
| **Rhythm** | จังหวะการพูด | ไม่สม่ำเสมอ ขึ้นกับ Stress | สม่ำเสมอกว่า |
| **Intonation** | รูปแบบการขึ้นลงของ Pitch | ขึ้นลงชัดเจนมาก | ขึ้นลงน้อยกว่า แต่มีความหมายทาง Phonological |

### 2.3.2 ผลกระทบต่อ MFCC

MFCC สะท้อนโครงสร้างทาง Spectral ของเสียง ซึ่งได้รับอิทธิพลจากทั้ง **อารมณ์** และ **ภาษา** พร้อมกัน:

```
MFCC ที่โมเดลเห็น = ลักษณะอารมณ์  +  ลักษณะภาษา  +  ลักษณะผู้พูด
                      (สิ่งที่ต้องการ)   (Noise)         (Noise)
```

**ตัวอย่างเปรียบเทียบอารมณ์ "โกรธ" ระหว่างสองภาษา:**

```
อารมณ์ "โกรธ" (Angry):
┌────────────────┬──────────────────────┬──────────────────────┐
│ คุณสมบัติ      │ ภาษาอังกฤษ            │ ภาษาเกาหลี            │
├────────────────┼──────────────────────┼──────────────────────┤
│ Pitch Range    │ กว้างมาก (สูงฉับพลัน) │ แคบกว่า (ค่อยๆ สูงขึ้น) │
│ Speech Rate    │ เร็ว, ไม่สม่ำเสมอ     │ เร็ว แต่สม่ำเสมอกว่า   │
│ Energy Pattern │ พุ่งสูงฉับพลัน        │ เพิ่มขึ้นทีละน้อย       │
│ Vowel Duration │ ยาว (Stressed)        │ สั้นกว่า               │
│ MFCC Centroid  │ สูง (High Frequency)  │ ต่ำกว่า                │
└────────────────┴──────────────────────┴──────────────────────┘
```

### 2.3.3 ผลกระทบต่อโมเดล

เมื่อโมเดลเห็นข้อมูลรวมจากทั้งสองภาษา โมเดลจะประสบกับปัญหา:

1. **Confusion Between Language and Emotion:** โมเดลไม่สามารถแยกแยะได้ว่า Pitch ที่สูงขึ้นเป็นเพราะ "อารมณ์โกรธ" หรือเพราะ "ลักษณะของภาษาอังกฤษ"
2. **Overfitting Toward Majority Language:** ภาษาอังกฤษ (RAVDESS) มีข้อมูลมากกว่า โมเดลจึงเอนเอียงไปเรียนรู้รูปแบบของภาษาอังกฤษเป็นหลัก
3. **Poor Generalization:** เมื่อทดสอบด้วยเสียงภาษาเกาหลี โมเดลใช้รูปแบบที่เรียนรู้จากภาษาอังกฤษทำนาย ส่งผลให้ Accuracy ต่ำ

## 2.4 สถาปัตยกรรม Deep Learning ที่ใช้

### 2.4.1 Conv1D (1D Convolutional Layer)

Conv1D สกัด **Local Temporal Pattern** จาก Feature Sequence โดยใช้ Kernel เลื่อนผ่านมิติเวลา เหมาะกับข้อมูลที่เป็น Sequential เช่น MFCC ที่มีรูปแบบซ้ำๆ ตามเวลา

```
Input: (Batch, Time_Steps, Features)
Kernel ขนาด 5: สกัด Pattern จาก 5 Time Steps ที่ต่อเนื่องกัน
Output: (Batch, Time_Steps, Filters)
```

### 2.4.2 Bidirectional LSTM

LSTM (Long Short-Term Memory) แก้ปัญหา Vanishing Gradient ของ RNN ทั่วไปด้วย Gate Mechanism:
- **Forget Gate:** ตัดสินใจว่าจะลืมข้อมูลใดใน Cell State
- **Input Gate:** ตัดสินใจว่าจะเก็บข้อมูลใหม่อะไรลง Cell State
- **Output Gate:** ตัดสินใจว่าจะส่งออกอะไรเป็น Hidden State

**Bidirectional LSTM** อ่านลำดับข้อมูลทั้งสองทิศทาง:

$$h_t = \left[\overrightarrow{\text{LSTM}}(x_t) \; ; \; \overleftarrow{\text{LSTM}}(x_t)\right]$$

ทำให้เข้าใจบริบทของเสียงพูดทั้งจากอดีตและอนาคตพร้อมกัน

### 2.4.3 Regularization Techniques

| เทคนิค | การทำงาน | ค่าที่ใช้ |
|---|---|---|
| **Dropout** | สุ่มปิด Neurons ระหว่าง Training | 0.3 (30%) |
| **L2 Regularization** | จำกัดขนาด Weight ด้วย Penalty Term | λ = 0.001 |
| **Batch Normalization** | Normalize Activation ของแต่ละ Batch | — |
| **EarlyStopping** | หยุด Training เมื่อ val_loss ไม่ดีขึ้น | patience = 10 |
| **ReduceLROnPlateau** | ลด Learning Rate เมื่อ val_loss หยุดนิ่ง | factor = 0.5 |

## 2.5 ชุดข้อมูลที่เกี่ยวข้อง

### 2.5.1 RAVDESS (ภาษาอังกฤษ)

- **ชื่อเต็ม:** Ryerson Audio-Visual Database of Emotional Speech and Song
- **ผู้จัดทำ:** Livingstone & Russo (2018)
- **ผู้แสดง:** 24 นักแสดงมืออาชีพ (12 ชาย, 12 หญิง)
- **สภาพแวดล้อม:** บันทึกในห้อง Anechoic Chamber ควบคุมเสียงสะท้อน
- **รูปแบบ:** ไฟล์ WAV, Mono, 48kHz
- **การ Label:** ผ่านระบบ Filename Code `XX-XX-[Emotion]-XX-XX-XX-XX.wav`

```python
# ตัวอย่าง Filename Mapping ที่ใช้ในโครงงาน
RAVDESS_MAP = {
    '01': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry'
}
# ไฟล์ 03-01-05-01-01-01-12.wav → parts[2] = '05' → Angry
```

### 2.5.2 Korean Voice Emotion Dataset (ภาษาเกาหลี)

- **แหล่งที่มา:** Hugging Face Datasets (Streaming Mode)
- **รูปแบบ:** ข้อมูล Audio bytes แปลงเป็น WAV ด้วย soundfile
- **การจัดเก็บ:** จัดไว้ในโฟลเดอร์ตามอารมณ์ผ่าน [Label.py](SeniorP1/รวมภาษา Ai/Label.py)

### 2.5.3 โครงสร้างชุดข้อมูลรวม

```
dataset/
├── angry/      ← ไฟล์ WAV อารมณ์โกรธ (ภาษาอังกฤษ + เกาหลีรวมกัน)
├── happy/      ← ไฟล์ WAV อารมณ์มีความสุข
├── sad/        ← ไฟล์ WAV อารมณ์เศร้า
├── neutral/    ← ไฟล์ WAV อารมณ์เป็นกลาง
└── surprise/   ← ไฟล์ WAV อารมณ์ประหลาดใจ
```

**ปัญหาที่ตามมาจากโครงสร้างนี้:** การรวมไฟล์จากสองภาษาไว้ในโฟลเดอร์เดียวกันทำให้โมเดลไม่รู้ว่าแต่ละไฟล์มาจากภาษาอะไร จึงไม่สามารถปรับพฤติกรรมตามภาษาได้

## 2.6 สรุปผลการศึกษางานวิจัยที่เกี่ยวข้อง

จากการทบทวนวรรณกรรม สามารถสรุปได้ว่า:

1. **Multilingual SER เป็นปัญหาที่ยังเปิดอยู่ (Open Problem)** — งานวิจัยส่วนใหญ่ยังคงใช้ Per-Language Model หรือ Fine-tuning แยกภาษา
2. **MFCC ไม่เพียงพอสำหรับ Cross-lingual SER** — ต้องใช้ Feature ที่ Language-Neutral มากกว่า เช่น Wav2Vec 2.0 หรือ HuBERT
3. **Data Imbalance เป็นปัญหาสำคัญ** — ภาษาอังกฤษมี Open Dataset มากกว่าภาษาเอเชียมาก

---

# บทที่ 3 วิธีการดำเนินการวิจัย

## 3.1 ภาพรวมสถาปัตยกรรมระบบ

ระบบ Multilingual SER แบ่งการทำงานออกเป็น 2 Pipeline หลัก คือ **Training Pipeline** (กระบวนการฝึกสอนโมเดล) และ **Inference Pipeline** (กระบวนการทำนายอารมณ์จากไฟล์เสียงใหม่) ทั้งสอง Pipeline ใช้ขั้นตอน Preprocessing และ Feature Extraction ที่เหมือนกันทุกประการเพื่อให้ผลลัพธ์สอดคล้องกัน

### Training Pipeline

| ลำดับ | ขั้นตอน | รายละเอียด | เครื่องมือที่ใช้ |
|---|---|---|---|
| 1 | **File Discovery** | สแกนโฟลเดอร์ Dataset แบบ Recursive เพื่อรวบรวมไฟล์เสียงทุกชนิด | `os.walk()` |
| 2 | **Label Detection** | ตรวจจับอารมณ์จากชื่อโฟลเดอร์ (Korean) หรือชื่อไฟล์ (RAVDESS) | Path/Filename Parsing |
| 3 | **Audio Load** | โหลดไฟล์เสียงและแปลงเป็น NumPy Array ที่ Sample Rate มาตรฐาน | `librosa.load()` |
| 4 | **Silence Removal** | ตัดความเงียบหัวท้ายออกเพื่อลด Noise ที่ไม่มีข้อมูล | `librosa.effects.trim()` |
| 5 | **Pad / Cut** | ปรับความยาวสัญญาณให้คงที่ 3 วินาที (66,150 samples) | `np.pad()` |
| 6 | **Feature Extraction** | สกัด MFCC จากสัญญาณเสียง ได้ Matrix ขนาด (T, 40) | `librosa.feature.mfcc()` |
| 7 | **Data Augmentation** | สร้างตัวอย่างเพิ่มจาก Train Set เท่านั้น (Noise / Pitch / Time) | `librosa` effects |
| 8 | **Train/Val/Test Split** | แบ่งข้อมูลก่อนทุกกระบวนการเพื่อป้องกัน Data Leakage | `train_test_split()` |
| 9 | **StandardScaler** | Normalize Feature ให้มีค่าเฉลี่ย 0 และ Std 1 (Fit บน Train เท่านั้น) | `StandardScaler` |
| 10 | **Model Training** | ฝึกสอน CNN + Bi-LSTM พร้อม EarlyStopping และ ReduceLROnPlateau | `model.fit()` |
| 11 | **Save Best Model** | บันทึกโมเดลที่ดีที่สุด (val_accuracy สูงสุด) พร้อม Scaler | `.keras` + `.pkl` |

### Inference Pipeline

| ลำดับ | ขั้นตอน | รายละเอียด | หมายเหตุ |
|---|---|---|---|
| 1 | **Audio Input** | รับไฟล์เสียงรูปแบบ WAV / MP3 / FLAC | ไฟล์ใหม่ที่ไม่เคยเห็นระหว่าง Train |
| 2 | **Preprocessing** | Trim → Pad/Cut ให้ยาว 3 วินาที เหมือน Training | ต้องใช้ parameter เดิมทุกอย่าง |
| 3 | **Feature Extraction** | สกัด MFCC 40 coefficients | ต้องใช้ `n_mfcc=40` เหมือนตอน Train |
| 4 | **Normalization** | Transform ด้วย Scaler ที่บันทึกไว้จาก Train | ห้าม Fit ใหม่ — ต้อง Load จาก `.pkl` |
| 5 | **Model Prediction** | ส่ง Feature เข้าโมเดล CNN + Bi-LSTM | โมเดลคำนวณความน่าจะเป็นทั้ง 5 อารมณ์ |
| 6 | **Softmax Output** | ได้ผลลัพธ์เป็น Probability ของแต่ละอารมณ์ รวมกัน = 1.0 | เช่น Angry=0.72, Happy=0.12, ... |

## 3.2 การออกแบบ Data Pipeline

### 3.2.1 Label Detection (2 วิธี)

&emsp;ระบบจำเป็นต้องอ่าน Label (ประเภทอารมณ์) ของไฟล์เสียงแต่ละไฟล์โดยอัตโนมัติ เนื่องจาก Dataset ที่ใช้มีสองแหล่งที่มีรูปแบบการจัดเก็บต่างกันโดยสิ้นเชิง จึงออกแบบระบบตรวจจับ Label ไว้ 2 วิธี ให้ทำงานตามลำดับความสำคัญ

**วิธีที่ 1 — Path-based Detection (สำหรับ Korean Dataset):**

&emsp;Korean Dataset จัดเก็บไฟล์เสียงโดยแยกโฟลเดอร์ตามอารมณ์ เช่น `dataset/angry/file.wav` ดังนั้นระบบจะตรวจสอบว่า path ของไฟล์มีชื่อโฟลเดอร์ที่ตรงกับคำสำคัญของอารมณ์หรือไม่ โดยแปลง path เป็นตัวพิมพ์เล็กและเปลี่ยน backslash เป็น forward slash ก่อน เพื่อให้การเปรียบเทียบทำงานได้ถูกต้องบนทุก OS จากนั้นวนลูปตรวจสอบกับ Dictionary `EMOTION_KEYWORDS` ทีละคำ ถ้าพบคำใดอยู่ในรูปแบบ `/keyword/` ในสาย path ก็กำหนด Label ทันทีและหยุดการค้นหา

```python
# แปลง path ให้เป็นมาตรฐาน (lowercase + forward slash)
path_check = file_path.lower().replace('\\', '/')

# EMOTION_KEYWORDS = {'angry': 'angry', 'happy': 'happy', 'sad': 'sad', ...}
for key, emotion_name in EMOTION_KEYWORDS.items():
    # ตรวจสอบว่า path มีโฟลเดอร์ชื่อ /angry/ หรือ /happy/ เป็นต้น
    if f"/{key}/" in path_check:
        label = emotion_name
        break  # พบแล้ว หยุดวนลูป
```

**วิธีที่ 2 — Filename-based Detection (สำหรับ RAVDESS):**

&emsp;RAVDESS Dataset ใช้รูปแบบชื่อไฟล์แบบ Structured Code เช่น `03-01-05-01-01-01-12.wav` โดยแต่ละตำแหน่งที่คั่นด้วยขีด (-) มีความหมายเฉพาะ ตำแหน่งที่ 3 (index 2 เมื่อนับจาก 0) คือ Emotion Code ได้แก่ `01` = neutral, `03` = happy, `04` = sad, `05` = angry ระบบจะแยกชื่อไฟล์ด้วย `.split('-')` แล้วตรวจสอบว่า `parts[2]` อยู่ใน RAVDESS_MAP หรือไม่ ถ้าใช่จะดึง Label ออกมาจาก Dictionary

```python
# RAVDESS_MAP จับคู่ Emotion Code กับชื่ออารมณ์
RAVDESS_MAP = {'01': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry'}

filename = os.path.basename(path_check)   # ดึงเฉพาะชื่อไฟล์
parts = filename.split('-')               # ['03','01','05','01','01','01','12.wav']

# ตรวจสอบว่าไฟล์มีรูปแบบ RAVDESS และ Emotion Code อยู่ใน Map
if len(parts) >= 3 and parts[2] in RAVDESS_MAP:
    label = RAVDESS_MAP[parts[2]]         # parts[2]='05' → label='angry'
```

&emsp;ระบบจะลองวิธีที่ 1 ก่อน หากไม่พบจึงลองวิธีที่ 2 และหากทั้งสองวิธีไม่สามารถระบุ Label ได้ ไฟล์นั้นจะถูกข้ามไปโดยไม่นำเข้า Dataset เพื่อป้องกันข้อมูลที่ไม่มี Label ปะปนกับข้อมูลที่ถูกต้อง

---

### 3.2.2 Preprocessing Pipeline

&emsp;ก่อนที่จะสกัด Feature จากไฟล์เสียง จำเป็นต้องผ่านขั้นตอน Preprocessing เพื่อทำให้ข้อมูลทุกไฟล์อยู่ในรูปแบบมาตรฐานเดียวกัน เนื่องจากไฟล์เสียงใน Dataset มีความยาวต่างกัน มี Sample Rate ต่างกัน และบางไฟล์มีความเงียบที่ต้นและปลายเสียง ซึ่งจะทำให้ Feature ที่สกัดออกมามีขนาดไม่เท่ากันและมี Noise ที่ไม่จำเป็น ขั้นตอน Preprocessing มีทั้งหมด 3 ขั้น ดังนี้

**ขั้นที่ 1 — โหลดไฟล์เสียง และแปลง Sample Rate:**

&emsp;ใช้ `librosa.load()` โหลดไฟล์เสียงและแปลงเป็น NumPy Array โดยบังคับ Sample Rate ให้เป็น 22,050 Hz เสมอ ไม่ว่าไฟล์ต้นฉบับจะบันทึกที่ความถี่ใดก็ตาม นอกจากนี้ยังกำหนด `duration=3` เพื่อตัดข้อมูลให้ไม่เกิน 3 วินาทีตั้งแต่ขั้นตอนการโหลด ซึ่งช่วยประหยัดหน่วยความจำสำหรับไฟล์ที่ยาวมาก

```python
# sr=22050 → บังคับ Resample เป็น 22,050 Hz ทุกไฟล์
# duration=3 → ตัดให้ไม่เกิน 3 วินาทีตั้งแต่ต้น (ประหยัด RAM)
data, sr = librosa.load(file_path, sr=22050, duration=3)
# data = NumPy Array รูปร่าง (N_samples,) เช่น (66150,) สำหรับ 3 วินาที
# sr   = 22050 เสมอ (ค่าที่เราบังคับ)
```

**ขั้นที่ 2 — ตัดความเงียบ (Silence Trimming):**

&emsp;หลังจากโหลดแล้ว ไฟล์เสียงอาจมีความเงียบที่หัวและท้าย (เช่น ช่วงที่นักแสดงหายใจก่อนพูด หรือช่วงหลังจากพูดจบ) ความเงียบเหล่านี้ไม่มีข้อมูล Emotion ใดๆ แต่จะทำให้ MFCC ของแต่ละไฟล์มีรูปแบบต่างกันโดยไม่จำเป็น `librosa.effects.trim()` จะตัดส่วนที่มีพลังงานต่ำกว่า `top_db=25` ออก (25 dB ต่ำกว่า Peak ของสัญญาณ) แล้วคืนเฉพาะส่วนที่มีเสียงจริงๆ

```python
# top_db=25 หมายความว่า ตัดส่วนที่เงียบกว่า Peak 25 dB ออก
# _ คือ index ของส่วนที่เหลือ (ไม่ได้ใช้)
data, _ = librosa.effects.trim(data, top_db=25)
# ผลลัพธ์: data มีความยาวสั้นลง เหลือเฉพาะส่วนที่มีเสียงพูดจริง
```

**ขั้นที่ 3 — ปรับความยาวให้เท่ากันทุกไฟล์ (Pad / Cut):**

&emsp;โมเดล CNN + Bi-LSTM ต้องการ Input ที่มีขนาดคงที่ทุก Batch ดังนั้นหลังจากตัดความเงียบแล้ว ต้องปรับความยาวสัญญาณให้เท่ากับ 3 วินาทีพอดี (66,150 samples = 22,050 Hz × 3 วินาที) โดยถ้าสัญญาณสั้นกว่า 66,150 samples จะเติม 0 ต่อท้าย (Zero Padding) และถ้ายาวกว่าจะตัดเอาเฉพาะ 66,150 samples แรก

```python
SAMPLES_PER_TRACK = 22050 * 3  # = 66,150 samples

if len(data) < SAMPLES_PER_TRACK:
    # สัญญาณสั้นกว่า 3 วินาที → เติม 0 ต่อท้าย (Zero Padding)
    # (0, SAMPLES_PER_TRACK - len(data)) = เติมด้านขวาเท่านั้น
    data = np.pad(data, (0, SAMPLES_PER_TRACK - len(data)), 'constant')
else:
    # สัญญาณยาวกว่า 3 วินาที → ตัดเอาแค่ 3 วินาทีแรก
    data = data[:SAMPLES_PER_TRACK]
# ผลลัพธ์: data.shape = (66150,) เสมอ ไม่ว่าต้นฉบับจะยาวแค่ไหน
```

---

### 3.2.3 การป้องกัน Data Leakage (Critical Process)

**Data Leakage คืออะไร และทำไมถึงอันตราย**

&emsp;Data Leakage คือสถานการณ์ที่ข้อมูลจากชุด Test Set "รั่วไหล" เข้าไปมีอิทธิพลต่อกระบวนการสร้างโมเดล ทำให้โมเดลได้ "ดู" ข้อมูลที่ควรจะเป็นข้อมูลทดสอบไปแล้วบางส่วนก่อนที่จะถูกทดสอบจริง ผลที่ตามมาคือ Accuracy ที่วัดได้สูงกว่าความเป็นจริง ทำให้ประเมินประสิทธิภาพของโมเดลผิดพลาดอย่างมีนัยสำคัญ และเมื่อนำโมเดลไปใช้งานจริงกับข้อมูลที่ไม่เคยเห็นมาก่อน ประสิทธิภาพจะต่ำกว่าที่รายงานไว้มาก

**แหล่งที่มาของ Data Leakage ในโครงงานนี้**

&emsp;ใน Pipeline ของงาน SER มีโอกาสเกิด Data Leakage ได้ 2 จุดหลักคือ

&emsp;**จุดที่ 1 — Data Augmentation ก่อน Split:** หากนำข้อมูลทั้งหมดมา Augment ก่อน แล้วค่อย Split โดยสมมติว่ามีไฟล์ต้นฉบับ A อยู่ในชุดข้อมูล เมื่อ Augment จะได้ไฟล์ A, A_noise, A_pitch, A_stretch จากนั้นเมื่อ Split แบบสุ่ม อาจเกิดกรณีที่ A อยู่ใน Test Set แต่ A_noise หรือ A_pitch ซึ่งสร้างมาจากไฟล์เดียวกันกลับอยู่ใน Train Set โมเดลจึงได้ฝึกกับข้อมูลที่แทบเหมือนกันกับข้อมูล Test ทำให้ผลประเมินสูงเกินจริง

&emsp;**จุดที่ 2 — StandardScaler Fit บนข้อมูลรวม:** หากนำข้อมูล Train + Test + Val ทั้งหมดมา Fit Scaler พร้อมกัน ค่า Mean และ Standard Deviation ที่คำนวณได้จะมีข้อมูลของ Test ปนอยู่ด้วย โมเดลจึงทราบสถิติของข้อมูล Test ล่วงหน้าโดยอ้อม ซึ่งถือเป็น Leakage เช่นกัน

**วิธีที่ผิด (เกิด Data Leakage):**

| ลำดับ | ขั้นตอน | ปัญหาที่เกิด |
|---|---|---|
| 1 | โหลดไฟล์เสียงทั้งหมดพร้อมกัน | — |
| 2 | Augment ข้อมูลทั้งหมด (Train + Test รวมกัน) | ไฟล์ Augment ของ Train ปะปนกับ Test |
| 3 | Fit Scaler บนข้อมูลรวมทุกชุด | Scaler "เรียนรู้" สถิติของ Test ล่วงหน้า |
| 4 | แบ่ง Train / Val / Test | แบ่งช้าเกินไป — Leakage เกิดขึ้นแล้ว |
| 5 | Train โมเดล → วัด Accuracy | ผล Accuracy สูงเกินจริง ไม่สะท้อนประสิทธิภาพจริง |

**วิธีที่ถูก (ป้องกัน Data Leakage ทุกจุด):**

| ลำดับ | ขั้นตอน | เหตุผลที่สำคัญ |
|---|---|---|
| 1 | รวบรวม **File Paths** (ยังไม่โหลดข้อมูล) | ทำงานกับ Path เท่านั้น ไม่ยุ่งกับเนื้อหาไฟล์ |
| 2 | **Split File Paths** → train_files / val_files / test_files | แบ่งก่อน ตั้งแต่ระดับ Path เพื่อรับประกันว่าไฟล์เดียวกันจะไม่อยู่คนละ Set |
| 3 | **Augment เฉพาะ train_files** (สร้าง 3 ตัวอย่างต่อไฟล์) | val/test ไม่ถูก Augment ไม่มีทางที่ Augmented version จะปนเข้า Test |
| 4 | โหลด val_files และ test_files แบบ **Original** (ไม่ Augment) | Test ใช้ข้อมูลจริงตามที่เป็น ไม่ผ่านการดัดแปลงใดๆ |
| 5 | **Fit StandardScaler บน X_train เท่านั้น** | Scaler เรียนรู้เฉพาะสถิติของ Train ไม่รู้จัก Val/Test เลย |
| 6 | **Transform** X_val และ X_test ด้วย Scaler ที่ Fit ไว้แล้ว | ใช้ค่า Mean/Std จาก Train มา Normalize Val/Test เท่านั้น |
| 7 | **บันทึก Scaler** ไว้ในไฟล์ `.pkl` | ตอน Inference ต้อง Load Scaler เดิมมาใช้ ห้ามสร้าง Scaler ใหม่ |

**ผลกระทบของการป้องกัน Data Leakage:**

&emsp;เมื่อดำเนินการอย่างถูกต้องตามขั้นตอนข้างต้น ผล Accuracy ที่วัดได้บน Test Set จะสะท้อนประสิทธิภาพที่แท้จริงของโมเดลเมื่อนำไปใช้กับข้อมูลใหม่ที่ไม่เคยเห็นมาก่อน แม้ตัวเลข Accuracy อาจดูต่ำกว่าระบบที่มี Leakage แต่ถือว่าเชื่อถือได้และนำไปเปรียบเทียบกับงานวิจัยอื่นได้อย่างยุติธรรม ในโครงงานนี้ค่า Test Accuracy ที่วัดได้ประมาณ 68% จึงเป็นค่าที่เชื่อถือได้จริง ไม่ใช่ค่าที่ถูกเพิ่มขึ้นจาก Leakage

---

### 3.2.4 Data Augmentation

&emsp;Data Augmentation คือกระบวนการสร้างตัวอย่างข้อมูลเพิ่มเติมจากข้อมูลที่มีอยู่ โดยดัดแปลงในรูปแบบที่ยังคงความหมาย (Label) ไว้เหมือนเดิม เพื่อเพิ่มความหลากหลายของข้อมูล Train และช่วยให้โมเดล Generalize ได้ดีขึ้น

**ตารางที่ 3.3 เทคนิค Data Augmentation ที่ใช้**

| เทคนิค | วิธีการ | สูตร | จุดประสงค์ |
|---|---|---|---|
| **Gaussian Noise** | เพิ่ม Noise แบบสุ่มเข้าสัญญาณ | $x' = x + \alpha\mathcal{N}(0,1)$ | ทำให้โมเดลทนต่อ Background Noise ในสภาพแวดล้อมจริง |
| **Pitch Shifting** | ปรับระดับ Pitch ขึ้น/ลง ±0.7 Semitones | `librosa.effects.pitch_shift(n_steps=±0.7)` | ทำให้โมเดลทนต่อความแตกต่างของระดับเสียงระหว่างผู้พูดแต่ละคน |
| **Time Stretching** | ยืดหรือหดเวลาของสัญญาณ (rate=0.8) | `librosa.effects.time_stretch(rate=0.8)` | ทำให้โมเดลทนต่อความเร็วในการพูดที่แตกต่างกัน |

&emsp;ผลลัพธ์ของ Augmentation คือไฟล์เสียง 1 ไฟล์จะถูกแปลงเป็น 3 ตัวอย่าง ได้แก่ ตัวอย่างที่เพิ่ม Noise / ตัวอย่างที่ปรับ Pitch / ตัวอย่างที่ยืดเวลา ทำให้ขนาดของ Train Set เพิ่มขึ้นเป็น 3 เท่า โดยไม่ต้องเก็บข้อมูลเพิ่ม สำคัญที่สุดคือ Augmentation ทำเฉพาะกับ **Train Set เท่านั้น** ตาม Anti-Data-Leakage Policy ที่อธิบายใน 3.2.3

---

### 3.2.5 Normalization Strategy

&emsp;ก่อนส่งข้อมูลเข้าโมเดล ต้อง Normalize Feature MFCC ให้อยู่ในช่วงที่เหมาะสม เพราะ MFCC แต่ละ Coefficient มีช่วงค่าที่แตกต่างกันมาก เช่น Coefficient ที่ 1 อาจมีค่าในช่วง -200 ถึง +50 ในขณะที่ Coefficient ที่ 10 อาจมีค่าในช่วง -30 ถึง +30 หากไม่ Normalize โมเดลจะให้ความสำคัญกับ Coefficient ที่มีค่าสูงกว่ามากเกินไป ทำให้เรียนรู้ได้ช้าและ Converge ได้ยาก

&emsp;โครงงานนี้ใช้ **StandardScaler** ซึ่งแปลงข้อมูลให้มีค่าเฉลี่ย (Mean) = 0 และ ส่วนเบี่ยงเบนมาตรฐาน (Std) = 1 ตามสูตร:

$$z = \frac{x - \mu}{\sigma}$$

โดยที่ $\mu$ คือค่าเฉลี่ยของ Train Set และ $\sigma$ คือ Standard Deviation ของ Train Set

**กระบวนการ Normalization ที่ถูกต้อง:**

```python
scaler = StandardScaler()
N, T, F = X_train.shape
# N = จำนวนตัวอย่าง, T = Time Steps (130), F = Features (40)

# ขั้นที่ 1: Reshape จาก (N, T, F) เป็น (N, T×F) เพื่อให้ Scaler คำนวณได้
# จากนั้น fit_transform จะ:
#   - คำนวณ Mean และ Std ของแต่ละ Feature จาก Train เท่านั้น
#   - Normalize ข้อมูล Train ด้วยค่า Mean/Std ที่คำนวณได้
#   - Reshape กลับเป็น (N, T, F)
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, F)

# ขั้นที่ 2: Normalize Val ด้วยค่า Mean/Std จาก Train (ห้าม fit ใหม่!)
# scaler.transform ใช้ค่า Mean/Std ที่เรียนรู้จาก Train มา Apply กับ Val
X_val = scaler.transform(X_val.reshape(-1, T*F)).reshape(-1, T, F)

# ขั้นที่ 3: Normalize Test ด้วยค่า Mean/Std จาก Train เช่นกัน
X_test = scaler.transform(X_test.reshape(-1, T*F)).reshape(-1, T, F)

# ขั้นที่ 4: บันทึก Scaler ไว้ใช้ตอน Inference
# เมื่อต้องการทดสอบไฟล์เสียงใหม่ ต้อง Load Scaler นี้มา Transform ก่อนเสมอ
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)  # บันทึกด้วย Pickle เพื่อ Load กลับมาใช้ได้
```

&emsp;เหตุผลที่ต้อง Fit Scaler บน Train Set เท่านั้น เพราะหากนำ Val หรือ Test มา Fit ด้วย ค่า Mean และ Std จะถูกกำหนดจากข้อมูล Test ทำให้โมเดลทราบสถิติของข้อมูลทดสอบล่วงหน้า ซึ่งเป็น Data Leakage รูปแบบหนึ่ง นอกจากนี้การบันทึก Scaler ไว้ในไฟล์ `.pkl` มีความสำคัญอย่างยิ่ง เพราะเมื่อต้องการทำนายอารมณ์จากไฟล์เสียงใหม่ในอนาคต จะต้อง Normalize ด้วยค่า Mean/Std ชุดเดิมกันกับตอน Train เท่านั้น หาก Normalize ด้วยค่าใหม่ที่คำนวณจากไฟล์เสียงเพียงไฟล์เดียว ผลการทำนายจะผิดพลาดอย่างมาก

## 3.3 สถาปัตยกรรมโมเดล EmotionResNet (Per-Language)

&emsp;โมเดลที่ใช้ในโครงงานนี้คือ **EmotionResNet** ซึ่งเป็น Residual Convolutional Neural Network สำหรับการจำแนกอารมณ์จากภาพ Mel Spectrogram รับ Input เป็น Tensor ขนาด **(Batch, 1, 128, 130)** ซึ่งเป็น Single-channel Mel Spectrogram (128 Mel Bands × 130 Time Frames) โมเดลออกแบบมาให้ฝึกสอนแยกสำหรับแต่ละภาษา โดยแต่ละภาษามีจำนวน Output Class ต่างกันตามอารมณ์ที่มีใน Dataset

**หลักการ Residual Network (ResNet):**

&emsp;ResNet (He et al., 2016) แก้ปัญหา Vanishing Gradient ใน Deep Network ด้วย **Skip Connection** ที่นำ Input ของ Block บวกเข้ากับ Output โดยตรง:

$$\text{Output} = \mathcal{F}(x, \{W_i\}) + x$$

ทำให้ Gradient ไหลผ่านเส้นทาง Residual ได้โดยตรง โมเดลเรียนรู้เฉพาะ "ส่วนต่าง" (Residual) จาก Identity Mapping ซึ่ง Converge ได้เร็วกว่าและ Generalize ได้ดีกว่า Plain CNN

**โครงสร้างสถาปัตยกรรม EmotionResNet:**

**ส่วนที่ 1 — Stem Layer (Feature Extraction เบื้องต้น):**
```
Input: (Batch, 1, 128, 130)
Conv2d(1→64, kernel=3, padding=1) → BatchNorm2d(64) → ReLU
MaxPool2d(kernel=2, stride=2)
Output: (Batch, 64, 64, 65)
```

**ส่วนที่ 2 — Residual Block Layer 1 (Low-level Features):**
```
ResBlock: Conv2d(64→64) + BN + ReLU + Conv2d(64→64) + BN + Skip Connection
Output: (Batch, 64, 64, 65)
```

**ส่วนที่ 3 — Residual Block Layer 2 (Mid-level Features):**
```
ResBlock: Conv2d(64→128, stride=2) + BN + ReLU + Conv2d(128→128) + BN
Skip Connection: Conv2d(64→128, stride=2)  [Projection Shortcut]
Output: (Batch, 128, 32, 33)
```

**ส่วนที่ 4 — Residual Block Layer 3 (High-level Features):**
```
ResBlock: Conv2d(128→256, stride=2) + BN + ReLU + Conv2d(256→256) + BN
Skip Connection: Conv2d(128→256, stride=2)  [Projection Shortcut]
Output: (Batch, 256, 16, 17)
```

**ส่วนที่ 5 — Adaptive Pooling + Classifier:**
```
AdaptiveAvgPool2d(output_size=(4,4))
Output: (Batch, 256, 4, 4)

Classifier:
  Flatten → Linear(256*4*4=4096, 256) → ReLU → Dropout(0.4)
  → Linear(256, n_classes)  [n_classes = 4-7 ตามภาษา]
```

**ตารางที่ 3.4 สรุปสถาปัตยกรรม EmotionResNet ชั้นต่อชั้น**

| ส่วน | Layer | Output Shape | บทบาทหน้าที่ |
|---|---|---|---|
| Stem | Conv2d(1→64,k=3)+BN+ReLU+MaxPool | (B, 64, 64, 65) | สกัด Low-level Spectral Feature |
| Layer1 | ResBlock(64→64) | (B, 64, 64, 65) | ปรับ Feature ระดับต่ำ |
| Layer2 | ResBlock(64→128, stride=2) | (B, 128, 32, 33) | Feature ระดับกลาง + Downsampling |
| Layer3 | ResBlock(128→256, stride=2) | (B, 256, 16, 17) | Feature ระดับสูง + Downsampling |
| Pool | AdaptiveAvgPool2d(4,4) | (B, 256, 4, 4) | ย่อเป็น Fixed-size Representation |
| Classifier | Flatten+Linear(4096→256)+ReLU+Drop+Linear(256→n) | (B, n) | จำแนกอารมณ์ n ประเภท |

**เหตุผลในการเลือก ResNet สำหรับ Mel Spectrogram:**

&emsp;Mel Spectrogram มีลักษณะเป็นภาพ 2D (ความถี่ × เวลา) ทำให้สถาปัตยกรรม 2D CNN เหมาะสมกว่า 1D CNN หรือ LSTM ที่รับ Sequential Feature ResNet โดยเฉพาะให้ประสิทธิภาพดีเพราะ:
1. **Skip Connection** ช่วยให้ Gradient ไหลย้อนกลับได้อย่างมีประสิทธิภาพในทุกชั้น ป้องกัน Vanishing Gradient
2. **Residual Learning** ทำให้โมเดลเรียนรู้เฉพาะส่วนที่ต่างจาก Identity ไม่ต้อง Re-learn ทุกอย่าง
3. **AdaptiveAvgPool** ทำให้รับ Input ที่มีขนาดต่างกันได้โดยไม่ต้องแก้ Architecture

**Language Detector (SVM Pipeline):**

&emsp;นอกจาก EmotionResNet สำหรับแต่ละภาษา ระบบยังมี **Language Detector** ที่ใช้ SVM Pipeline:

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=80)),
    ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
])
```

&emsp;Input คือ Mel Spectrogram Flatten เป็น Vector แล้วผ่าน PCA ลดมิติเหลือ 80 Principal Components ก่อนส่งเข้า SVM ผลลัพธ์คือ 5-class Language Prediction (TH/ZH/JA/KO/EN) ด้วย Accuracy **98.54%** ซึ่งสูงกว่า Deep Learning เพราะปัญหา Language ID มี Decision Boundary ที่ชัดเจนกว่า Emotion Recognition

## 3.4 การกำหนด Training Configuration

**ตารางที่ 3.2 Training Configuration และเหตุผลในการเลือกใช้**

| Parameter | ค่าที่ใช้ | เหตุผล |
|---|---|---|
| Sample Rate | **16,000 Hz** | มาตรฐาน Speech Processing สมัยใหม่, ลดขนาดข้อมูล |
| Duration | 3 วินาที | ครอบคลุมอารมณ์ที่แสดงออก, ไม่หนักเกินไป |
| Mel Bands | 128 | ความละเอียดเพียงพอ, ครอบคลุม 0–8000 Hz |
| Time Frames | 130 | จาก hop_length=512 กับ 3 วินาที × 16000 Hz |
| Optimizer | Adam (lr=0.001) | ปรับ Learning Rate อัตโนมัติ |
| Loss Function | Cross-Entropy | Multi-class Classification มาตรฐาน |
| Batch Size | 32 | สมดุลระหว่างความเร็วและ Generalization |
| Max Epochs | 50 | เพียงพอสำหรับข้อมูลขนาดนี้ |
| EarlyStopping | patience=5 | ป้องกัน Overfitting หยุดเร็ว |
| ReduceLROnPlateau | factor=0.5, patience=3 | Fine-tune ใกล้ Optimum |
| Framework | **PyTorch** + torchaudio | Flexibility สูง, Dynamic Graph |

**ตารางที่ 3.3 เทคนิค Data Augmentation ที่ใช้**

| เทคนิค | รายละเอียด | จุดประสงค์ |
|---|---|---|
| **SpecAugment (Frequency Masking)** | Mask แบบสุ่มบนแกนความถี่ (F=27) จำนวน 1-2 แถบ | ทำให้โมเดลทนต่อการสูญเสียข้อมูลบางความถี่ |
| **SpecAugment (Time Masking)** | Mask แบบสุ่มบนแกนเวลา (T=30) จำนวน 1-2 แถบ | ทำให้โมเดลทนต่อการสูญเสียข้อมูลบางช่วงเวลา |
| **Pitch Shift** | ±1, ±2 Semitones | ทำให้โมเดลทนต่อความแตกต่างของระดับเสียง |
| **Time Stretch** | Rate 0.85 และ 1.15 | ทำให้โมเดลทนต่อความเร็วในการพูดที่ต่างกัน |
| **Gaussian Noise** | SNR = 20 dB | ทำให้โมเดลทนต่อ Background Noise |

**ตารางที่ 3.1 ไฟล์หลักใน Pipeline ระบบ Phase 2**

| ไฟล์ | หน้าที่ | Output |
|---|---|---|
| [data_pipeline.py](SeniorP1/แยกภาษา/data_pipeline.py) | สกัด Mel Spectrogram ทุกภาษา | `{Lang}_mel_features.npy` |
| [language_detector.py](SeniorP1/แยกภาษา/language_detector.py) | Train/Eval Language Identifier SVM | `lang_detector.pkl` |
| [train_per_language.py](SeniorP1/แยกภาษา/train_per_language.py) | Train EmotionResNet แยกต่อภาษา | `{Lang}_model.pt`, `{Lang}_label_encoder.pkl` |
| [evaluate.py](SeniorP1/แยกภาษา/evaluate.py) | Batch Evaluation ทุกภาษา | `results/evaluation_report.txt` |
| [baseline_comparison.py](SeniorP1/แยกภาษา/baseline_comparison.py) | เปรียบ ResNet กับ SVM/RF/GB | `results/baseline_comparison/` |
| [crosslingual_analysis.py](SeniorP1/แยกภาษา/crosslingual_analysis.py) | วิเคราะห์ Feature Space ข้ามภาษา | `results/crosslingual/` |
| [wav2vec2_pipeline.py](SeniorP1/แยกภาษา/wav2vec2_pipeline.py) | สกัด Wav2Vec2 embeddings + Train MLP | `{Lang}_w2v_model.pt` |
| [web_demo.py](SeniorP1/แยกภาษา/web_demo.py) | Gradio Web Interface | URL localhost:7860 |
| [run_all.py](SeniorP1/แยกภาษา/run_all.py) | รัน Pipeline ทั้งหมดต่อเนื่อง | — |

---

# บทที่ 4 ระบบต้นแบบ

## 4.1 ภาพรวมการออกแบบ (Overview Design)

### 4.1.1 ภาพรวมสถาปัตยกรรมระบบ Phase 2

&emsp;ระบบ Phase 2 ใช้สถาปัตยกรรม **Per-Language Model** ที่ประกอบด้วย 3 ส่วนหลัก ดังนี้

```
[ไฟล์เสียงใหม่]
       │
       ▼
[Mel Spectrogram Extraction]
  SR=16kHz, n_mels=128, hop=512, T=130
       │
       ▼
[Language Detector — SVM Pipeline]
  StandardScaler → PCA(80) → SVM(RBF)
  Accuracy: 98.54%
       │
       ├─── ภาษาไทย  ──► [Thai EmotionResNet]    88.40% (4 emotions)
       ├─── ภาษาจีน  ──► [Chinese EmotionResNet] 86.62% (6 emotions)
       ├─── ภาษาญี่ปุ่น ─► [Japan EmotionResNet]  84.36% (6 emotions)
       ├─── ภาษาอังกฤษ ─► [English EmotionResNet] 82.84% (7 emotions)
       └─── ภาษาเกาหลี ─► [Korean EmotionResNet]  53.57% (5 emotions)
                           [Korean SVM Baseline]   72.62% ← ดีกว่า
       │
       ▼
[Predicted Emotion + Confidence Score]
```

### 4.1.2 คำอธิบายการทำงานของแต่ละไฟล์ Python (Phase 2)

&emsp;โครงงานนี้ประกอบด้วยไฟล์ Python หลัก ดังนี้

---

**กลุ่มที่ 1 — Data Pipeline (เตรียมข้อมูล)**

**1. data_pipeline.py**

&emsp;ไฟล์หลักสำหรับสกัด Mel Spectrogram จากไฟล์เสียงทุกภาษา กระบวนการทำงาน: โหลดไฟล์เสียง → Resample เป็น 16,000 Hz → Trim Silence → Pad/Cut ให้ยาว 3 วินาที → คำนวณ Mel Spectrogram (n_mels=128, hop_length=512) → แปลงเป็น dB Scale → บันทึกเป็น `.npy` array รูปร่าง (1, 128, 130) นอกจากนี้ยังรวม Label Detection จากชื่อโฟลเดอร์และชื่อไฟล์ ป้องกัน Data Leakage ด้วยการ Split ก่อน Augmentation

**2. download_extra_data.py**

&emsp;ไฟล์สำหรับหาและดาวน์โหลดข้อมูลเพิ่มเติมสำหรับภาษาที่มีข้อมูลน้อย มีสองกลยุทธ์: กลยุทธ์ที่ 1 ค้นหา Dataset จาก Hugging Face Hub API, กลยุทธ์ที่ 2 (Fallback) สร้าง Audio Augmentation จากข้อมูลที่มีอยู่ (Pitch Shift ±1/±2 Semitones, Time Stretch 0.85/1.15, Gaussian Noise SNR=20dB)

---

**กลุ่มที่ 2 — Language Detection**

**3. language_detector.py**

&emsp;สร้างและประเมิน Language Identifier SVM Pipeline กระบวนการ: โหลด Mel Spectrogram ทุกภาษา → Flatten เป็น Vector → StandardScaler → PCA(80) → SVM(RBF, C=10) → Train/Test Split 80/20 → บันทึกเป็น `lang_detector.pkl` ผลลัพธ์: 98.54% accuracy บน 5 ภาษา

---

**กลุ่มที่ 3 — Training โมเดล**

**4. train_per_language.py**

&emsp;Train EmotionResNet สำหรับแต่ละภาษาแยกกัน กระบวนการ: โหลด Mel Spectrogram ของแต่ละภาษา → Train/Val/Test Split (70/15/15) → สร้าง EmotionResNet ด้วย n_classes ตามภาษา → Train ด้วย Adam + ReduceLROnPlateau + EarlyStopping → บันทึกโมเดลที่ดีที่สุด (`{Lang}_model.pt`) พร้อม Label Encoder

**5. wav2vec2_pipeline.py**

&emsp;สกัด Wav2Vec2 Embeddings จาก `facebook/wav2vec2-large-xlsr-53` สำหรับภาษาที่มีไฟล์เสียง (TH, EN, KO) แล้ว Train MLP Classifier บน Embeddings ที่ได้ เปรียบเทียบกับ ResNet เพื่อดูว่า Pre-trained Speech Representation ช่วยได้หรือไม่ ใช้ `Wav2Vec2FeatureExtractor` แทน `Wav2Vec2Processor` เพราะ `facebook/wav2vec2-large-xlsr-53` ไม่มี vocab.json

---

**กลุ่มที่ 4 — ประเมินผลและวิเคราะห์**

**6. evaluate.py**

&emsp;Batch Evaluation ทุกภาษาพร้อมกัน โหลด Test Set ที่บันทึกไว้ (`{Lang}_X_test.npy`, `{Lang}_y_test.npy`) → โหลดโมเดลแต่ละภาษา → คำนวณ Accuracy, F1-Score, Classification Report → บันทึกรายงาน `results/evaluation_report.txt`

**7. baseline_comparison.py**

&emsp;เปรียบเทียบ EmotionResNet กับ Classical ML Baselines (SVM, Random Forest, Gradient Boosting) บน Mel Spectrogram Features แบบ Flatten + PCA ผลบันทึกใน `results/baseline_comparison/`

**8. crosslingual_analysis.py**

&emsp;วิเคราะห์ความคล้ายคลึงระหว่างภาษาต่างๆ ในระดับ Feature Space สร้าง Radar Charts, Similarity Matrices และ Within/Between Language Distance Plots บันทึกผลใน `results/crosslingual/`

---

**กลุ่มที่ 5 — Web Demo และระบบอัตโนมัติ**

**9. web_demo.py**

&emsp;Web Interface แบบ Interactive สร้างด้วย Gradio 6.x ผู้ใช้สามารถอัพโหลดหรือบันทึกเสียง → ระบบตรวจจับภาษาอัตโนมัติ → ทำนายอารมณ์ → แสดง Confidence Scores พร้อม Bar Chart สำหรับทุกอารมณ์ รันที่ `http://localhost:7860`

**10. run_all.py**

&emsp;Runner Script ที่รัน Pipeline ทั้งหมดต่อเนื่อง มี Flags ให้ข้ามขั้นตอนที่ทำเสร็จแล้ว (`--skip-download`, `--skip-extract`, `--skip-train`, `--skip-w2v`, `--skip-crosslingual`) แสดง Status Table สรุปผลทุกขั้นตอนเมื่อเสร็จสิ้น

### 4.1.2 รายละเอียด Parameter ของโมเดล EmotionResNet

&emsp;EmotionResNet มีโครงสร้างที่กะทัดรัดแต่มีประสิทธิภาพสูง ด้านล่างแสดงจำนวน Parameter โดยประมาณ (สำหรับ n_classes=5):

**ตารางที่ 4.0 จำนวน Parameter แยกตามส่วนของ EmotionResNet**

| ส่วน | รายละเอียด | Parameters (approx.) |
|---|---|---|
| Stem | Conv2d(1,64,3)+BN+MaxPool | ~1,856 |
| Layer1 | ResBlock(64→64) ×2 | ~148,736 |
| Layer2 | ResBlock(64→128, stride=2) | ~230,656 |
| Layer3 | ResBlock(128→256, stride=2) | ~919,808 |
| Classifier | Linear(4096→256)+Dropout+Linear(256→n) | ~1,049,856 |
| **รวมทั้งหมด** | | **~2,350,000** |

&emsp;ขนาดโมเดลประมาณ 2.3 ล้าน Parameter ต่อโมเดลต่อภาษา เมื่อรวมทั้ง 5 ภาษาจะมีโมเดลรวมประมาณ 11.5 ล้าน Parameter (ไม่รวม Language Detector SVM)

### 4.1.3 ลำดับการรันระบบ

```bash
# รันทั้ง Pipeline ครั้งแรก
python run_all.py

# รัน Pipeline โดยข้ามขั้นตอนที่ทำเสร็จแล้ว
python run_all.py --skip-download --skip-extra-data --skip-extract

# รัน Web Demo
python web_demo.py --port 7860
```
## 4.2 ผลการทดลองและการวิเคราะห์

### 4.2.1 ผลการทดลองหลัก — Per-Language ResNet

**ตารางที่ 4.1 ผลการทดลองของ Per-Language ResNet (Test Set)**

| ภาษา | จำนวน Test Samples | จำนวนอารมณ์ | Accuracy | F1-Macro | F1-Weighted |
|---|---|---|---|---|---|
| Thai | 6,042 | 4 | **88.40%** | 88.49% | 88.41% |
| Chinese | 538 | 6 | **86.62%** | 86.29% | 86.55% |
| Japan | 243 | 6 | **84.36%** | 84.34% | 84.31% |
| English | 2,285 | 7 | **82.84%** | 81.53% | 82.76% |
| Korean | 84 | 5 | **53.57%** | 47.77% | 52.86% |
| **เฉลี่ย** | | | **79.16%** | | **78.98%** |
| **เป้าหมาย** | | | **75%** | | |
| **ผ่าน?** | | | **✅ บรรลุ** | | |

### 4.2.2 เปรียบเทียบ ResNet กับ Classical ML Baseline

**ตารางที่ 4.2 เปรียบเทียบ Accuracy ระหว่าง ResNet และ Classical ML**

| ภาษา | SVM | Random Forest | Gradient Boosting | ResNet | ผลที่ดีที่สุด |
|---|---|---|---|---|---|
| Chinese | 81.8% | 66.7% | 68.6% | **86.6%** | ResNet ✅ |
| Japanese | 75.3% | 66.7% | 65.0% | **84.4%** | ResNet ✅ |
| Korean | **72.6%** | 61.9% | 54.8% | 53.6% | SVM ✅ |
| Thai | 63.9% | 58.0% | 58.6% | **88.4%** | ResNet ✅ |
| English | 50.8% | 56.8% | 55.8% | **82.8%** | ResNet ✅ |

**การวิเคราะห์:**
- **ResNet ชนะ 4/5 ภาษา** ด้วยส่วนต่างที่ชัดเจน (Thai: +24.5%, Chinese: +4.8%, Japanese: +9.1%, English: +26.0%)
- **Korean เป็นข้อยกเว้น** — SVM (72.6%) ชนะ ResNet (53.6%) เนื่องจากมีข้อมูลเพียง ~420 ตัวอย่าง เมื่อมีข้อมูลน้อยกว่า 1,000 ตัวอย่างต่อ Class Classical ML มักให้ผลดีกว่า Deep Learning

### 4.2.3 วิเคราะห์ Korean — กรณีพิเศษ

&emsp;ภาษาเกาหลีมีจำนวนข้อมูลน้อยมากเมื่อเทียบกับภาษาอื่น (Test set เพียง 84 ตัวอย่าง, ข้อมูลทั้งหมด ~420 ตัวอย่าง) ทำให้ ResNet ซึ่งมี ~2.3 ล้าน Parameter ต้องเรียนรู้จากข้อมูลน้อยเกินไป ส่งผลให้ Overfit

&emsp;สำหรับ Korean จึงแนะนำให้ใช้ **Korean SVM Classifier (72.62%)** แทน ResNet ในการ Production ซึ่งเป็นการตัดสินใจตาม Empirical Evidence ไม่ใช่ทฤษฎี ข้อค้นพบนี้สอดคล้องกับหลักการ **"No Free Lunch" ของ Machine Learning** ที่ระบุว่าไม่มีอัลกอริทึมที่ดีที่สุดสำหรับทุกสถานการณ์

**ตารางที่ 4.3 เปรียบเทียบโมเดลสำหรับ Korean**

| โมเดล | Accuracy | F1-Macro | สาเหตุ |
|---|---|---|---|
| SVM (RBF) | **72.62%** | 0.58 | เหมาะกับข้อมูลน้อย, Decision Boundary ชัด |
| ResNet | 53.57% | 0.48 | Overfit เพราะ Parameter มากเกินข้อมูล |
| Wav2Vec2-MLP | ~39.76% | — | Pre-trained features ไม่ match Korean data น้อย |

## 4.3 การวิเคราะห์ผลลัพธ์รายอารมณ์

### 4.3.1 Thai (88.40%) — ผลดีที่สุด

&emsp;ภาษาไทยให้ผลดีที่สุดเพราะมีข้อมูลมากที่สุด (~30,210 ตัวอย่าง) ทำให้ ResNet เรียนรู้ได้ดีมาก อารมณ์ทั้ง 4 ที่มีใน Dataset ได้แก่ Angry, Happy, Neutral, Sad มีความแตกต่างของ Mel Spectrogram Pattern ที่ชัดเจน

### 4.3.2 English (82.84%) — 7 อารมณ์

&emsp;ภาษาอังกฤษมีอารมณ์มากที่สุด 7 ประเภท จาก RAVDESS+CREMA-D รวม ~11,425 ตัวอย่าง ทั้ง Disgust และ Fear ซึ่งเป็นอารมณ์ที่ยากได้ F1-Score ต่ำกว่า (0.65 สำหรับ Fear) ขณะที่ Happy และ Neutral ได้สูงกว่า (0.87-0.90)

### 4.3.3 ข้อค้นพบสำคัญจาก Cross-lingual Analysis

&emsp;การวิเคราะห์ Mel Spectrogram Feature Space ข้ามภาษาพบว่า:
- อารมณ์ **Angry** มีความคล้ายกันสูงข้ามภาษา (High-energy, broadband pattern)
- อารมณ์ **Neutral** แตกต่างกันมากข้ามภาษา (ขึ้นกับ baseline prosody ของแต่ละภาษา)
- **Within-language distance** (ความแตกต่างระหว่างอารมณ์ในภาษาเดียว) น้อยกว่า **Between-language distance** สำหรับอารมณ์เดียวกัน ซึ่งเป็นหลักฐานเชิงประจักษ์ที่สนับสนุนแนวทาง Per-Language Model

---

# บทที่ 5 สรุปผลการดำเนินงาน

## 5.1 สรุปผลการดำเนินงาน

&emsp;โครงงานนี้พัฒนา **ระบบรู้จำอารมณ์จากเสียงพูดแบบรองรับหลายภาษา** โดยใช้สถาปัตยกรรม **Per-Language ResNet** ที่แก้ปัญหา Prosody Mismatch ซึ่งเป็นข้อจำกัดพื้นฐานของแนวทาง Unified Multilingual Model ระบบรองรับ 5 ภาษา ได้แก่ ไทย จีน ญี่ปุ่น เกาหลี และอังกฤษ โดยมี Language Detector SVM ที่แม่นยำ 98.54% เป็นตัวคัดเลือกโมเดล

**ผลการดำเนินงานในแต่ละด้าน:**

&emsp;**ด้าน Data Pipeline** — สำเร็จสมบูรณ์ ระบบสกัด Mel Spectrogram (128×130) จากไฟล์เสียง 16kHz แบบอัตโนมัติ รองรับ Dataset หลายรูปแบบ พร้อม Anti-Data-Leakage Split และ SpecAugment + Audio Augmentation

&emsp;**ด้าน Language Detection** — สร้าง SVM Pipeline ที่ระบุภาษาจาก Mel Spectrogram ได้ **98.54% Accuracy** บน 5 ภาษา โดยใช้ StandardScaler → PCA(80) → SVM(RBF) เป็น Lightweight Classifier ที่ไม่ต้องการ GPU

&emsp;**ด้านโมเดล Per-Language ResNet** — สร้างโมเดล EmotionResNet แยกสำหรับแต่ละภาษาสำเร็จ โมเดลมี ~2.3M Parameter ต่อภาษา และให้ผลดีสำหรับ 4/5 ภาษา

&emsp;**ด้านเป้าหมาย Test Accuracy** — **บรรลุเป้าหมาย** ค่าเฉลี่ย Test Accuracy = **79.16%** สูงกว่าเป้าหมาย 75% ที่กำหนดไว้

**ตารางที่ 5.0 สรุปผลการประเมินครบทุกภาษา**

| ภาษา | Dataset | Test Samples | Accuracy | สถานะ |
|---|---|---|---|---|
| Thai | TDED | 6,042 | **88.40%** | ✅ บรรลุ |
| Chinese | Chinese Emotion Corpus | 538 | **86.62%** | ✅ บรรลุ |
| Japan | JANON | 243 | **84.36%** | ✅ บรรลุ |
| English | RAVDESS + CREMA-D | 2,285 | **82.84%** | ✅ บรรลุ |
| Korean (ResNet) | hi_kia | 84 | 53.57% | ⚠️ ข้อมูลน้อย |
| Korean (SVM) | hi_kia | 84 | **72.62%** | ✅ ใช้ SVM แทน |
| Language Detector | All 5 languages | — | **98.54%** | ✅ บรรลุ |
| **เฉลี่ย (ResNet)** | | | **79.16%** | ✅ > 75% |

---

**ผลการ Training โดยละเอียด:**

 

 

**สรุปภาพรวม:**

&emsp;โครงงานนี้ประสบความสำเร็จในการพัฒนาระบบ Per-Language ResNet ที่แก้ปัญหา Prosody Mismatch ได้อย่างมีประสิทธิภาพ ด้วยค่าเฉลี่ย 79.16% ซึ่งสูงกว่าเป้าหมาย 75% ข้อค้นพบสำคัญที่มีคุณค่าทางวิชาการได้แก่ การพิสูจน์เชิงประจักษ์ว่า Per-Language Model ให้ผลดีกว่า Unified Model อย่างมีนัยสำคัญ และการค้นพบว่าสำหรับภาษาที่มีข้อมูลน้อย (<1,000 ตัวอย่าง) Classical ML (SVM) ให้ผลดีกว่า Deep Learning (ResNet) ซึ่งสอดคล้องกับหลักการ Bias-Variance Tradeoff

## 5.2 ปัญหาและอุปสรรคที่พบในการดำเนินงาน

&emsp;ตลอดระยะเวลาการพัฒนาโครงงาน พบปัญหาและอุปสรรคหลายประการที่ต้องวิเคราะห์และแก้ไขอย่างเป็นระบบ รายละเอียดของแต่ละปัญหามีดังนี้

---

**ปัญหาที่ 1 — Korean Data Scarcity: ข้อมูลภาษาเกาหลีน้อยเกินไปสำหรับ Deep Learning**

&emsp;ภาษาเกาหลีมีข้อมูลเพียง ~420 ตัวอย่าง (5 classes, ~84/class) ซึ่งไม่เพียงพอสำหรับ ResNet ที่มี ~2.3M Parameter ส่งผลให้ Overfit รุนแรง ResNet ทำได้เพียง 53.57% ในขณะที่ SVM ซึ่งมี Inductive Bias ที่ดีกว่าสำหรับข้อมูลน้อยได้ถึง 72.62%

&emsp;วิธีแก้ที่ใช้: สร้าง Audio Augmentation (Pitch Shift ±1/±2 semitones, Time Stretch 0.85/1.15, Gaussian Noise) เพื่อเพิ่มข้อมูล อย่างไรก็ตามพบว่า Augmentation ทำให้ Accuracy แย่ลง (28% < 38%) เนื่องจากไฟล์ hi_kia มีความยาวเพียง 0.4-1.0 วินาที Augmented versions จึงแทบเหมือนต้นฉบับและเพิ่มเฉพาะ Noise ข้อสรุปสุดท้ายคือใช้ Original 556 files โดยไม่ Augment และเลือก SVM เป็นโมเดลสำหรับ Korean

---

**ปัญหาที่ 2 — HuggingFace Repository 404: Download Script ล้มเหลว**

&emsp;`download_extra_data.py` เดิมใช้ Hardcoded Repository Names ที่ไม่มีอยู่จริงใน Hugging Face Hub ทำให้ดาวน์โหลดข้อมูลเพิ่มเติมไม่ได้เลย

&emsp;วิธีแก้: เปลี่ยนเป็นสองกลยุทธ์ คือ (1) ค้นหา Dataset จาก Hugging Face Hub API แบบ Dynamic และ (2) Audio Augmentation Fallback ที่รับประกันว่าจะสร้างข้อมูลได้เสมอ

---

**ปัญหาที่ 3 — Wav2Vec2Processor TypeError: ใช้ Class ผิด**

&emsp;`facebook/wav2vec2-large-xlsr-53` ไม่มี `vocab.json` ทำให้ `Wav2Vec2Processor.from_pretrained()` ล้มเหลวด้วย TypeError: expected str not NoneType

&emsp;วิธีแก้: ใช้ `Wav2Vec2FeatureExtractor` แทน `Wav2Vec2Processor` เพราะ FeatureExtractor ไม่ต้องการ vocab.json

---

**ปัญหาที่ 4 — torchaudio Version Mismatch**

&emsp;`torchaudio 2.11.0` ไม่ Compatible กับ PyTorch 2.6.0 ทำให้ Import Error ทันทีที่โหลด

&emsp;วิธีแก้: ติดตั้ง `torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124` ให้ตรงกับ CUDA version

---

**ปัญหาที่ 5 — Web Demo Runtime Errors**

&emsp;Web Demo มีสองปัญหาหลัก: (1) `RuntimeError: Missing key fc.*` เพราะ Architecture ใน `web_demo.py` ใช้ `self.fc` แต่โมเดลที่บันทึกไว้ใช้ `self.classifier` (2) `TypeError: Audio.__init__() show_download_button` เพราะ Gradio 6.x ลบ parameter นี้ออก

&emsp;วิธีแก้: (1) Align Architecture ให้ตรงกัน โดยใช้ `self.classifier` และ `256*4*4` (2) ลบ `show_download_button=True` ออกจาก `gr.Audio()`

---

**ปัญหาที่ 6 — Japan Dataset ไม่มีไฟล์เสียง RAW**

&emsp;`dataset/Japan/` มีเฉพาะ `.npy` Mel Spectrogram Features ไม่มีไฟล์ `.wav`/`.mp3` ดั้งเดิม ทำให้ไม่สามารถรัน Wav2Vec2 Feature Extraction หรือ Audio Augmentation สำหรับ Japanese ได้

&emsp;ผลลัพธ์: ข้าม Wav2Vec2 สำหรับ Japanese, โมเดล ResNet ที่ฝึกด้วย Original Features ได้ 84.36% ซึ่งดีพอ

---

**สรุปปัญหาทั้งหมด:**

| ปัญหา | ระดับผลกระทบ | วิธีแก้ที่ใช้ | ประสิทธิภาพ |
|---|---|---|---|
| Korean Data Scarcity | สูง (ResNet 53% < SVM 73%) | ใช้ SVM แทน ResNet สำหรับ Korean | ✅ แก้ได้ด้วย Model Selection |
| HuggingFace 404 | ปานกลาง | API Search + Augmentation Fallback | ✅ แก้ได้สมบูรณ์ |
| Wav2Vec2Processor TypeError | สูง | ใช้ Wav2Vec2FeatureExtractor แทน | ✅ แก้ได้สมบูรณ์ |
| torchaudio Mismatch | สูง | ติดตั้งเวอร์ชันที่ตรงกัน | ✅ แก้ได้สมบูรณ์ |
| Web Demo Architecture Mismatch | ปานกลาง | Align `self.classifier` + `256*4*4` | ✅ แก้ได้สมบูรณ์ |
| Japan ไม่มี Raw Audio | ต่ำ | ข้าม Wav2Vec2 สำหรับ Japanese | ✅ ไม่กระทบผลหลัก |

---

## 5.3 แนวทางการพัฒนาในอนาคต

&emsp;ระบบ Per-Language ResNet ที่พัฒนาแล้วเสร็จในโครงงานนี้ถือเป็นจุดเริ่มต้นที่มั่นคงสำหรับการพัฒนาต่อยอด แนวทางที่น่าสนใจในอนาคตมีดังนี้

### 5.3.1 เพิ่มข้อมูลภาษาเกาหลีและญี่ปุ่น

&emsp;ปัญหาหลักของ Korean คือ Dataset ขนาดเล็ก (~420 ตัวอย่าง) ทำให้ ResNet Overfit อย่างรุนแรง หากสามารถรวบรวมข้อมูลให้ได้ ≥ 1,000 ตัวอย่างต่ออารมณ์โดยอาจใช้ KEMDy หรือบันทึกเสียงเพิ่มเติม ResNet จะสามารถแซงหน้า SVM ได้ตามที่ No Free Lunch Theorem คาดการณ์ไว้ สำหรับ Japanese, JANON มีเพียง ~1,215 ตัวอย่าง การเพิ่มจาก JNV (Japanese Natural Voice) หรือ JTES จะช่วยเพิ่ม Accuracy ได้อีก

### 5.3.2 Ensemble Method สำหรับ Low-Resource Languages

&emsp;แทนที่จะเลือก ResNet หรือ SVM อย่างใดอย่างหนึ่งสำหรับ Korean Ensemble Method ที่รวม Prediction ของทั้งสองโมเดลด้วย Confidence-Weighted Voting อาจให้ผลดีกว่าโมเดลเดี่ยว:

```python
resnet_prob = resnet_model.predict_proba(mel_features)   # shape: (n_classes,)
svm_prob    = svm_model.predict_proba(mfcc_features)     # shape: (n_classes,)
ensemble    = 0.4 * resnet_prob + 0.6 * svm_prob
emotion     = classes[ensemble.argmax()]
```

### 5.3.3 ขยายระบบไปยังภาษาอื่น

&emsp;สถาปัตยกรรม Per-Language แบบ Modular รองรับการเพิ่มภาษาใหม่โดยไม่กระทบโมเดลที่มีอยู่ ภาษาที่น่าเพิ่มและมี Public Dataset พร้อมใช้ ได้แก่

| ภาษา | Dataset | ตัวอย่าง (โดยประมาณ) |
|---|---|---|
| Arabic | AESDD | ~500 |
| German | Emodb | ~535 |
| French | ADAS | ~900 |
| Hindi | IEMOCAP-Hindi | ~1,000+ |

### 5.3.4 Fine-tune Wav2Vec2 สำหรับ Emotion Recognition

&emsp;ขั้นตอน Wav2Vec2 ในโครงงานนี้ดึง Embedding แบบ Zero-shot โดยไม่ได้ Fine-tune เพิ่ม การ Fine-tune `facebook/wav2vec2-large-xlsr-53` บน Emotion Dataset โดยตรงจะสร้าง Representation ที่เหมาะกับ Emotion Feature มากกว่า และอาจช่วยภาษาที่มีข้อมูลน้อยเช่น Korean ได้อย่างมีนัยสำคัญ เนื่องจาก XLS-R ถูก Pre-train บนเสียง 128 ภาษาและมี Cross-lingual Transfer ที่ดี

### 5.3.5 ระบบทำงานแบบ Real-time Streaming

&emsp;ระบบปัจจุบันประมวลผลไฟล์เสียงที่บันทึกไว้ทั้งไฟล์ การพัฒนาให้รองรับ Streaming Input จาก Microphone แบบ Real-time โดยใช้ Sliding Window (เช่น 2 วินาที Step 0.5 วินาที) จะขยายขอบเขตการใช้งานไปยัง Call Center Monitoring, Live Presentation Feedback และ Interactive Entertainment

### 5.3.6 Deployment บน Mobile และ Edge Devices

&emsp;ResNet ที่ใช้มี ~2.35M Parameter ซึ่งหนักเกินไปสำหรับอุปกรณ์ Mobile การใช้ Model Compression เช่น Knowledge Distillation (ถ่ายความรู้ไปยัง Model ขนาดเล็กกว่า), Weight Pruning และ Quantization (ลด Precision เป็น INT8) จะลดขนาดโมเดลให้เหมาะสำหรับ Mobile App หรือ Embedded System

---

# บรรณานุกรม

1. Ekman, P. (1992). *An argument for basic emotions.* Cognition & Emotion, 6(3–4), 169–200.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.

3. Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English.* PLOS ONE, 13(5), e0196391.

4. Cao, H., Cooper, D. G., Kuchinsky, M. K., Ding, H., Bhanu, B., & Bhanu, B. (2014). *CREMA-D: Crowd-sourced emotional multimodal actors dataset.* IEEE Transactions on Affective Computing, 5(4), 377–390.

5. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A framework for self-supervised learning of speech representations.* Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 12449–12460.

6. Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). *SpecAugment: A simple data augmentation method for automatic speech recognition.* Interspeech 2019, 2613–2617.

7. Schuller, B., Vlasenko, B., Eyben, F., Wöllmer, M., Stuhlsatz, A., Wendemuth, A., & Rigoll, G. (2010). *Cross-corpus acoustic emotion recognition: Variances and strategies.* IEEE Transactions on Affective Computing, 1(2), 119–131.

8. McFee, B., Raffel, C., Liang, D., Ellis, D., McVicar, M., Battenberg, E., & Nieto, O. (2015). *librosa: Audio and music signal analysis in python.* Proceedings of the 14th Python in Science Conference (SciPy 2015), 18–25.

9. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). *PyTorch: An imperative style, high-performance deep learning library.* Advances in Neural Information Processing Systems (NeurIPS 2019), 32, 8026–8037.

10. Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization.* arXiv preprint arXiv:1412.6980.

11. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A simple way to prevent neural networks from overfitting.* The Journal of Machine Learning Research, 15(1), 1929–1958.

12. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). *Scikit-learn: Machine learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.

13. Davis, S. B., & Mermelstein, P. (1980). *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences.* IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357–366.

---

# ประวัติผู้เขียน

| รายการ | รายละเอียด |
|---|---|
| รหัสนักศึกษา | 66070131 |
| สาขาวิชา | วิทยาการคอมพิวเตอร์ |
| ภาคการศึกษา | 2 / 2567 |
| โครงงาน | ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษา ด้วยเทคนิคการเรียนรู้เชิงลึก |
| Hardware ที่ใช้ | NVIDIA GeForce RTX 3060 (12GB VRAM) |
| ภาษาโปรแกรม | Python 3.x |
| Framework หลัก | PyTorch 2.6 / torchaudio, Librosa, scikit-learn, Gradio |

---

*รายงานฉบับนี้จัดทำขึ้นเพื่อการศึกษา ภาคการศึกษา 2/2567*
