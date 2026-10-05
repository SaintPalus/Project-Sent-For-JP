# 🎙️ Multilingual Speech Emotion Recognition

A system that recognizes emotion from speech in **5 languages**: Thai, Chinese, Japanese, Korean and English.

**Developer:** Palus Kaewaram · KMITL, B.Sc. Artificial Intelligence Technology

**Senior Project** · To be presented at **ICCAS 2026** (October 26, 2026)

---

## Results

| Metric | Value |
|---|---|
| Overall deployed accuracy | **82.97%** |
| Languages supported | 5 (TH, ZH, JA, KO, EN) |

---

## Architecture

```
Audio input
    │
    ▼
┌──────────────────────┐
│  Preprocessing       │  Librosa → Mel Spectrogram
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Language Detector   │  SVM
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  EmotionResNet       │  one PyTorch model per language
└──────────┬───────────┘
           ▼
     Predicted emotion
```

Different languages express emotion through different prosody (pitch, rhythm, stress). A single shared model confuses these patterns, so the system first detects the language, then routes the audio to a ResNet trained for that language.

---

## What I Built

- **Data labeling tool** for annotating emotion in audio clips
- **Audio preprocessing pipeline** with Librosa (Mel Spectrograms, SpecAugment for augmentation)
- **Model experiments** comparing CNN + BiLSTM against per-language ResNet architectures
- **SVM language detector** that routes each clip to the right emotion model
- **Expert validation** of labels and results, plus full project documentation

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Deep learning | PyTorch |
| Audio processing | Librosa |
| Models | EmotionResNet (per language), CNN + BiLSTM |
| Language detection | SVM (scikit-learn) |

---

## Project Structure

```
SeniorP1/
├── models/                                  # Trained PyTorch models (EmotionResNet per language)
├── results/                                 # Evaluation results
├── แยกภาษา/                                  # Per-language ResNet code (Phase 2)
├── Full_Report_Senior_Project.md            # Full project report
└── SpeechEmotionRecognition_Presentation.pptx  # Presentation slides (17 slides)
```

---

## Documentation

- 📄 [Full project report](SeniorP1/Full_Report_Senior_Project.md)
- 📊 [Presentation slides](SeniorP1/SpeechEmotionRecognition_Presentation.pptx)
