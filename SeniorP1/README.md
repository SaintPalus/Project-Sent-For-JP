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
│  Per-Language ResNet │  one emotion model per language
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
| Audio processing | Librosa |
| Models | ResNet, CNN + BiLSTM |
| Language detection | SVM (scikit-learn) |

---

## Project Structure

```
SeniorP1/        # Senior project source code
```

<!-- TODO: add the main files inside SeniorP1/ and how to run them -->

---

## How to Run

```bash
git clone https://github.com/SaintPalus/Project-Sent-For-JP.git
cd Project-Sent-For-JP
pip install -r requirements.txt
```

<!-- TODO: add the command to train or run inference -->
