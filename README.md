# 🔊 VOXSENSE — Animal Voice Emotion Detection System

An intelligent deep learning system that detects **animal type** and **emotional state** from audio recordings using CNN-based spectrogram classification.

---

## 📁 Project Structure

```
VOXSENSE/
├── app/
│   └── app.py                  # Streamlit web application
├── data/
│   ├── raw/                    # Raw audio files (.wav)
│   │   ├── dog/angry/
│   │   ├── dog/happy/
│   │   ├── cat/angry/
│   │   └── ...
│   ├── processed/              # Processed numpy arrays
│   └── spectrograms/           # Saved spectrogram images
├── models/
│   └── model.py                # Saved trained model (.h5)
├── notebooks/
│   └── exploration.ipynb       # EDA and experimentation
├── reports/
│   ├── presentation.pptx       # Project presentation
│   └── report.py               # Report generator
├── src/
│   ├── evaluation/
│   │   └── evaluate.py         # Model evaluation metrics
│   ├── features/
│   │   └── extract_features.py # Audio feature extraction
│   ├── models/
│   │   ├── model_architecture.py  # CNN model definition
│   │   └── train_model.py         # Training pipeline
│   ├── preprocessing/
│   │   └── preprocess.py       # Audio preprocessing
│   └── utils/
│       └── helpers.py          # Utility functions
├── main.py                     # Entry point
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourname/voxsense.git
cd voxsense
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data
```
data/raw/
├── dog/
│   ├── angry/  → dog_angry_001.wav, ...
│   ├── happy/  → dog_happy_001.wav, ...
│   └── normal/ → dog_normal_001.wav, ...
├── cat/
│   ├── angry/
│   └── ...
```

### 3. Run Training Pipeline
```bash
python main.py --mode train
```

### 4. Evaluate Model
```bash
python main.py --mode evaluate
```

### 5. Launch Web App
```bash
streamlit run app/app.py
```

---

## 🧠 Model Architecture

| Layer | Type | Details |
|-------|------|---------|
| Input | — | 128×128×1 Mel Spectrogram |
| Conv1 | Conv2D + BN + MaxPool | 32 filters, 3×3 |
| Conv2 | Conv2D + BN + MaxPool | 64 filters, 3×3 |
| Conv3 | Conv2D + BN + MaxPool | 128 filters, 3×3 |
| Conv4 | Conv2D + BN | 256 filters, 3×3 |
| FC1 | Dense + Dropout(0.4) | 512 units |
| FC2 | Dense + Dropout(0.3) | 256 units |
| Output | Dense + Softmax | num_classes |

---

## 📊 Features Extracted

- **MFCC** (Mel Frequency Cepstral Coefficients) — 40 coefficients
- **Chroma** — 12 pitch class features
- **Mel Spectrogram** — 128 mel bands
- **Zero Crossing Rate** — temporal feature
- **RMS Energy** — amplitude envelope
- **Spectral Contrast** — 7 bands
- **Tonnetz** — tonal centroid features

---

## 🐾 Supported Animals & Emotions

| Animal | Emotions |
|--------|----------|
| 🐕 Dog | Angry, Happy, Normal, Fear, Sad |
| 🐈 Cat | Angry, Happy, Normal, Fear, Sad |
| 🐦 Bird | Angry, Happy, Normal, Fear, Sad |
| 🦁 Lion | Angry, Normal, Fear |
| 🐺 Wolf | Angry, Happy, Normal |

---

## 📈 Expected Results

- **Animal Classification Accuracy**: ~92–96%
- **Emotion Detection Accuracy**: ~85–91%
- **Combined (Animal + Emotion)**: ~83–88%

---

## 👥 Team Roles

| Role | Tasks |
|------|-------|
| Data Engineer | Data collection, cleaning, augmentation |
| DSP Engineer | Audio preprocessing, feature extraction |
| ML Engineer | CNN architecture, training, tuning |
| Evaluator | Metrics, confusion matrix, reports |
| UI Developer | Streamlit app, visualization |

---

## 📜 License
MIT License © 2025 VOXSENSE Team