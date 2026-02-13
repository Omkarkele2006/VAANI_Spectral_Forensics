# 🎙️ VAANI: Deepfake Audio Detection System

![Status](https://img.shields.io/badge/Status-Complete-success)
![Version](https://img.shields.io/badge/Version-1.0-blue)
![Tech Stack](https://img.shields.io/badge/Tech-React%20%7C%20Flask%20%7C%20TensorFlow-black)

**VAANI** is a full-stack forensic Deepfake Audio Detection System. It utilizes a custom **Hybrid CRNN (Convolutional Recurrent Neural Network)** to analyze audio files, generate visual Mel-spectrograms, and accurately classify whether a human voice is **Real** or **Synthetic** (AI-generated).

---

## Key Features

* **Real-Time Forensic Dashboard:** A React.js frontend that allows users to upload audio files and instantly view the AI's prediction, confidence score, and generated spectrogram evidence.
* **Hybrid CRNN Engine:** Combines Convolutional Neural Networks (CNNs) to extract spatial features from audio spectrograms, and Bidirectional LSTMs to analyze temporal sequences.
* **Lossy Compression Robustness:** Specifically trained to see through audio compression. The model successfully identifies deepfakes even when disguised by `.mp3` compression, downsampling (`16k`), or volume normalization.
* **Batch Testing Suite:** Includes an automated Python script (`batch_test.py`) to process hundreds of audio files simultaneously and generate tabular forensic reports (CSV).

---

## System Architecture

The core intelligence of VAANI relies on transforming audio into the visual domain:

1. **Preprocessing (The Eyes):** Audio files (`.wav`, `.mp3`, etc.) are processed using `librosa`. The audio is converted into a borderless **Mel-spectrogram**, capturing the frequency artifacts often left behind by AI voice synthesizers (e.g., high-frequency cutoffs).
2. **CNN Layers:** Extracts visual shapes, textures, and anomalies from the spectrogram.
3. **Bi-LSTM Layers:** Analyzes how these frequencies change over time.
4. **Classification:** Outputs a confidence score (0.0 to 1.0) classifying the audio as Synthetic or Real.

---

## Dataset & Training Performance

The model was trained on the **Kaggle Fake-or-Real (FoR) Dataset**, utilizing a cloud GPU (Tesla T4) workflow. 

To ensure real-world robustness, the training dataset (~10,000 samples) included:
* **Original pristine audio**
* **Volume-normalized audio**
* **Re-recorded audio** (simulating phone calls/speakers)

**Performance Metrics:**
* **Validation Accuracy:** ~99.60%
* **Testing:** Achieved near-perfect performance on controlled evaluation sets(including extreme MP3 compression and 16kHz downsampling).

---

## Tech Stack

* **Frontend:** React.js, HTML/CSS
* **Backend:** Python, Flask, SQLite (for logging analysis history)
* **Machine Learning:** TensorFlow / Keras, Librosa, Matplotlib, NumPy, Pandas
* **Model Architecture:** Custom CRNN (CNN + Bidirectional LSTM)

---

## Installation & Setup

Follow these steps to run the VAANI system locally on your machine.

### Prerequisites
* Node.js & npm installed
* Python 3.8+ installed
* Anaconda (Recommended for managing ML environments)
