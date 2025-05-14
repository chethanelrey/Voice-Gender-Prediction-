# 🎤 Voice Gender Prediction using Logistic Regression

This project allows you to record a short voice sample through your microphone and uses a logistic regression model to predict the gender (male/female) based on audio features extracted using MFCC (Mel Frequency Cepstral Coefficients).

## 🧠 How It Works

1. **Record**: The program records a 5-second voice sample from the user.
2. **Extract**: MFCC features are extracted from the audio.
3. **Predict**: A logistic regression model (dummy-trained here) predicts the gender based on the features.

> ⚠️ This is a demo version with a dummy model trained on random data. For accurate predictions, train the model on a real voice dataset.

---

## 📋 Features

- 🎙️ Voice recording via microphone
- 🔍 MFCC feature extraction using `librosa`
- 📈 Gender prediction using logistic regression
- 🧪 Scalable to real datasets for training and improvement

---

## 🧰 Requirements

Install the required packages using pip:

```bash
pip install numpy scipy sounddevice librosa scikit-learn


