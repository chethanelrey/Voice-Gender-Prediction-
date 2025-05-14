# Voice gender prediction using logistic regression

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Step 1: Function to record voice
def record_voice(duration=5, fs=44100, output_file='user_recorded_voice.wav'):
    print("Recording your voice... Please speak into the microphone.")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    wav.write(output_file, fs, audio_data)  # Save the recorded audio as WAV
    print(f"Audio recorded and saved as {output_file}")
    return output_file

# Step 2: Feature extraction from the recorded voice
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCC features
    mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean across time for each MFCC
    return mfcc_mean

# Step 3: Load your pre-trained model and scaler (dummy model for now)
log_reg = LogisticRegression()
scaler = StandardScaler()

# Dummy training (replace this with loading your real trained model and scaler)
log_reg.fit(np.random.rand(10, 13), np.random.choice(['male', 'female'], 10))
scaler.fit(np.random.rand(10, 13))

# Step 4: Record voice
duration = 5  # Duration in seconds
file_path = record_voice(duration)

# Step 5: Extract features
features = extract_features(file_path)

# Step 6: Scale the features
features_scaled = scaler.transform([features])

# Step 7: Make prediction
prediction_log_reg = log_reg.predict(features_scaled)

# Step 8: Output the result
print("\nPrediction from Logistic Regression:")
if prediction_log_reg[0] == 'male':
    print("The gender is classified as: Male")
else:
    print("The gender is classified as: Female")