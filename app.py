import streamlit as st
import os
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import openai

# Load OpenAI key from Streamlit secrets (configured on cloud)
openai.api_key = st.secrets["openai_api_key"]

st.title("🎙️ Accent Classifier AI Agent")

# File upload
uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", 
type=["wav", "mp3"])

if uploaded_file:
    # Save uploaded file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Feature extraction
    y, sr = librosa.load("temp_audio.wav", sr=16000)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)

    # Simple placeholder model
    # Replace with your own trained model later!
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit([mfccs], ["American"])  # Dummy fit
    predicted_accent = knn.predict([mfccs])[0]

    st.success(f"Predicted Accent: {predicted_accent}")
    
    # Optionally, ask OpenAI for additional info (replace with your API 
usage)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Explain accent detection for {predicted_accent}.",
        max_tokens=50
    )
    st.write(response.choices[0].text)

