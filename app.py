import streamlit as st
import os
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import openai

# Load OpenAI key securely from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

st.title("🎙️ Accent Classifier AI Agent")

# File upload widget
uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded audio file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.info("✅ File uploaded and saved successfully.")

    # Feature extraction using librosa
    try:
        y, sr = librosa.load("temp_audio.wav", sr=16000)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        st.info("🎵 Audio features extracted successfully.")
    except Exception as e:
        st.error(f"❌ Error processing audio file: {e}")
        st.stop()

    # Simple placeholder model (to be replaced with a real trained model)
    knn = KNeighborsClassifier(n_neighbors=3)
    # Dummy training with one sample (for demo purposes)
    knn.fit([mfccs], ["American"])  # Replace with real training data!
    predicted_accent = knn.predict([mfccs])[0]

    st.success(f"🎤 Predicted Accent: **{predicted_accent}**")

    # Optional: Use OpenAI to explain accent detection
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Explain how accent detection works for a {predicted_accent} accent.",
            max_tokens=100
        )
        explanation = response.choices[0].text.strip()
        st.write(f"🧠 **Accent Detection Explanation:**\n{explanation}")
    except Exception as e:
        st.warning(f"OpenAI explanation unavailable: {e}")
