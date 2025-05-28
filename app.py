import streamlit as st
import os
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

# Load OpenAI API key securely
try:
    openai_client = OpenAI(api_key=st.secrets["openai"]["api_key"])
except KeyError:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("🎙️ Accent Classifier AI Agent")

uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file:
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.read())
    st.info("✅ Audio file uploaded successfully.")

    # Extract audio features
    try:
        y, sr = librosa.load(temp_filename, sr=16000)
        mfccs_input = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        st.info("🎵 Audio features extracted.")
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
        st.stop()

    # Dummy classifier (replace with real model)
    dummy_features = [np.random.rand(13), np.random.rand(13), np.random.rand(13)]
    dummy_labels = ["American", "British", "Indian"]
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(dummy_features, dummy_labels)
    predicted_accent = knn.predict([mfccs_input])[0]

    st.success(f"🎤 Predicted Accent: **{predicted_accent}**")

    # Generate explanation using OpenAI Chat Completions
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are an AI assistant that explains technical concepts in simple terms."},
                {"role": "user", "content": f"Explain how accent detection works for a {predicted_accent} accent."}
            ],
            max_tokens=150
        )
        explanation = response.choices[0].message.content.strip()
        st.write(f"🧠 **Accent Detection Explanation:**\n{explanation}")
    except Exception as e:
        st.warning(f"🔑 OpenAI explanation unavailable: {e}")
