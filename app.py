import streamlit as st
import os
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import openai

# Load OpenAI API key from secrets or environment variable
try:
    openai.api_key = st.secrets["openai"]["openai_api_key"]
except KeyError:
    openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("🎙️ Accent Classifier AI Agent")

uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.read())
    st.info("✅ Audio file uploaded successfully.")

    # Extract features from the uploaded audio
    try:
        y, sr = librosa.load(temp_filename, sr=16000)
        mfccs_input = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        st.info("🎵 Audio features extracted.")
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
        st.stop()

    # 🔥 Dummy training data (replace with a real model in production)
    dummy_features = [
        np.random.rand(13),  # Random MFCC-like vector
        np.random.rand(13),
        np.random.rand(13)
    ]
    dummy_labels = ["American", "British", "Indian"]
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(dummy_features, dummy_labels)

    predicted_accent = knn.predict([mfccs_input])[0]
    st.success(f"🎤 Predicted Accent: **{predicted_accent}**")

    # Optional OpenAI explanation
    if openai.api_key:
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
    else:
        st.warning("🔑 OpenAI API key not found. Skipping explanation.")
