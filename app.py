import streamlit as st
import os
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import openai

# Load OpenAI API key from Streamlit secrets or environment variable
try:
    openai.api_key = st.secrets["openai"]["openai_api_key"]
except KeyError:
    openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("🎙️ Accent Classifier AI Agent")

# File upload widget
uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded audio file
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.read())

    st.info("✅ Audio file uploaded successfully.")

    # Extract features from the audio
    try:
        y, sr = librosa.load(temp_filename, sr=16000)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        st.info("🎵 Audio features extracted.")
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
        st.stop()

    # Placeholder KNN model (replace with a real trained model)
    knn = KNeighborsClassifier(n_neighbors=3)
    # Dummy fitting with one sample (replace with real data during training)
    knn.fit([mfccs], ["American"])
    predicted_accent = knn.predict([mfccs])[0]

    st.success(f"🎤 Predicted Accent: **{predicted_accent}**")

    # Optional: Use OpenAI for an explanation
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
