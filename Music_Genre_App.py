import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import pandas as pd
from tensorflow.image import resize

# Load your clustered dataset
df = pd.read_csv("clustered_df.csv")

# Label list
labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load your model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("Trained_model.h5")

# Preprocessing audio
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2

    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        if len(chunk) < chunk_samples:
            break
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

# Model prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Recommendation logic
def recommend_by_genre(genre, df, top_n=10):
    genre = genre.lower()
    df_filtered = df[df["Cluster"] == 4]  # Cluster 4 is hiphop
    genre_filtered = df_filtered[df_filtered["year"] >= 2010]
    genre_filtered = genre_filtered.sort_values(by="popularity", ascending=False).head(top_n)
    return genre_filtered[["name", "artists", "year", "popularity", "tempo", "energy", "valence", "danceability"]]

# Streamlit UI
st.set_page_config(page_title="🎵 Music Genre Classifier & Recommender", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #1e1e2f;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #facc15;
}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Music Genre Prediction & Song Recommender")
st.markdown("Upload your MP3 and get a genre prediction along with top song recommendations!")

test_mp3 = st.file_uploader("🎵 Upload an audio file (.mp3)", type=["mp3"])

if test_mp3 is not None:
    filepath = 'Test_Music/' + test_mp3.name
    with open(filepath, "wb") as f:
        f.write(test_mp3.getbuffer())

    st.audio(test_mp3)

    if st.button("🔮 Predict Genre & Recommend Songs"):
        with st.spinner("Analyzing... Please wait!"):
            X_test = load_and_preprocess_data(filepath)
            result_index = model_prediction(X_test)
            predicted_genre = labels[result_index]
            st.balloons()

            st.markdown(f"<h2 style='text-align:center;' class='prediction'>🎶 Predicted Genre: <span style='color:#00e0ff'>{predicted_genre.capitalize()}</span></h2>", unsafe_allow_html=True)

            # Recommend songs based on genre
            recommendations = recommend_by_genre(predicted_genre, df)

            if not recommendations.empty:
                st.markdown("### 🔥 **Top 10 Recommendations Based on Genre (Sorted by Popularity)**")

                for _, row in recommendations.iterrows():
                    st.markdown(f"""
<div style='
    background-color: #1f2937;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
'>
    <h4 style='color:#facc15; margin-bottom:10px;'>🎵 {row['name']}</h4>
    <p style='margin: 5px 0;'><strong>👤 Artist(s):</strong> {row['artists']}</p>
    <p style='margin: 5px 0;'><strong>📅 Year:</strong> {row['year']} &nbsp;&nbsp;|&nbsp;&nbsp; <strong>🌟 Popularity:</strong> {row['popularity']}</p>
    <p style='margin: 5px 0;'>
        🎚️ <strong>Tempo:</strong> {row['tempo']} BPM &nbsp;&nbsp;|&nbsp;&nbsp;
        ⚡ <strong>Energy:</strong> {row['energy']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
        💖 <strong>Valence:</strong> {row['valence']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
        💃 <strong>Danceability:</strong> {row['danceability']:.2f}
    </p>
</div>
""", unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for the predicted genre.")
