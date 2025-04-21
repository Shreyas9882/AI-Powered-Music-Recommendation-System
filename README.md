# AI-Powered-Music-Recommendation-System
This project is an interactive web application that predicts the genre of uploaded music files and recommends popular songs from the same genre. Built with Streamlit for the user interface and powered by a TensorFlow deep learning model, it processes audio data, predicts its genre, and displays top recommendations using a clustered music dataset.


### **Objective**
The goal of this project is to create a web application that classifies music into one of several predefined genres based on audio input and then provides personalized song recommendations based on the predicted genre. This application bridges deep learning and data-driven recommendation systems within an interactive user interface.


### **Functionality Overview**

1. **Audio Upload & Playback**
   - Users can upload an `.mp3` audio file through a Streamlit-based interface.
   - The app allows users to listen to the uploaded audio directly on the platform using an embedded audio player.

2. **Audio Preprocessing**
   - The uploaded audio is processed using the **Librosa** library.
   - It is divided into overlapping chunks of 4 seconds (with a 2-second overlap) to ensure sufficient temporal coverage.
   - Each chunk is converted into a **mel spectrogram**, which is a visual representation of the audio's frequency content over time.
   - These spectrograms are resized to a fixed dimension (150x150) and formatted for input into the prediction model.

3. **Genre Classification**
   - The app uses a **pre-trained TensorFlow Keras model** (loaded from `Trained_model.h5`) to classify the genre.
   - The model predicts the genre for each chunk and selects the most frequently predicted genre across all chunks to determine the final output.
   - The supported genres include: **blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock**.

4. **Recommendation Engine**
   - Based on the predicted genre, the app filters a pre-clustered dataset (`clustered_df.csv`) to retrieve relevant songs.
   - The results are sorted by **popularity** and the **top 10 songs** are displayed with detailed metadata.

5. **User Interface**
   - The frontend is built using **Streamlit** and styled with custom CSS for a modern, dark-themed aesthetic.
   - Key features of the UI include:
     - Clean layout with centralized audio upload and playback
     - Button-triggered genre prediction and recommendation
     - Animated feedback (e.g., loading spinner, balloon effect)
     - Visually structured recommendation cards with attributes like artist, year, tempo, energy, valence, and danceability


### **Technologies Used**

- **Streamlit**: For building the interactive web interface
- **TensorFlow/Keras**: For deep learning model prediction
- **Librosa**: For audio analysis and mel spectrogram generation
- **NumPy and Pandas**: For numerical operations and data handling
- **HTML/CSS**: For customizing the visual design within Streamlit


### **Key Design Decisions**
- **Chunk-Based Audio Processing**: Ensures that predictions are not biased by a single part of the song and gives a more holistic view of the genre.
- **Genre Voting Mechanism**: Selecting the most frequent prediction from all chunks increases reliability.
- **Cluster-Based Recommendation**: Leveraging clustering allows recommendations to align better with sonic characteristics rather than just genre labels.
