import streamlit as st
import librosa
import numpy as np
import joblib
import os
import tempfile
import pandas as pd
from visualization import plot_audio_waveform, plot_feature_breakdown, plot_genre_probabilities, plot_spectrogram, create_audio_dashboard



@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        model = joblib.load('models/model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'models/model.joblib' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def extract_features(file_path):
    """Extract audio features from uploaded file"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=30)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Calculate statistics
        mfcc_mean = np.mean(mfccs, axis=1).flatten()
        mfcc_std = np.std(mfccs, axis=1).flatten()
        chroma_mean = np.mean(chroma, axis=1).flatten()
        chroma_std = np.std(chroma, axis=1).flatten()

        # Handle tempo
        if isinstance(tempo, np.ndarray):
            tempo_val = tempo.item() if tempo.size == 1 else tempo[0]
        else:
            tempo_val = float(tempo)

        # Combine all features
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            chroma_mean,
            chroma_std,
            [tempo_val]
        ])

        return features, y, sr
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None, None



def main():
    st.set_page_config(
        page_title="Music Genre Classifier",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Music Genre Classification")
    st.markdown("Upload an audio file to predict its music genre!")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Define genres 
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'au', 'flac', 'm4a'],
        help="Supported formats: WAV, MP3, AU, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract features
            with st.spinner("Extracting audio features..."):
                features, y, sr = extract_features(tmp_file_path)
            
            if features is not None:
                # Define feature names (used throughout the analysis)
                feature_names = (
                    [f'mfcc_mean_{i}' for i in range(13)] +
                    [f'mfcc_std_{i}' for i in range(13)] +
                    [f'chroma_mean_{i}' for i in range(12)] +
                    [f'chroma_std_{i}' for i in range(12)] +
                    ['tempo']
                )
                
                # Make prediction
                features_reshaped = features.reshape(1, -1)
                prediction = model.predict(features_reshaped)[0]
                probabilities = model.predict_proba(features_reshaped)[0]
                
                # Create two columns for results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🎯 Prediction Results")
                    
                    # Display prediction
                    st.success(f"**Predicted Genre: {prediction.upper()}**")
                    
                    # Display confidence
                    confidence = max(probabilities) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Display all probabilities
                    st.subheader("📊 Genre Probabilities")
                    
                    # Create probability bar chart using visualization module
                    fig_prob = plot_genre_probabilities(genres, probabilities, prediction)
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                with col2:
                    st.subheader("🎵 Audio Analysis")
                    
                    # Audio player
                    st.audio(uploaded_file, format='audio/wav')
                    
                    # Display audio info
                    duration = len(y) / sr
                    st.info(f"""
                    **Audio Information:**
                    - Duration: {duration:.2f} seconds
                    - Sample Rate: {sr} Hz
                    - Tempo: {features[-1]:.1f} BPM
                    """)
                    
                    # Plot waveform using visualization module
                    if y is not None and sr is not None:
                        fig_wave = plot_audio_waveform(y, sr)
                        st.plotly_chart(fig_wave, use_container_width=True)
                
                # Advanced visualizations
                st.subheader("🔍 Advanced Audio Analysis")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["📈 Features", "🎵 Spectrogram", "🎯 Dashboard"])
                
                with tab1:
                    fig_features = plot_feature_breakdown(features, feature_names)
                    st.plotly_chart(fig_features, use_container_width=True)
                
                with tab2:
                    fig_spec = plot_spectrogram(y, sr)
                    st.plotly_chart(fig_spec, use_container_width=True)
                
                with tab3:
                    # Create comprehensive dashboard
                    dashboard_figs = create_audio_dashboard(
                        y, sr, features, feature_names, genres, probabilities, prediction
                    )
                    
                    # Display radar chart
                    st.plotly_chart(dashboard_figs['radar'], use_container_width=True)
                
                # Feature table
                with st.expander("View Detailed Features"):
                    feature_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': features
                    })
                    st.dataframe(feature_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses a Random Forest classifier trained on audio features to predict music genres.
        
        **Features used:**
        - MFCC (Mel-frequency cepstral coefficients)
        - Chroma features
        - Tempo
        
        **Supported Genres:**
        """)
        for genre in genres:
            st.write(f"• {genre.title()}")
        
        st.header("📝 Instructions")
        st.write("""
        1. Upload an audio file (WAV, MP3, AU, FLAC, M4A)
        2. Wait for feature extraction
        3. View the predicted genre and confidence
        4. Explore audio analysis and features
        """)
        
        st.header("⚠️ Notes")
        st.write("""
        - Audio files are processed for 30 seconds maximum
        - Model accuracy depends on audio quality
        - Best results with clear, single-genre music
        """)


if __name__ == "__main__":
    main()