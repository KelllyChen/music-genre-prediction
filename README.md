# music-genre-prediction
This mini-project is for learning how to work with audio data, including feature extraction, model training, and visualization.

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/KelllyChen/music-genre-prediction.git
cd music-genre-prediction
```
### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the App(Demo)
```bash
streamlit run app.py
```

## Data
Audio files were downloaded from [GTZAN Genre Collection](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection)

## Features Extracting

For each `.au` file, the following audio features are extracted:
- MFCCs (mean & std of 13 coefficients)
- Chroma features (mean & std of 12 bins)
- Tempo

## Training Process

- Audio features are extracted using `librosa`.
- A Random Forest classifier is trained using **5-fold stratified cross-validation**.
- The best model (based on validation accuracy) is selected and evaluated on a held-out test set.
- Model performance is reported via classification metrics.
- The best model is saved to `models/model.joblib`.

## Demo
- Upload and visualize audio waveform and spectrogram
- Breakdown of MFCC and chroma features
- Radar chart of feature category strengths
- Predict music genre using a pre-trained ML model
- View genre classification probabilities


