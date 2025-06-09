import librosa
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mfcc_mean = np.mean(mfccs, axis=1).flatten()
    mfcc_std = np.std(mfccs, axis=1).flatten()
    chroma_mean = np.mean(chroma, axis=1).flatten()
    chroma_std = np.std(chroma, axis=1).flatten()

    if isinstance(tempo, np.ndarray):
        tempo_val = tempo.item() if tempo.size == 1 else tempo[0]
    else:
        tempo_val = float(tempo)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        chroma_mean,
        chroma_std,
        [tempo_val]
    ])

    return features


def main():
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    data = []
    labels = []

    for genre in genres:
        folder = f'genres/{genre}'
        for filename in os.listdir(folder):
            if filename.endswith('.au'):
                path = os.path.join(folder, filename)
                try:
                    features = extract_features(path)
                    data.append(features)
                    labels.append(genre)
                    print(f"Processed: {genre}/{filename}")
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue

    print(f"Total samples processed: {len(data)}")

    X = np.array(data)
    y = np.array(labels)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    accuracies = []
    models = []

    for train_index, val_index in skf.split(X_main, y_main):
        X_train, X_val = X_main[train_index], X_main[val_index]
        y_train, y_val = y_main[train_index], y_main[val_index]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_val_pred)
        print(f"Fold {fold} Validation Accuracy: {acc:.4f}")
        accuracies.append(acc)
        models.append(clf)
        fold += 1

    print(f"\nAverage Cross-Validation Accuracy: {np.mean(accuracies):.4f}")

    # Choose the best model 
    best_model_idx = np.argmax(accuracies)
    best_model = models[best_model_idx]

    y_test_pred = best_model.predict(X_test)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Save the best model
    joblib.dump(best_model, 'models/model.joblib')


    

    


if __name__ == "__main__":
    main()
