import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


def plot_audio_waveform(y, sr, title="Audio Waveform"):
    """
    Plot audio waveform using Plotly
    
    Args:
        y: Audio time series
        sr: Sample rate
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    time = np.linspace(0, len(y) / sr, len(y))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=y,
        mode='lines',
        name='Waveform',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        height=300,
        template='plotly_white',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def plot_feature_breakdown(features, feature_names, title="Extracted Audio Features"):
    """
    Plot feature values in separate subplots for different feature types
    
    Args:
        features: Array of feature values
        feature_names: List of feature names
        title: Main title for the plot
    
    Returns:
        Plotly figure object
    """
    # Create subplots for different feature types
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('MFCC Mean', 'MFCC Std', 'Chroma Mean', 'Chroma Std'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # MFCC Mean (first 13 features)
    fig.add_trace(
        go.Bar(
            x=list(range(13)), 
            y=features[:13], 
            name='MFCC Mean',
            marker_color=colors[0],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # MFCC Std (next 13 features)
    fig.add_trace(
        go.Bar(
            x=list(range(13)), 
            y=features[13:26], 
            name='MFCC Std',
            marker_color=colors[1],
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Chroma Mean (next 12 features)
    fig.add_trace(
        go.Bar(
            x=list(range(12)), 
            y=features[26:38], 
            name='Chroma Mean',
            marker_color=colors[2],
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Chroma Std (next 12 features)
    fig.add_trace(
        go.Bar(
            x=list(range(12)), 
            y=features[38:50], 
            name='Chroma Std',
            marker_color=colors[3],
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=500, 
        title_text=title,
        template='plotly_white'
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Coefficient Index", row=1, col=1)
    fig.update_xaxes(title_text="Coefficient Index", row=1, col=2)
    fig.update_xaxes(title_text="Chroma Bin", row=2, col=1)
    fig.update_xaxes(title_text="Chroma Bin", row=2, col=2)
    
    return fig


def plot_genre_probabilities(genres, probabilities, predicted_genre=None):
    """
    Plot genre classification probabilities as horizontal bar chart
    
    Args:
        genres: List of genre names
        probabilities: Array of probabilities for each genre
        predicted_genre: Name of predicted genre (for highlighting)
    
    Returns:
        Plotly figure object
    """
    prob_df = pd.DataFrame({
        'Genre': genres,
        'Probability': probabilities * 100
    }).sort_values('Probability', ascending=True)  # Sort for better visualization
    
    # Create color array - highlight predicted genre
    colors = ['#ff7f0e' if genre == predicted_genre else '#1f77b4' 
              for genre in prob_df['Genre']]
    
    fig = px.bar(
        prob_df,
        x='Probability',
        y='Genre',
        orientation='h',
        title='Genre Classification Probabilities (%)',
        color='Probability',
        color_continuous_scale='viridis',
        text='Probability'
    )
    
    # Update text format
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        xaxis_title='Probability (%)',
        yaxis_title='Genre',
        coloraxis_showscale=False
    )
    
    return fig


def plot_feature_importance_radar(feature_values, feature_categories):
    """
    Create a radar chart for feature importance visualization
    
    Args:
        feature_values: Array of normalized feature values
        feature_categories: Dict mapping category names to feature indices
    
    Returns:
        Plotly figure object
    """
    # Calculate average values for each category
    categories = []
    values = []
    
    for category, indices in feature_categories.items():
        if isinstance(indices, tuple):
            start, end = indices
            avg_value = np.mean(np.abs(feature_values[start:end]))
        else:
            avg_value = abs(feature_values[indices])
        categories.append(category)
        values.append(avg_value)
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Feature Strength',
        line_color='rgb(32, 201, 151)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )),
        showlegend=False,
        title="Feature Category Strength",
        height=400
    )
    
    return fig


def plot_spectrogram(y, sr, title="Spectrogram"):
    """
    Plot spectrogram using Plotly
    
    Args:
        y: Audio time series
        sr: Sample rate
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    import librosa
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    # Create time and frequency axes
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)
    freqs = librosa.fft_frequencies(sr=sr)
    
    fig = go.Figure(data=go.Heatmap(
        z=D,
        x=times,
        y=freqs[:D.shape[0]],
        colorscale='viridis',
        colorbar=dict(title="dB")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        height=400,
        template='plotly_white'
    )
    
    return fig


# Function for training/evaluation visualizations (from your original code)
def visualize_results(y_true, y_pred, X, y_labels, feature_names, feature_importances, label_order):
    """
    Create comprehensive visualization for model evaluation results
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        X: Feature matrix
        y_labels: All labels
        feature_names: List of feature names
        feature_importances: Feature importance scores
        label_order: Order of labels for consistent plotting
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_order, yticklabels=label_order)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance plot
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=True).tail(20)  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(label_order)))
    
    for i, genre in enumerate(label_order):
        mask = y_labels == genre
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], label=genre, alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Visualization of Audio Features by Genre')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualizations saved: confusion_matrix.png, feature_importance.png, pca_visualization.png")


def create_audio_dashboard(y, sr, features, feature_names, genres, probabilities, predicted_genre):
    """
    Create a comprehensive dashboard with multiple visualizations
    
    Args:
        y: Audio time series
        sr: Sample rate
        features: Extracted features
        feature_names: Feature names
        genres: List of genres
        probabilities: Prediction probabilities
        predicted_genre: Predicted genre
    
    Returns:
        Dictionary of Plotly figures
    """
    figures = {}
    
    # Waveform
    figures['waveform'] = plot_audio_waveform(y, sr)
    
    # Feature breakdown
    figures['features'] = plot_feature_breakdown(features, feature_names)
    
    # Genre probabilities
    figures['probabilities'] = plot_genre_probabilities(genres, probabilities, predicted_genre)
    
    # Spectrogram
    figures['spectrogram'] = plot_spectrogram(y, sr)
    
    # Feature radar chart
    feature_categories = {
        'MFCC Mean': (0, 13),
        'MFCC Std': (13, 26),
        'Chroma Mean': (26, 38),
        'Chroma Std': (38, 50),
        'Tempo': 50
    }
    figures['radar'] = plot_feature_importance_radar(features, feature_categories)
    
    return figures