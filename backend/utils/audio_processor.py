import os
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg') # Fix for running on server without a monitor
import matplotlib.pyplot as plt
import numpy as np

def generate_spectrogram(audio_path, image_path):
    """
    Generates a Mel-spectrogram from audio, matching Kaggle training settings exactly.
    """
    try:
        # 1. Load Audio (Limit to 3 seconds to match training)
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        
        # 2. Generate Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 3. Plot without Axes or Borders (CRITICAL FIX)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, fmax=8000)
        plt.axis('off')             # Hide numbers
        plt.tight_layout(pad=0)     # Remove padding
        
        # 4. Save
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return image_path
        
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None