import matplotlib
# CRITICAL: Must be the very first line to prevent server crashes
matplotlib.use('Agg') 

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_spectrogram(file_path, output_image_path):
    """
    Converts audio to a Mel-Spectrogram (Headless Mode).
    """
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path, sr=22050)

        # 2. Generate Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 3. Save as Image (No GUI window)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()
        
        # Save and Close immediately
        plt.savefig(output_image_path)
        plt.close() 

        return output_image_path

    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None

if __name__ == "__main__":
    print("Audio Processor Utility (Headless) is ready.")