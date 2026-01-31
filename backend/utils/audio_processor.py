import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_spectrogram(file_path, output_image_path):
    """
    Converts an audio file into a Mel-Spectrogram image.
    
    Args:
        file_path (str): Path to the input audio file (.mp3, .wav).
        output_image_path (str): Where to save the generated spectrogram image.
    
    Returns:
        str: Path to the saved image (or None if failed).
    """
    try:
        # 1. Load Audio
        # sr=22050 is the industry standard sample rate for speech analysis.
        y, sr = librosa.load(file_path, sr=22050)

        # 2. Generate Mel-Spectrogram
        # n_mels=128: Height of the image (128 frequency bands).
        # fmax=8000: We focus on frequencies up to 8kHz (human speech range).
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

        # 3. Convert to Log-Scale (dB)
        # Raw spectral power is exponential; we convert to decibels (dB) so the AI can "see" details better.
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 4. Save as Image (No Axes, No Labels - Just Data)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close() # Close plot to free memory

        return output_image_path

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

if __name__ == "__main__":
    # Define paths relative to where we run the script
    # We assume we are running this from the 'backend' folder
    input_file = "test_audio.wav" 
    output_file = "test_spectrogram.png"

    print(f"Processing {input_file}...")
    
    # Call the function
    result = generate_spectrogram(input_file, output_file)
    
    if result:
        print(f"Success! Spectrogram saved at: {result}")
    else:
        print("Failed to process audio.")