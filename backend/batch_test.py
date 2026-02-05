import os
import numpy as np
import pandas as pd # For nice tables
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils.audio_processor import generate_spectrogram

# --- CONFIGURATION ---
TEST_FOLDER = "test_zone"
MODEL_PATH = "vaani_model.h5"
IMG_SIZE = (128, 128)

def prepare_image(img_path):
    """Loads and preprocesses image exactly like the app."""
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

def run_batch_test():
    print(f"🚀 Starting Batch Forensic Analysis on '{TEST_FOLDER}'...")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found!")
        return
    
    print("🧠 Loading VAANI Super-Model...")
    model = load_model(MODEL_PATH)
    
    # 2. Find Audio Files
    audio_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')
    files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(audio_extensions)]
    
    if not files:
        print(f"❌ No audio files found in {TEST_FOLDER}. Please add some!")
        return

    results = []
    print(f"📂 Found {len(files)} files. Processing...\n")

    # 3. Process Loop
    for filename in files:
        audio_path = os.path.join(TEST_FOLDER, filename)
        temp_img_path = os.path.join(TEST_FOLDER, filename + "_spec.png")
        
        try:
            # A. Generate Spectrogram
            spec_path = generate_spectrogram(audio_path, temp_img_path)
            
            if not spec_path:
                results.append({"File": filename, "Status": "Error (Spec Generation)"})
                continue
                
            # B. Predict
            processed_img = prepare_image(spec_path)
            prediction_value = model.predict(processed_img, verbose=0)[0][0]
            
            # C. Logic (Same as App.py)
            if prediction_value > 0.5:
                label = "Real"
                confidence = prediction_value * 100
            else:
                label = "Synthetic"
                confidence = (1 - prediction_value) * 100
            
            # Record Result
            results.append({
                "File": filename,
                "Prediction": label,
                "Confidence": f"{confidence:.2f}%",
                "Raw_Score": f"{prediction_value:.4f}"
            })
            
            # Cleanup Image
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
            print(f"  ✅ Checked {filename}: {label} ({confidence:.1f}%)")
            
        except Exception as e:
            print(f"  ❌ Failed {filename}: {e}")
            results.append({"File": filename, "Status": f"Error: {str(e)}"})

    # 4. Final Report
    print("\n" + "="*50)
    print("📊 FINAL FORENSIC REPORT")
    print("="*50)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save to CSV for Excel
    df.to_csv("batch_report.csv", index=False)
    print("\n📄 Report saved to 'batch_report.csv'")

if __name__ == "__main__":
    run_batch_test()