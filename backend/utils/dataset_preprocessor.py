import os
import shutil
from utils.audio_processor import generate_spectrogram

# Define paths
RAW_DATASET_PATH = "../data_store/dataset"
PROCESSED_DATASET_PATH = "../data_store/processed_dataset"

def process_dataset():
    """
    Iterates through the 'real' and 'fake' folders in the raw dataset,
    converts every audio file to a spectrogram, and saves it in the processed folder.
    """
    categories = ['real', 'fake']
    
    print(f"🚀 Starting Data Preprocessing...")
    print(f"   Source: {RAW_DATASET_PATH}")
    print(f"   Target: {PROCESSED_DATASET_PATH}")

    for category in categories:
        # 1. Setup paths
        source_folder = os.path.join(RAW_DATASET_PATH, category)
        target_folder = os.path.join(PROCESSED_DATASET_PATH, category)
        
        # 2. Create target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)
        
        # 3. Get list of files
        if not os.path.exists(source_folder):
            print(f"⚠️ Warning: Source folder not found: {source_folder}")
            continue
            
        files = [f for f in os.listdir(source_folder) if f.endswith(('.mp3', '.wav'))]
        print(f"\n📂 Processing '{category}' category: {len(files)} files found.")

        # 4. Process each file
        count = 0
        for filename in files:
            source_file = os.path.join(source_folder, filename)
            
            # Create a matching image filename (audio.wav -> audio_wav.png)
            image_filename = filename.replace('.', '_') + '.png'
            target_file = os.path.join(target_folder, image_filename)
            
            # Generate!
            result = generate_spectrogram(source_file, target_file)
            
            if result:
                count += 1
                if count % 10 == 0:
                    print(f"   - Processed {count}/{len(files)}...")
        
        print(f"✅ Finished '{category}': {count} images generated.")

if __name__ == "__main__":
    # This block allows us to run this script directly to prepare data
    process_dataset()