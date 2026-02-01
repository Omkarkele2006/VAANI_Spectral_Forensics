import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from models.cnn_model import build_model

# --- CONFIGURATION ---
DATASET_PATH = "../data_store/processed_dataset"
MODEL_SAVE_PATH = "vaani_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 2   # Reduced to 2 so it works with only 6 images
EPOCHS = 10      

def train():
    print("🚀 Initializing Training Pipeline (Pilot Mode)...")

    # 1. Data Generator
    # We REMOVED validation_split because the dataset is too small (6 images).
    datagen = ImageDataGenerator(rescale=1./255)

    print(f"📂 Loading data from: {DATASET_PATH}")

    # Load ALL images for training
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )

    # 2. Build Model
    print("🧠 Building Hybrid CRNN Architecture...")
    model = build_model(input_shape=(128, 128, 1))
    
    # 3. Callbacks
    # Changed monitor to 'loss' (training loss) since we have no validation data yet
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH, 
        monitor='loss', 
        save_best_only=True, 
        mode='min', 
        verbose=1
    )
    
    # 4. START TRAINING
    print(f"🔥 Starting Training for {EPOCHS} Epochs...")
    
    try:
        model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
            epochs=EPOCHS,
            callbacks=[checkpoint]
        )

        print("✅ Training Complete.")
        print(f"💾 Model saved to: {os.path.abspath(MODEL_SAVE_PATH)}")
        
    except Exception as e:
        print(f"❌ Training Error: {e}")

if __name__ == "__main__":
    train()