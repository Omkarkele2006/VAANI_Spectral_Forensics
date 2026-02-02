import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models.cnn_model import build_model

# --- CONFIGURATION ---
DATASET_PATH = "../data_store/processed_dataset"
REPORTS_PATH = "../data_store/reports"
MODEL_SAVE_PATH = "vaani_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 10

# Create reports folder if it doesn't exist
os.makedirs(REPORTS_PATH, exist_ok=True)

def plot_training_history(history):
    """Generates and saves the Accuracy & Loss Graphs."""
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', [])
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    if val_acc:
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training Accuracy')

    # Plot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training Loss')
    
    save_path = os.path.join(REPORTS_PATH, "training_graphs.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Graphs saved to: {save_path}")

def plot_confusion_matrix(model, val_generator):
    """Generates and saves the Confusion Matrix Heatmap."""
    print("🔍 Generating Confusion Matrix...")
    
    # 1. Get Predictions
    # Reset generator to start to ensure order matches labels
    val_generator.reset()
    predictions = model.predict(val_generator, steps=len(val_generator))
    y_pred = (predictions > 0.5).astype(int).ravel()
    
    # 2. Get True Labels
    y_true = val_generator.classes
    
    # 3. Compute Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 4. Draw Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    save_path = os.path.join(REPORTS_PATH, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"mw Heatmap saved to: {save_path}")
    
    # 5. Print Classification Report (Precision/Recall)
    report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'])
    print("\n--- Detailed Forensic Report ---")
    print(report)
    
    # Save text report
    with open(os.path.join(REPORTS_PATH, "metrics_report.txt"), "w") as f:
        f.write(report)

def train():
    print("🚀 Initializing Training Pipeline (Phase 7: Visualization Mode)...")

    # 1. Data Generators
    # Validation Split: 20% for testing the graphs
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    print(f"📂 Loading data from: {DATASET_PATH}")

    train_generator = datagen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', color_mode='grayscale', subset='training', shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', color_mode='grayscale', subset='validation', shuffle=False # Shuffle False is vital for Confusion Matrix!
    )

    # 2. Build Model
    model = build_model(input_shape=(128, 128, 1))
    
    # 3. Callbacks
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

    # 4. Train
    print(f"🔥 Starting Training for {EPOCHS} Epochs...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    print("✅ Training Complete.")
    
    # 5. Generate Reports
    plot_training_history(history)
    plot_confusion_matrix(model, val_generator)

if __name__ == "__main__":
    train()