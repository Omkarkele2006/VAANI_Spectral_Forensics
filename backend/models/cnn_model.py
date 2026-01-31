import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Bidirectional, LSTM, BatchNormalization

def build_model(input_shape=(128, 128, 1)):
    """
    Constructs a Hybrid CRNN (Convolutional Recurrent Neural Network).
    
    Architecture:
    1. CNN Block: Extracts spatial features (artifacts in frequencies).
    2. Reshape Layer: Converts 2D feature maps into Time-Series data.
    3. Bi-LSTM Block: Captures temporal dependencies (how voice evolves).
    4. Dense Block: Final classification.
    
    Why this beats pure CNN:
    - Pure CNN loses time information (Flatten).
    - BiLSTM remembers the sequence of speech.
    """
    model = Sequential()

    # --- BLOCK 1: SPATIAL FEATURE EXTRACTION (CNN) ---
    # Input: (128, 128, 1) -> [Height (Freq), Width (Time), Channels]
    
    # Conv Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization()) # Stabilizes training
    model.add(MaxPooling2D(pool_size=(2, 2))) # Output: (64, 64, 32)

    # Conv Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) # Output: (32, 32, 64)

    # Conv Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) # Output: (16, 16, 128)
    
    # --- BLOCK 2: THE BRIDGE (Reshape) ---
    # Current Shape: (Batch, 16, 16, 128) -> (Batch, Freq, Time, Filters)
    # We want: (Batch, Time, Features) for the LSTM.
    # We will keep 'Time' (16) and merge 'Freq' and 'Filters' (16 * 128 = 2048).
    
    # Target Shape: (16, 2048)
    # Note: We assume the 2nd dim is Freq and 3rd is Time. 
    # If dimensions are swapped in preprocessing, we Permute first.
    # Here we treat the 16x16 grid as a sequence of 16 time steps, each with 16*128 features.
    model.add(Reshape((16, 16 * 128))) 
    
    # --- BLOCK 3: TEMPORAL LEARNING (BiLSTM) ---
    # Bidirectional allows the model to see context from both past and future audio frames.
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    
    # --- BLOCK 4: CLASSIFICATION ---
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4)) # Prevents overfitting
    
    model.add(Dense(1, activation='sigmoid')) # Output: 0 (Real) to 1 (Fake)

    # Compile
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Smoke Test
    try:
        model = build_model()
        model.summary()
        print("Hybrid CRNN (CNN + BiLSTM) Architecture built successfully.")
    except Exception as e:
        print(f"Error building model: {e}")