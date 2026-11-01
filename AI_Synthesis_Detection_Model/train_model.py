import tensorflow as tf
import numpy as np
import os
import librosa
import glob

print("TensorFlow version:", tf.__version__)

# Hyperparameters
SAMPLE_LENGTH = 16000 # 1 second of audio at 16kHz
BATCH_SIZE = 8
EPOCHS = 20

def load_audio_files(directory, label):
    X = []
    Y = []
    filepaths = glob.glob(os.path.join(directory, "*.wav"))
    for file in filepaths:
        try:
            # Load and resample to 16kHz
            y, sr = librosa.load(file, sr=16000)
            
            # We want exact 1-second chunks (16000 samples)
            # If the audio is longer, we can extract multiple 1-second chunks
            for i in range(0, len(y) - SAMPLE_LENGTH, SAMPLE_LENGTH):
                chunk = y[i:i + SAMPLE_LENGTH]
                X.append(chunk)
                Y.append(label)
                
            # Handle the last chunk or if the audio is shorter than 1 second
            if len(y) < SAMPLE_LENGTH:
                # Pad with zeros
                padded_chunk = np.pad(y, (0, SAMPLE_LENGTH - len(y)), 'constant')
                X.append(padded_chunk)
                Y.append(label)
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    return X, Y

# Load the dataset
print("Loading Real Audio (Class 0)...")
x_real, y_real = load_audio_files("dataset/real", 0.0)

print("Loading Fake/Synthetic Audio (Class 1)...")
x_synth, y_synth = load_audio_files("dataset/fake", 1.0)

if not x_real or not x_synth:
    print("ERROR: Dataset is empty! Please run download_sample_data.py first.")
    exit(1)

# Combine and shape
X = np.concatenate([x_real, x_synth], axis=0)
Y = np.concatenate([y_real, y_synth], axis=0)

# Normalize all audio chunks by max amplitude to remove volume dependency
for i in range(len(X)):
    max_amp = np.max(np.abs(X[i]))
    if max_amp > 0.001:
        X[i] = X[i] / max_amp

# Add random Gaussian noise augmentation to simulate microphone static
noise = np.random.normal(0, 0.015, X.shape)
X_noisy = X + noise

# Combine clean and noisy data for robust training
X = np.concatenate([X, X_noisy], axis=0)
Y = np.concatenate([Y, Y], axis=0)

X = np.expand_dims(X, axis=-1) # Shape: (samples, 16000, 1)
Y = np.expand_dims(Y, axis=-1) # Shape: (samples, 1)

# Shuffle
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

print(f"Total samples: {len(X)}")
print(f"X shape: {X.shape}")

# Define a deeper 1D CNN model robust to raw audio waveforms
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(SAMPLE_LENGTH, 1)),
    
    # Block 1
    tf.keras.layers.Conv1D(32, kernel_size=64, strides=8, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Dropout(0.2),
    
    # Block 2
    tf.keras.layers.Conv1D(64, kernel_size=32, strides=4, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Dropout(0.2),
    
    # Block 3
    tf.keras.layers.Conv1D(128, kernel_size=16, strides=2, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid') # 0 = Real, 1 = Synthetic
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
print(model.summary())

# Train the model
print("Training model on real and fake audio data...")
model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

# Convert to TFLite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Note: No optimizations to avoid Android FULLY_CONNECTED v12 issues with quantization
tflite_model = converter.convert()

# Save the model
model_path = "voice_spoof_detector.tflite"
with open(model_path, "wb") as f:
    f.write(tflite_model)
    
print(f"Successfully saved TFLite model to {model_path}!")
print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
