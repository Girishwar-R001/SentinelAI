import tensorflow as tf
import numpy as np
import os
import glob
import soundfile as sf
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score

print("TensorFlow version:", tf.__version__)

SAMPLE_RATE = 16000
SAMPLE_LENGTH = 16000  # 1 second of audio
BATCH_SIZE = 8
EPOCHS = 2  # Reduced for hackathon speed

# ==============================================================================
# SincConv1D Layer (RawNet2 Front-end)
# ==============================================================================
class SincConv1D(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, sample_rate=16000, min_low_hz=50, min_band_hz=50, **kwargs):
        super(SincConv1D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
    def build(self, input_shape):
        low_hz_init = np.linspace(self.min_low_hz, self.sample_rate / 2 - self.min_band_hz, self.num_filters)
        band_hz_init = np.ones(self.num_filters) * self.min_band_hz
        
        self.low_hz = self.add_weight(
            name="low_hz", shape=(self.num_filters,),
            initializer=tf.keras.initializers.Constant(low_hz_init / self.sample_rate), trainable=True
        )
        self.band_hz = self.add_weight(
            name="band_hz", shape=(self.num_filters,),
            initializer=tf.keras.initializers.Constant(band_hz_init / self.sample_rate), trainable=True
        )
        
        n = np.arange(self.kernel_size)
        t = n - (self.kernel_size - 1) / 2
        self.t = tf.constant(t, dtype=tf.float32)
        window = 0.54 - 0.46 * np.cos(2 * np.pi * n / self.kernel_size)
        self.window = tf.constant(window, dtype=tf.float32)
        super(SincConv1D, self).build(input_shape)
        
    def call(self, x):
        f_min = self.min_low_hz / self.sample_rate
        f_max = (self.sample_rate / 2 - self.min_band_hz) / self.sample_rate
        
        low = f_min + tf.abs(self.low_hz)
        band = self.min_band_hz / self.sample_rate + tf.abs(self.band_hz)
        high = tf.clip_by_value(low + band, f_min, f_max)
        
        low_t = 2 * np.pi * tf.expand_dims(low, -1) * tf.expand_dims(self.t, 0)
        high_t = 2 * np.pi * tf.expand_dims(high, -1) * tf.expand_dims(self.t, 0)
        
        mid = (self.kernel_size - 1) // 2
        sinc_low = tf.math.divide_no_nan(tf.sin(low_t), low_t)
        sinc_high = tf.math.divide_no_nan(tf.sin(high_t), high_t)
        
        mask = np.zeros((1, self.kernel_size), dtype=np.float32)
        mask[0, mid] = 1.0
        mask_tensor = tf.constant(mask, dtype=tf.float32)
        
        sinc_low = sinc_low * (1.0 - mask_tensor) + tf.expand_dims(2 * low, -1) * mask_tensor
        sinc_high = sinc_high * (1.0 - mask_tensor) + tf.expand_dims(2 * high, -1) * mask_tensor
        
        filters = (sinc_high - sinc_low) * tf.expand_dims(self.window, 0)
        filters = tf.transpose(tf.expand_dims(filters, 1), perm=[2, 1, 0])
        
        return tf.nn.conv1d(x, filters, stride=1, padding='SAME')
        
    def get_config(self):
        config = super(SincConv1D, self).get_config()
        config.update({"num_filters": self.num_filters, "kernel_size": self.kernel_size})
        return config

# ==============================================================================
# Squeeze-and-Excitation Residual Block
# ==============================================================================
def se_res_block(inputs, filters, kernel_size=3):
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.3)(x)
    
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Squeeze-and-Excitation
    se = tf.keras.layers.GlobalAveragePooling1D()(x)
    se = tf.keras.layers.Dense(filters // 8, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    x = tf.keras.layers.Multiply()([x, se])
    
    # Residual Connection
    if inputs.shape[-1] != filters:
        inputs = tf.keras.layers.Conv1D(filters, 1, padding='same')(inputs)
        
    x = tf.keras.layers.Add()([inputs, x])
    x = tf.keras.layers.MaxPooling1D(pool_size=3)(x)
    return tf.keras.layers.LeakyReLU(negative_slope=0.3)(x)

# ==============================================================================
# Model Builder
# ==============================================================================
def build_rawnet2_model():
    """Lightweight RawNet2 optimized for edge devices"""
    inputs = tf.keras.Input(shape=(SAMPLE_LENGTH, 1))
    
    # 1. Feature Extraction (SincConv)
    x = SincConv1D(num_filters=64, kernel_size=129)(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.3)(x)
    
    # 2. Residual SE Blocks
    x = se_res_block(x, filters=32)
    x = se_res_block(x, filters=32)
    x = se_res_block(x, filters=64)
    x = se_res_block(x, filters=64)
    
    # 3. GRU Head
    x = tf.keras.layers.GRU(256, return_sequences=False)(x)
    
    # 4. Fully Connected
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.3)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ==============================================================================
# Data Loader
# ==============================================================================
def load_asvspoof_protocol(protocol_path, audio_dir):
    """
    Parses ASVspoof format protocol file:
    Format: LA_0079 LA_T_1000137 - - bonafide
    """
    X = []
    Y = []
    if not os.path.exists(protocol_path):
        print(f"ERROR: Protocol file not found: {protocol_path}")
        return np.array(X), np.array(Y)
        
    print(f"Parsing protocol: {protocol_path}")
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            file_id = parts[1]
            label_str = parts[4]
            label = 0.0 if label_str == 'bonafide' else 1.0
            
            # Check for wav or flac
            file_path = os.path.join(audio_dir, f"{file_id}.wav")
            if not os.path.exists(file_path):
                file_path = os.path.join(audio_dir, f"{file_id}.flac")
                
            if os.path.exists(file_path):
                try:
                    y, _ = sf.read(file_path)
                    
                    # Exact 1 second chunks
                    for i in range(0, len(y) - SAMPLE_LENGTH + 1, SAMPLE_LENGTH):
                        X.append(y[i:i + SAMPLE_LENGTH])
                        Y.append(label)
                        
                    if len(y) > 0 and len(y) < SAMPLE_LENGTH:
                        padded_chunk = np.pad(y, (0, SAMPLE_LENGTH - len(y)), 'constant')
                        X.append(padded_chunk)
                        Y.append(label)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    
    X = np.array(X)
    Y = np.array(Y)
    if len(X) > 0:
        X = np.expand_dims(X, axis=-1)
        Y = np.expand_dims(Y, axis=-1)
    return X, Y

# ==============================================================================
# Metrics Evaluation
# ==============================================================================
def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def evaluate_metrics(model, X_test, Y_test):
    if len(X_test) == 0:
        return
        
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)
    
    acc = accuracy_score(Y_test, y_pred_classes)
    prec = precision_score(Y_test, y_pred_classes, zero_division=0)
    rec = recall_score(Y_test, y_pred_classes, zero_division=0)
    f1 = f1_score(Y_test, y_pred_classes, zero_division=0)
    
    try:
        eer = compute_eer(Y_test, y_pred_probs)
    except:
        eer = float('nan')
        
    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"EER:       {eer*100:.2f}%")
    print("--------------------------------\n")
    return acc, eer

# ==============================================================================
# Main execution
# ==============================================================================
if __name__ == "__main__":
    protocol_file = "dataset/asvspoof/ASVspoof2019_LA_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    audio_folder = "dataset/asvspoof/ASVspoof2019_LA_train/wav"
    
    print("Loading ASVspoof training dataset...")
    X_train, Y_train = load_asvspoof_protocol(protocol_file, audio_folder)
    
    if len(X_train) == 0:
        print("ERROR: Dataset empty. Run download_asvspoof_data.py first.")
        exit(1)
        
    print(f"Dataset Shape: X={X_train.shape}, Y={Y_train.shape}")
    
    # Shuffle
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    
    # Normalize
    for i in range(len(X_train)):
        max_amp = np.max(np.abs(X_train[i]))
        if max_amp > 0.001:
            X_train[i] = X_train[i] / max_amp
            
    print("Building Mobile-RawNet2...")
    model = build_rawnet2_model()
    model.summary()
    
    print(f"Training on {len(X_train)} samples for {EPOCHS} epochs...")
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
    
    # Evaluate on the training set (for prototyping verification)
    evaluate_metrics(model, X_train, Y_train)
    
    # Export to TFLite with Float16 Quantization
    print("Converting to TFLite (Float16)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    
    tflite_path = "rawnet2_float16.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Mobile-RawNet2 exported to {tflite_path} ({len(tflite_model)/1024/1024:.2f} MB)")
