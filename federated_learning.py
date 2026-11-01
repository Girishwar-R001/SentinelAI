import tensorflow as tf
import numpy as np
import os
from train_rawnet2 import build_rawnet2_model, load_asvspoof_protocol, evaluate_metrics

print("Starting Federated Learning Simulation...")

NUM_CLIENTS = 3
ROUNDS = 3
LOCAL_EPOCHS = 3

def federated_averaging(global_model, client_models):
    """Averages the weights of client models to update the global model"""
    global_weights = global_model.get_weights()
    new_weights = []
    
    for i in range(len(global_weights)):
        # Calculate the average of the i-th layer's weights across all clients
        layer_weights = np.array([client.get_weights()[i] for client in client_models])
        new_weights.append(np.mean(layer_weights, axis=0))
        
    global_model.set_weights(new_weights)

if __name__ == "__main__":
    protocol_file = "dataset/asvspoof/ASVspoof2019_LA_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    audio_folder = "dataset/asvspoof/ASVspoof2019_LA_train/wav"
    
    X, Y = load_asvspoof_protocol(protocol_file, audio_folder)
    
    if len(X) == 0:
        print("Dataset empty. Exiting.")
        exit(1)
        
    # Normalize
    for i in range(len(X)):
        max_amp = np.max(np.abs(X[i]))
        if max_amp > 0.001:
            X[i] = X[i] / max_amp
            
    # Simulate data partitioning for clients
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    
    # Reserve some data for global validation
    val_split = int(0.2 * len(X))
    X_val, Y_val = X[:val_split], Y[:val_split]
    X_train, Y_train = X[val_split:], Y[val_split:]
    
    client_X = np.array_split(X_train, NUM_CLIENTS)
    client_Y = np.array_split(Y_train, NUM_CLIENTS)
    
    print("Building Global Mobile-RawNet2 Model...")
    global_model = build_rawnet2_model()
    
    for round_num in range(1, ROUNDS + 1):
        print(f"\n==============================================")
        print(f" FEDERATED LEARNING ROUND {round_num}/{ROUNDS}")
        print(f"==============================================")
        
        client_models = []
        
        for c in range(NUM_CLIENTS):
            print(f"\n--- Client {c+1} Local Training ---")
            # In a real system, the client builds the model and sets global weights
            local_model = build_rawnet2_model()
            local_model.set_weights(global_model.get_weights())
            
            # Train locally
            local_model.fit(client_X[c], client_Y[c], epochs=LOCAL_EPOCHS, batch_size=8, verbose=1)
            client_models.append(local_model)
            
        print("\n[Server] Aggregating client updates (FedAvg)...")
        federated_averaging(global_model, client_models)
        
        print("\n[Server] Evaluating Global Model on Validation Set...")
        evaluate_metrics(global_model, X_val, Y_val)
        
    print("\nFederated Learning Simulation Complete.")
    
    tflite_path = "voice_spoof_detector.tflite"
    print(f"Exporting final Aggregated Model to {tflite_path}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(global_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Export successful.")
