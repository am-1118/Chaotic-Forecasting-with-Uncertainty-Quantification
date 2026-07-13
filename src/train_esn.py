import os
import time
import json
import torch
import numpy as np
from sklearn.linear_model import Ridge
from deep_esn import DeepESN

def set_seed(seed: int = 42):
    """Locks all random number generators for strict reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_continuous_splits():
    """Loads the raw sequential data for the ESN."""
    print("Loading sequential data...")
    # ESNs process continuous time series, not sliding windows
    train_data = np.load('data/train.npy')
    val_data = np.load('data/val.npy')
    test_data = np.load('data/test.npy')
    
    # Format for PyTorch RNNs: (Time, Batch, Features)
    # Batch size is 1 because we have 1 continuous chaotic trajectory
    u_train = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
    u_val = torch.tensor(val_data, dtype=torch.float32).unsqueeze(1)
    u_test = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
    
    return u_train, u_val, u_test

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    u_train, u_val, u_test = load_continuous_splits()
    u_train, u_val, u_test = u_train.to(device), u_val.to(device), u_test.to(device)
    
    washout = 20 # Discard the first 20 steps to let the reservoir dynamics settle
    
    # 1. Initialize the Base Model
    model = DeepESN(input_dim=40, reservoir_size=100, leaky_rate=0.1, spectral_radius=0.9, input_scaling=0.1)
    model.to(device)
    
    # 2. Dynamic Architectural Design (The FFT Phase)
    start_time = time.time()
    model.design_architecture(u_train, max_layers=20, eta=0.01, washout=washout)
    
    # 3. State Harvesting (The Echo Phase)
    print("\nHarvesting global states...")
    with torch.no_grad():
        # Shape: (Time, Batch=1, N_L * N_R)
        states_train = model(u_train).squeeze(1).cpu().numpy()
        states_val = model(u_val).squeeze(1).cpu().numpy()
        states_test = model(u_test).squeeze(1).cpu().numpy()
        
    # 4. Prepare Supervised Targets
    # We use the state at time t to predict the target at time t+1
    # We also slice off the washout period
    X_train_readout = states_train[washout:-1]
    Y_train_target = u_train[washout+1:].squeeze(1).cpu().numpy()
    
    X_val_readout = states_val[washout:-1]
    Y_val_target = u_val[washout+1:].squeeze(1).cpu().numpy()
    
    X_test_readout = states_test[washout:-1]
    Y_test_target = u_test[washout+1:].squeeze(1).cpu().numpy()
    
    # 5. Train the Linear Readout
    print("\nTraining Linear Readout (Ridge Regression)...")
    readout = Ridge(alpha=1.0)
    readout.fit(X_train_readout, Y_train_target)
    
    train_time = time.time() - start_time
    
    # 6. Evaluation
    val_preds = readout.predict(X_val_readout)
    test_preds = readout.predict(X_test_readout)
    
    test_mse = np.mean((test_preds - Y_test_target) ** 2)
    test_mae = np.mean(np.abs(test_preds - Y_test_target))
    
    # Save predictions and targets for Conformal Prediction in the notebook
    os.makedirs('results', exist_ok=True)
    np.save('results/esn_val_preds.npy', val_preds)
    np.save('results/esn_val_true.npy', Y_val_target)
    np.save('results/esn_test_preds.npy', test_preds)
    np.save('results/esn_test_true.npy', Y_test_target)
    
    # Hardware metrics
    peak_vram_mb = 0.0
    if device.type == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
    total_params = sum(p.numel() for p in model.parameters())
    
    results = {
        "performance": {
            "test_mse": float(test_mse),
            "test_mae": float(test_mae)
        },
        "computational_metrics": {
            "train_time_seconds": round(train_time, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "total_reservoir_parameters": total_params,
            "selected_layers": len(model.layers)
        }
    }
    
    with open('results/esn_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n--- DeepESN Training Complete ---")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Training Time (Design + Harvest + Readout): {train_time:.2f} seconds")
    print(f"Peak VRAM: {peak_vram_mb:.1f} MB")

if __name__ == '__main__':
    main()