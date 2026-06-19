import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from kan_model import TimeSeriesKAN
import random

def set_seed(seed: int = 42):
    """Locks all random number generators for exact reproducibility."""
    # 1. Set Python built-in random seed
    random.seed(seed)
    
    # 2. Set NumPy seed
    np.random.seed(seed)
    
    # 3. Set PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU, though you are on a single 3050
        
    # 4. Force cuDNN to behave deterministically
    # Note: This might slightly reduce training speed, but guarantees exact same math
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to {seed}. Deterministic mode enabled.")

def load_data():
    """Loads the windowed datasets."""
    print("Loading data...")
    X_train = torch.tensor(np.load('data/X_train.npy'))
    y_train = torch.tensor(np.load('data/y_train.npy'))
    X_val = torch.tensor(np.load('data/X_val.npy'))
    y_val = torch.tensor(np.load('data/y_val.npy'))
    X_test = torch.tensor(np.load('data/X_test.npy'))
    y_test = torch.tensor(np.load('data/y_test.npy'))
    return X_train, y_train, X_val, y_val, X_test, y_test

def count_parameters(model):
    """Calculates the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # --- Reproducibility ---
    set_seed(42)
    
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    batch_size = 128
    epochs = 300
    patience = 20
    l1_lambda = 1e-5  # Sparsity penalty for KAN edges
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # --- Data Prep ---
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Initialization ---
    model = TimeSeriesKAN(seq_len=10, num_vars=40, hidden_size=128, grid_size=5).to(device)
    num_params = count_parameters(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # --- Tracking Variables ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    
    # Reset GPU memory stats before training
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        
    start_time = time.time()

    # --- Training Loop ---
    print(f"Starting training... (Total Parameters: {num_params:,})")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            # Base MSE Loss
            mse_loss = criterion(predictions, batch_y)
            
            # L1 Regularization (crucial for KAN sparsity)
            l1_loss = sum(p.abs().sum() for p in model.parameters())
            
            loss = mse_loss + (l1_lambda * l1_loss)
            loss.backward()
            optimizer.step()
            
            train_loss += mse_loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                val_loss += criterion(predictions, batch_y).item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        # --- Early Stopping & Model Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'models/kan_best.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # --- Compute Computational Metrics ---
    total_time = time.time() - start_time
    peak_vram_mb = 0.0
    if device.type == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
    model_size_mb = os.path.getsize('models/kan_best.pt') / (1024 ** 2)

    # --- Testing & Evaluation ---
    print("\nEvaluating best model on Test Set...")
    model.load_state_dict(torch.load('models/kan_best.pt', weights_only=True))
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    
    mse_per_var = np.mean((y_true - y_pred) ** 2, axis=0)
    mae_per_var = np.mean(np.abs(y_true - y_pred), axis=0)
    
    avg_mse = float(np.mean(mse_per_var))
    avg_mae = float(np.mean(mae_per_var))

    # --- Save Results ---
    results = {
        "performance": {
            "avg_mse": avg_mse,
            "avg_mae": avg_mae,
            "mse_per_var": mse_per_var.tolist(),
            "mae_per_var": mae_per_var.tolist()
        },
        "computational_metrics": {
            "train_time_seconds": round(total_time, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "model_size_mb": round(model_size_mb, 2),
            "total_parameters": num_params,
            "epochs_to_converge": best_epoch,
            "early_stop_epoch": epoch + 1
        }
    }

    with open('results/kan_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Training Complete ---")
    print(f"Test MSE: {avg_mse:.4f}")
    print(f"Test MAE: {avg_mae:.4f}")
    print(f"Training Time: {total_time:.1f} seconds")
    print(f"Peak VRAM: {peak_vram_mb:.1f} MB")
    print("Metrics saved to results/kan_metrics.json")

if __name__ == '__main__':
    main()