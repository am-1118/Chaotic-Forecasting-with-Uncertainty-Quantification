import os
import json
import torch
import numpy as np
from kan_model import TimeSeriesKAN

def load_data():
    """Loads the validation and test sets."""
    print("Loading datasets...")
    X_val = torch.tensor(np.load('data/X_val.npy'))
    y_val = np.load('data/y_val.npy')
    X_test = torch.tensor(np.load('data/X_test.npy'))
    y_test = np.load('data/y_test.npy')
    return X_val, y_val, X_test, y_test

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Conformal Prediction on: {device}")
    
    # 1. Load Data & Model
    X_val, y_val_true, X_test, y_test_true = load_data()
    
    model = TimeSeriesKAN(seq_len=10, num_vars=40, hidden_size=128, grid_size=5).to(device)
    model.load_state_dict(torch.load('models/kan_best.pt', map_location=device, weights_only=True))
    model.eval()
    
    # 2. Calibration (Validation Set)
    print("Calibrating on Validation Set...")
    with torch.no_grad():
        y_val_pred = model(X_val.to(device)).cpu().numpy()
        
    # Calculate absolute residuals
    residuals = np.abs(y_val_true - y_val_pred)
    
    # Pool all residuals across samples and the 40 variables
    # This leverages the translational invariance of the Lorenz 96 system
    pooled_residuals = residuals.flatten()
    n = len(pooled_residuals)
    
    # Calculate the finite-sample corrected quantile for 90% coverage (alpha = 0.1)
    alpha = 0.1
    correction_factor = np.ceil((n + 1) * (1 - alpha)) / n
    # Cap at 1.0 to prevent standard quantile function from throwing an error
    q_level = min(correction_factor, 1.0) 
    
    q_hat = float(np.quantile(pooled_residuals, q_level))
    print(f"Calculated pooled q_hat (90% coverage): {q_hat:.4f}")
    
    # 3. Evaluation (Test Set)
    print("Applying intervals to Test Set...")
    with torch.no_grad():
        y_test_pred = model(X_test.to(device)).cpu().numpy()
        
    # Construct intervals
    lower_bounds = y_test_pred - q_hat
    upper_bounds = y_test_pred + q_hat
    
    # Calculate empirical coverage (fraction of true values inside the bounds)
    # Boolean array of shape (test_samples, 40)
    is_covered = (y_test_true >= lower_bounds) & (y_test_true <= upper_bounds)
    empirical_coverage = float(np.mean(is_covered))
    
    # Calculate average interval width
    avg_width = float(2 * q_hat)
    
    print(f"\n--- Conformal Prediction Results ---")
    print(f"Target Coverage:    {(1 - alpha) * 100:.1f}%")
    print(f"Empirical Coverage: {empirical_coverage * 100:.2f}%")
    print(f"Average Width:      {avg_width:.4f}")
    
    # 4. Save Results
    results = {
        "target_coverage": 1 - alpha,
        "empirical_coverage": empirical_coverage,
        "q_hat": q_hat,
        "average_width": avg_width
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/kan_conformal.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Results saved to results/kan_conformal.json")

if __name__ == '__main__':
    main()