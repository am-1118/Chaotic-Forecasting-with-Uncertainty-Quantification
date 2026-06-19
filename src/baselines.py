import numpy as np
from sklearn.linear_model import Ridge
from typing import Dict, Any

def persistence_forecast(X: np.ndarray) -> np.ndarray:
    """
    Baseline 1: Persistence model. 
    Predicts the next time step will be identical to the last observed state.
    """
    return X[:, -1, :]

def train_linear_forecast(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Ridge:
    """
    Baseline 2: Linear Regression (Ridge).
    Flattens the input window and maps it to the target variables.
    """
    samples = X_train.shape[0]
    # Flatten window: (samples, window_size * N)
    X_train_flat = X_train.reshape(samples, -1)
    
    model = Ridge(alpha=alpha)
    model.fit(X_train_flat, y_train)
    return model

def predict_linear_forecast(model: Ridge, X_test: np.ndarray) -> np.ndarray:
    """
    Generates predictions using the trained Ridge model.
    """
    samples = X_test.shape[0]
    X_test_flat = X_test.reshape(samples, -1)
    return model.predict(X_test_flat)

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes MSE and MAE per variable, and returns the averages across all variables.
    """
    mse_per_var = np.mean((y_true - y_pred) ** 2, axis=0)
    mae_per_var = np.mean(np.abs(y_true - y_pred), axis=0)
    
    avg_mse = float(np.mean(mse_per_var))
    avg_mae = float(np.mean(mae_per_var))
    
    return {
        'avg_mse': avg_mse,
        'avg_mae': avg_mae,
        'mse_per_var': mse_per_var.tolist(),
        'mae_per_var': mae_per_var.tolist()
    }