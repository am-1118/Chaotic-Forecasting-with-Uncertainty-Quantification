import numpy as np

def create_windows(data: np.ndarray, window_size: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates overlapping windows for time-series forecasting.
    
    Args:
        data: 2D array of shape (time_steps, features).
        window_size: Number of past time steps to use as input.
        
    Returns:
        X: 3D array of shape (samples, window_size, features)
        y: 2D array of shape (samples, features) representing the next step.
    """
    X, y = [], []
    # Total samples will be len(data) - window_size
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)