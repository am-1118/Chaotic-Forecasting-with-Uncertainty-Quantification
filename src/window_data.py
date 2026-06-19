import os
import numpy as np

def create_windows(data_2d: np.ndarray, L: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms a 2D time series into 3D supervised learning windows.
    
    Args:
        data_2d: Array of shape (T, 40)
        L: Lookback window length
        
    Returns:
        X: Input tensor of shape (T-L, L, 40)
        y: Target tensor of shape (T-L, 40)
    """
    X, y = [], []
    for i in range(len(data_2d) - L):
        X.append(data_2d[i : i + L])
        y.append(data_2d[i + L])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    print("Loading raw data splits...")
    train_data = np.load('data/train.npy')
    val_data = np.load('data/val.npy')
    test_data = np.load('data/test.npy')
    
    L = 10
    print(f"Applying sliding window (L={L})...")
    
    X_train, y_train = create_windows(train_data, L)
    X_val, y_val = create_windows(val_data, L)
    X_test, y_test = create_windows(test_data, L)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    print("Saving windowed datasets to 'data/'...")
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/y_val.npy', y_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    
    print("Data preparation complete!")

if __name__ == '__main__':
    main()