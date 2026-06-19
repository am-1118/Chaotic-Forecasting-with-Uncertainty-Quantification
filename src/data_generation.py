import os
import numpy as np
from scipy.integrate import solve_ivp

def lorenz96_deriv(t: float, x: np.ndarray, F: float) -> np.ndarray:
    """
    Computes the derivative of the Lorenz 96 system.
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    """
    # np.roll shifts the array elements along the specified axis.
    # shift=-1 -> x_{i+1}, shift=2 -> x_{i-2}, shift=1 -> x_{i-1}
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def generate_lorenz96_data(
    N: int = 40, 
    F: float = 8.0, 
    t_span: tuple = (0, 250), 
    dt: float = 0.01, 
    transient_t: float = 50.0
) -> np.ndarray:
    """
    Generates time series data for the Lorenz 96 system.
    """
    print(f"Integrating Lorenz 96 system (N={N}, F={F})...")
    
    # Initial conditions: x_i = F, except a tiny perturbation at x_19
    x0 = np.full(N, F, dtype=np.float64)
    x0[19] += 0.01
    
    # Create strict evaluation points
    num_points = int((t_span[1] - t_span[0]) / dt) + 1
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    # Solve IVP using RK45
    sol = solve_ivp(
        fun=lambda t, x: lorenz96_deriv(t, x, F),
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method='RK45'
    )
    
    # Determine the index to cut off transient dynamics
    transient_idx = int(transient_t / dt) + 1
    
    # Transpose to get shape (time, variables) and cast to float32
    data = sol.y[:, transient_idx:].T.astype(np.float32)
    return data

def main():
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    data = generate_lorenz96_data()
    print(f"Generated data shape: {data.shape}, dtype: {data.dtype}")
    
    # Split definitions
    total_steps = data.shape[0]
    train_end = int(total_steps * 0.60)
    val_end = train_end + int(total_steps * 0.20)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Train split: {train_data.shape}")
    print(f"Validation split: {val_data.shape}")
    print(f"Test split: {test_data.shape}")
    
    # Save arrays
    np.save('data/lorenz96.npy', data)
    np.save('data/train.npy', train_data)
    np.save('data/val.npy', val_data)
    np.save('data/test.npy', test_data)
    print("Files successfully saved in the 'data/' directory.")

if __name__ == '__main__':
    main()