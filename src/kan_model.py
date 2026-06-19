import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    """
    A single Kolmogorov-Arnold Network Layer.
    Edges contain learnable functions defined as:
    phi(x) = w_base * SiLU(x) + Sum(w_spline_i * RBF_i(x))
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # 1. Base Activation Weights (w_base)
        # Acts like a standard linear layer applied to the SiLU activation
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # 2. Localized Curve Weights (w_spline)
        # One weight for every grid point, on every edge connecting in to out
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        
        # 3. Fixed Grid for RBFs
        # We spread Gaussian centers from -2.0 to 2.0. 
        # (Inputs outside this will rely mostly on the base SiLU function)
        grid = torch.linspace(-2.0, 2.0, grid_size)
        self.register_buffer("grid", grid)  # Buffer means it moves to GPU but isn't a learned parameter
        
        # Variance of the Gaussians, scales with grid density
        self.sigma = 4.0 / (grid_size - 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialization is critical for KANs. 
        Base weights start standard, spline weights start near zero so the 
        model begins behaving like a standard MLP before adapting the fine curves.
        """
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, in_features)
        Returns:
            Tensor of shape (batch_size, out_features)
        """
        # --- PATH A: Base Function ---
        base_out = F.silu(x)
        # Standard matrix multiplication for the base component
        base_term = F.linear(base_out, self.base_weight)
        
        # --- PATH B: Spline/RBF Function ---
        # Expand x to compute distances to all grid points simultaneously
        # Shape: (batch_size, in_features, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Compute Gaussian activations
        # Shape: (batch_size, in_features, grid_size)
        basis = torch.exp(-((x_expanded - self.grid) / self.sigma) ** 2)
        
        # Multiply basis activations by learnable spline weights and sum
        # 'b' = batch, 'i' = in_features, 'k' = grid_size, 'o' = out_features
        # This elegantly calculates the summation at the hidden nodes in one step
        spline_term = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        
        # Combine both paths
        return base_term + spline_term


class TimeSeriesKAN(nn.Module):
    """
    The macro-architecture for chaotic time-series forecasting.
    Implements a bottleneck structure with Layer Normalization.
    """
    def __init__(self, seq_len: int = 10, num_vars: int = 40, hidden_size: int = 128, grid_size: int = 5):
        super().__init__()
        self.flatten_size = seq_len * num_vars
        
        # Layer 1: Flattened Input -> Hidden Bottleneck
        self.kan1 = KANLayer(self.flatten_size, hidden_size, grid_size)
        
        # Layer Normalization to stabilize intermediate chaotic variances
        self.ln = nn.LayerNorm(hidden_size)
        
        # Layer 2: Hidden Bottleneck -> Next Step Output
        self.kan2 = KANLayer(hidden_size, num_vars, grid_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, num_vars)
        Returns:
            Tensor of shape (batch_size, num_vars) representing the next time step
        """
        batch_size = x.size(0)
        
        # Flatten the temporal window into a single feature vector
        x = x.view(batch_size, self.flatten_size)
        
        # Pass through KAN layers
        x = self.kan1(x)
        x = self.ln(x)
        x = self.kan2(x)
        
        return x

if __name__ == '__main__':
    # A quick local test to ensure tensors flow correctly before training
    print("Testing TimeSeriesKAN Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TimeSeriesKAN(seq_len=10, num_vars=40, hidden_size=128).to(device)
    
    # Create a dummy batch of 32 samples
    dummy_input = torch.randn(32, 10, 40).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params:,}")
    print("Success! The architecture is ready.")