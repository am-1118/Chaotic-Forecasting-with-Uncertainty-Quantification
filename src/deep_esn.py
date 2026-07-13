import torch
import torch.nn as nn
import numpy as np

class LeakyRecurrentLayer(nn.Module):
    """
    A single untrained reservoir layer with leaky-integrator dynamics.
    """
    def __init__(self, in_features: int, out_features: int, leaky_rate: float, spectral_radius: float, input_scaling: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a = leaky_rate
        
        # 1. Input/Inter-layer Weights (W_in)
        # Uniformly distributed in [-input_scaling, input_scaling]
        self.W_in = nn.Parameter(
            (torch.rand(out_features, in_features) * 2 - 1) * input_scaling, 
            requires_grad=False
        )
        
        # 2. Recurrent Weights (W_hat)
        # Uniformly distributed in [-1, 1], then scaled by spectral radius
        W_hat_raw = torch.rand(out_features, out_features) * 2 - 1
        
        # Calculate the spectral radius of W_hat_raw
        eigenvalues = torch.linalg.eigvals(W_hat_raw)
        current_radius = torch.max(torch.abs(eigenvalues)).item()
        
        # Scale W_hat so its spectral radius matches the target
        self.W_hat = nn.Parameter(
            (W_hat_raw / current_radius) * spectral_radius, 
            requires_grad=False
        )
        
    def forward(self, u: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """
        Updates the state using the leaky-integrator equation.
        """
        # x(t) = (1 - a) * x(t-1) + a * tanh(W_in * u(t) + W_hat * x(t-1))
        update = torch.tanh(torch.nn.functional.linear(u, self.W_in) + 
                            torch.nn.functional.linear(x_prev, self.W_hat))
        
        x_next = (1 - self.a) * x_prev + self.a * update
        return x_next


class DeepESN(nn.Module):
    """
    A dynamically constructed Deep Echo State Network.
    """
    def __init__(self, input_dim: int, reservoir_size: int = 100, leaky_rate: float = 0.1, 
                 spectral_radius: float = 0.9, input_scaling: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.N_R = reservoir_size
        self.a = leaky_rate
        self.rho = spectral_radius
        self.scale_in = input_scaling
        
        # ModuleList to hold our dynamically added layers
        self.layers = nn.ModuleList()
        
    def add_layer(self):
        """Appends a new recurrent layer to the top of the stack."""
        in_features = self.input_dim if len(self.layers) == 0 else self.N_R
        
        new_layer = LeakyRecurrentLayer(
            in_features=in_features, 
            out_features=self.N_R, 
            leaky_rate=self.a, 
            spectral_radius=self.rho, 
            input_scaling=self.scale_in
        )
        self.layers.append(new_layer)
        
    def compute_layer_states(self, u_seq: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Passes the input sequence through the network up to a specific layer 
        and returns the full state history of that top layer.
        """
        time_steps, batch_size, _ = u_seq.shape
        device = u_seq.device
        
        # We need to compute states sequentially through time and layers
        # Store the current input to the layer being processed
        current_input = u_seq
        
        for i in range(layer_idx + 1):
            layer = self.layers[i]
            layer_states = []
            x_prev = torch.zeros(batch_size, self.N_R, device=device)
            
            for t in range(time_steps):
                x_prev = layer(current_input[t], x_prev)
                layer_states.append(x_prev)
                
            # The output of this layer becomes the input to the next
            current_input = torch.stack(layer_states)
            
        return current_input

    def design_architecture(self, u_train: torch.Tensor, max_layers: int = 50, eta: float = 0.01, washout: int = 20):
        """
        Implements the automatic design algorithm based on frequency analysis.
        Adds layers incrementally until the shift in the spectral centroid converges.
        """
        print("Starting Dynamic Architectural Design via FFT...")
        device = u_train.device
        
        # Ensure sequence is (Time, Batch, Features)
        if u_train.dim() == 2:
            u_train = u_train.unsqueeze(1)
            
        time_steps = u_train.shape[0]
        # Calculate available frequencies (cycles/timestep)
        # f^{(l)} <- [1 : floor(timesteps/2)] / timesteps
        f_l = torch.fft.rfftfreq(time_steps - washout, d=1.0).to(device)
        
        prev_centroid = None
        prev_spread = None
        
        for l in range(max_layers):
            self.add_layer()
            self.layers[-1].to(device)
            # 1. Compute state on layer l
            state_seq = self.compute_layer_states(u_train, layer_idx=l)
            
            # Discard initial transient dynamics (washout)
            state_seq = state_seq[washout:, 0, :] # Shape: (T_eff, N_R)
            
            # 2. FFT Algorithm 
            # Calculate magnitudes for positive frequencies
            fft_vals = torch.fft.rfft(state_seq, dim=0)
            comps_u = torch.abs(fft_vals)
            
            # Average magnitudes across units (columns)
            p_l = torch.mean(comps_u, dim=1) 
            
            # 3. Compute Spectral Centroid (mu)
            mu_l = torch.sum(p_l * f_l) / torch.sum(p_l)
            
            # 4. Compute Spectral Spread (sigma)
            variance = torch.sum(p_l * ((f_l - mu_l) ** 2)) / torch.sum(p_l)
            sigma_l = torch.sqrt(variance)
            
            print(f"Layer {l+1:02d} | Centroid: {mu_l.item():.6f} | Spread: {sigma_l.item():.6f}")
            
            # 5. Stop Condition Check (Algorithm 2)
            if l > 0:
                shift = torch.abs(mu_l - prev_centroid)
                threshold = prev_spread * eta
                
                if shift <= threshold:
                    print(f">>> Stop condition met at Layer {l+1}! Shift ({shift.item():.6f}) <= Threshold ({threshold.item():.6f})")
                    break
                    
            prev_centroid = mu_l
            prev_spread = sigma_l
            
        print(f"Final Architecture Designed: {len(self.layers)} Layers.")
        
    def forward(self, u_seq: torch.Tensor) -> torch.Tensor:
        """
        Passes data through all designed layers.
        Returns the concatenated global state [x^(1)(t), ..., x^(N_L)(t)].
        """
        time_steps, batch_size, _ = u_seq.shape
        device = u_seq.device
        
        global_states = []
        current_input = u_seq
        
        for layer in self.layers:
            layer_states = []
            x_prev = torch.zeros(batch_size, self.N_R, device=device)
            
            for t in range(time_steps):
                x_prev = layer(current_input[t], x_prev)
                layer_states.append(x_prev)
                
            # The output of this layer is the input to the next
            current_input = torch.stack(layer_states)
            global_states.append(current_input)
            
        # Concatenate states from all layers along the feature dimension
        # Shape: (Time, Batch, N_L * N_R)
        return torch.cat(global_states, dim=-1)