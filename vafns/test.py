import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Dummy Dataset
data = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0]
])

# Normalize the dataset
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Convert to PyTorch tensor
data_tensor = torch.tensor(data_normalized, dtype=torch.float32)

# Prepare a sequence of length 3
seq_length = 3  # Sequence length for testing
data_tensor = data_tensor.unsqueeze(0)  # Add batch dimension: shape (1, 5, 3)
data_tensor = data_tensor[:, :seq_length, :]  # Use only part of the sequence

# Simplified Transition Model
class SimpleTransitionModel(nn.Module):
    def __init__(self, latent_dim, noise_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim

        # Simple linear transformation for testing
        self.net = nn.Sequential(
            nn.Linear(latent_dim + noise_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )

    def forward(self, latent, noise):
        print(f"Transition Model Input Latent: {latent.shape}, Noise: {noise.shape}")
        inputs = torch.cat([latent, noise], dim=-1)
        print(f"Transition Model Concatenated Input: {inputs.shape}")
        output = self.net(inputs)
        print(f"Transition Model Output: {output.shape}")
        return output

# Simplified DVBF
class SimplifiedBayesFilter(nn.Module):
    def __init__(self, transition_model, latent_dim, input_dim):
        super().__init__()
        self.transition_model = transition_model
        self.latent_dim = latent_dim

        # Encoder to map inputs to latent space
        self.encoder = nn.Linear(input_dim, latent_dim)

        # Decoder to map latents back to input space
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, observations):
        batch_size, seq_len, input_dim = observations.shape
        print(f"Observations Shape: {observations.shape}")

        # Initialize latent state
        z_t = torch.zeros(batch_size, self.latent_dim)
        print(f"Initial Latent State: {z_t.shape}")

        losses = []
        for t in range(seq_len):
            print(f"Time Step {t + 1}")
            x_t = observations[:, t, :]
            print(f"Current Observation: {x_t.shape}")

            # Encode to latent space
            z_t = self.encoder(x_t)
            print(f"Encoded Latent: {z_t.shape}")

            # Predict next latent state
            noise = torch.randn(batch_size, self.latent_dim)
            print(f"Generated Noise: {noise.shape}")
            z_t = self.transition_model(z_t, noise)

            # Decode to observation space
            x_pred = self.decoder(z_t)
            print(f"Decoded Prediction: {x_pred.shape}")

            # Compute loss
            loss = nn.functional.mse_loss(x_pred, x_t)
            print(f"Step Loss: {loss.item()}")
            losses.append(loss)

        total_loss = torch.mean(torch.stack(losses))
        print(f"Total Loss: {total_loss.item()}")
        return total_loss

# Initialize models
latent_dim = 4
noise_dim = 4
input_dim = 3

transition_model = SimpleTransitionModel(latent_dim, noise_dim)
dvbf = SimplifiedBayesFilter(transition_model, latent_dim, input_dim)

# Forward pass
print("Starting DVBF Forward Pass")
loss = dvbf(data_tensor)
print(f"Final Loss: {loss.item()}")
