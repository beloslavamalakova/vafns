import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

class OrnsteinUhlenbeckTransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim, noise_dim, vasicek, a, b, delta, shortrate, k, dt, hidden_size=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.hidden_size = hidden_size
        self.a = a
        self.b = b
        self.delta = delta
        self.shortrate = shortrate
        self.k = k
        self.dt = dt

        # Correct input size to match concatenated tensor
        input_size = latent_dim + action_dim + noise_dim  # 8 + 8 + 8 = 24
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Ensure input_size matches concatenated input (24)
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),  # Output size is latent_dim (8)
        )

    def forward(self, latent, action, noise):
        # Concatenate latent, action, and noise vectors
        inputs = torch.cat([latent, action, noise], dim=-1)
        latent_out = self.net(inputs)  # Pass through the network
        return latent_out



# Deep Variational Bayes Filter
class BayesFilter(nn.Module):
    def __init__(
        self,
        transition_model,
        noise_dim,
        action_dim,
        latent_dim,
        input_dim,
        hidden_size,
        kl_weight,
        annealing_steps=100,
    ):
        super().__init__()
        self.transition_model = transition_model
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.kl_weight = kl_weight

        self.extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
        )

        self.inference = nn.Sequential(
            nn.Linear(noise_dim + latent_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * noise_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

        self.anneal_steps = annealing_steps
        self.anneal_rate = 1 / annealing_steps if annealing_steps > 0 else 1.0

    def forward(self, observations, actions):
        seq_len, batch_size = observations.shape[:2]

        latents = []
        losses = []

        # Initial latent state
        z_t = torch.zeros(batch_size, self.latent_dim, device=observations.device)

        for t in range(seq_len):
            x_t = observations[t]
            u_t = actions[t] if actions is not None else torch.zeros_like(z_t)

            # Infer noise distribution
            noise_params = self.inference(torch.cat([z_t, u_t], dim=-1))
            mu, log_std = noise_params.chunk(2, dim=-1)
            std = torch.exp(log_std)
            noise = torch.randn_like(std) * std + mu

            # Update latent state using transition model
            z_t = self.transition_model(z_t, u_t, noise)
            latents.append(z_t)

            # Decode observations
            x_pred = self.decoder(z_t)
            rec_loss = nn.functional.mse_loss(x_pred, x_t, reduction='mean')

            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + log_std - mu.pow(2) - std.pow(2)) / batch_size

            # Total loss
            loss = rec_loss + self.kl_weight * kl_loss
            losses.append(loss)

        return torch.mean(torch.stack(losses))

# Data Preparation
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def normalize_data(data):
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data)
    return normalized, scaler

def prepare_dataloader(data, seq_length, batch_size):
    dataset = TimeSeriesDataset(data, seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Example Usage
if __name__ == "__main__":
    # Example data loading
    data = np.random.rand(1000, 5)  # Replace with actual dataset
    data_normalized, _ = normalize_data(data)

    # DataLoader
    seq_length = 30
    batch_size = 64
    data_loader = prepare_dataloader(data_normalized, seq_length, batch_size)

    # Transition model
    ou_model = OrnsteinUhlenbeckTransitionModel(
        latent_dim=8,
        action_dim=0,
        noise_dim=8,
        vasicek=1,
        a=0.2,
        b=0.05,
        delta=0.01,
        shortrate=0.02,
        k=0.3,
        dt=0.1,
    )

    # DVBF Model
    dvbf = BayesFilter(
        transition_model=ou_model,
        noise_dim=8,
        action_dim=0,
        latent_dim=8,
        input_dim=data.shape[1],
        hidden_size=64,
        kl_weight=0.01,
        annealing_steps=100,
    )


    # Optimizer
    optimizer = torch.optim.Adam(dvbf.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for observations, _ in data_loader:
            optimizer.zero_grad()
            loss = dvbf(observations, actions=None)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
