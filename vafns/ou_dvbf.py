# 11 Jan
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


# ====== Ornstein-Uhlenbeck Transition Model ======
class OrnsteinUhlenbeckTransition(nn.Module):
    def __init__(self, latent_dim, theta=0.15, mu=0.0, sigma=0.2, dt=1.0):
        """
        Ornstein-Uhlenbeck Transition Model.
        :param latent_dim: Dimension of the latent space.
        :param theta: Rate of mean reversion.
        :param mu: Mean of the process.
        :param sigma: Volatility.
        :param dt: Time step.
        """
        super(OrnsteinUhlenbeckTransition, self).__init__()
        self.latent_dim = latent_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def forward(self, latent, noise):
        """
        Transition model for the Ornstein-Uhlenbeck process.
        :param latent: Current latent state.
        :param noise: Gaussian noise to inject.
        :return: Next latent state.
        """
        # OU Process formula
        next_latent = latent + self.theta * (self.mu - latent) * self.dt + self.sigma * torch.sqrt(
            torch.tensor(self.dt)
        ) * noise
        return next_latent


# ====== Simplified DVBF with OU Transition ======
class SimplifiedBayesFilterWithOU(nn.Module):
    def __init__(self, transition_model, latent_dim, input_dim):
        super(SimplifiedBayesFilterWithOU, self).__init__()
        self.transition_model = transition_model
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(input_dim, latent_dim)  # Maps input to latent space
        self.decoder = nn.Linear(latent_dim, input_dim)  # Maps latent space to input

    def forward(self, observations):
        batch_size, num_sequences, seq_len, input_dim = observations.shape
        print(f"Observations Shape: {observations.shape}")

        # Reshape observations
        observations = observations.view(batch_size * num_sequences, seq_len, input_dim)
        z_t = torch.zeros(batch_size * num_sequences, self.latent_dim)
        losses = []
        actual = []
        predicted = []

        for t in range(seq_len):
            x_t = observations[:, t, :]
            z_t = self.encoder(x_t)
            noise = torch.randn(batch_size * num_sequences, self.latent_dim)
            z_t = self.transition_model(z_t, noise)  # OU Process
            x_pred = self.decoder(z_t)

            # Store actual and predicted for evaluation
            actual.append(x_t.detach().cpu().numpy())
            predicted.append(x_pred.detach().cpu().numpy())

            # Debugging outputs
            print(f"x_t (input): {x_t}")
            print(f"x_pred (prediction): {x_pred}")
            print(f"Loss at timestep {t + 1}: {nn.functional.mse_loss(x_pred, x_t).item()}")

            loss = nn.functional.mse_loss(x_pred, x_t)
            losses.append(loss)

        total_loss = torch.mean(torch.stack(losses))
        print(f"Total Loss: {total_loss}")

        # Convert to numpy arrays for evaluation
        actual = np.array(actual)
        predicted = np.array(predicted)

        return total_loss, actual, predicted


# ====== Evaluation Metrics ======
def evaluate_performance_with_metrics(actual, predicted):
    """
    Evaluates the model's performance using multiple metrics.
    :param actual: Actual observations, shape (timesteps, features)
    :param predicted: Predicted observations, shape (timesteps, features)
    """
    metrics = {
        "MSE": [],
        "MAE": [],
        "R²": [],
        "Explained Variance": []
    }

    for t in range(actual.shape[0]):  # Iterate over timesteps
        y_true = actual[t]
        y_pred = predicted[t]

        # Compute metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)

        # Append to metrics dictionary
        metrics["MSE"].append(mse)
        metrics["MAE"].append(mae)
        metrics["R²"].append(r2)
        metrics["Explained Variance"].append(explained_var)

        print(f"Timestep {t + 1}:")
        print(f"  MSE: {mse}")
        print(f"  MAE: {mae}")
        print(f"  R²: {r2}")
        print(f"  Explained Variance: {explained_var}")
        print()

    # Overall metrics (averaged across timesteps)
    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    print("Overall Metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value}")

    return avg_metrics


# ====== Main ======
if __name__ == "__main__":
    # Parameters
    file_path = "yield-curve-rates.csv"
    seq_length = 5  # Sequence length
    input_dim = 13  # Number of features in your dataset
    latent_dim = 8  # Latent space dimension
    noise_dim = latent_dim  # Match noise to latent dimension

    # Load data
    def load_and_preprocess_data(file_path, seq_length):
        df = pd.read_csv(file_path)
        df = df.dropna()  # Drop rows with NaN values
        data = df.drop(columns=["Date"]).values

        # Normalize the dataset
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data)
        data_tensor = torch.tensor(data_normalized, dtype=torch.float32)

        # Create sequences
        def create_sequences(data, seq_len):
            sequences = []
            for i in range(len(data) - seq_len + 1):
                seq = data[i:i + seq_len]
                sequences.append(seq)
            return torch.stack(sequences)

        sequences = create_sequences(data_tensor, seq_length)
        sequences = sequences.unsqueeze(0)  # Add batch dimension
        return sequences, scaler

    sequences, scaler = load_and_preprocess_data(file_path, seq_length)
    print(f"Dataset Shape: {sequences.shape}")

    # Initialize models
    ou_transition_model = OrnsteinUhlenbeckTransition(latent_dim)
    dvbf_ou = SimplifiedBayesFilterWithOU(ou_transition_model, latent_dim, input_dim)

    # Forward pass
    print("Running DVBF with OU on Real Dataset...")
    loss, actual, predicted = dvbf_ou(sequences)
    print(f"Final Loss on Real Data: {loss.item()}")

    # Evaluate metrics
    print("Evaluating Performance Metrics...")
    metrics = evaluate_performance_with_metrics(actual, predicted)

