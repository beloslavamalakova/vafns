import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt


# ====== Data Preprocessing ======
def load_and_preprocess_data(file_path, seq_length):
    # Load your real dataset
    df = pd.read_csv(file_path)
    df = df.dropna()  # Drop rows with NaN values

    # Drop unnecessary columns (e.g., Date)
    data = df.drop(columns=["Date"]).values

    # Normalize the dataset
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    print(f"Min Value in Dataset: {data_tensor.min()}")
    print(f"Max Value in Dataset: {data_tensor.max()}")
    print(f"Are there NaN values? {torch.isnan(data_tensor).any()}")

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


# ====== Simple Transition Model ======
class SimpleTransitionModel(nn.Module):
    def __init__(self, latent_dim, noise_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + noise_dim, 32),  # Increased hidden size for real data
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, latent, noise):
        inputs = torch.cat([latent, noise], dim=-1)
        return self.net(inputs)

# ====== Simplified DVBF ======
class SimplifiedBayesFilter(nn.Module):
    def __init__(self, transition_model, latent_dim, input_dim):
        super().__init__()
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
        all_x_t = []
        all_x_pred = []

        for t in range(seq_len):
            x_t = observations[:, t, :]
            z_t = self.encoder(x_t)
            noise = torch.randn(batch_size * num_sequences, self.latent_dim)
            z_t = self.transition_model(z_t, noise)
            x_pred = self.decoder(z_t)

            all_x_t.append(x_t.detach().cpu().numpy())
            all_x_pred.append(x_pred.detach().cpu().numpy())

            print(f"x_t (input): {x_t}")
            print(f"x_pred (prediction): {x_pred}")

        return np.array(all_x_t), np.array(all_x_pred)


# ====== Running the DVBF ======
if __name__ == "__main__":
    # Parameters
    file_path = "yield-curve-rates.csv"  # Replace with your dataset path
    seq_length = 5  # Sequence length
    input_dim = 13  # Number of features in your dataset
    latent_dim = 8  # Latent space dimension
    noise_dim = latent_dim  # Match noise to latent dimension

    # Load data
    sequences, scaler = load_and_preprocess_data(file_path, seq_length)
    print(f"Dataset Shape: {sequences.shape}")

    # Initialize models
    transition_model = SimpleTransitionModel(latent_dim, noise_dim)
    dvbf = SimplifiedBayesFilter(transition_model, latent_dim, input_dim)

    # Forward pass
    print("Running DVBF on Real Dataset...")
    actual, predicted = dvbf(sequences)

    # Evaluate metrics
    print("Evaluating Performance Metrics...")
    metrics = evaluate_performance_with_metrics(actual, predicted)

    # Plot metrics
    avg_metrics_df = pd.DataFrame(metrics, index=["Average"])
    avg_metrics_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Average Metrics Across All Features")
    plt.ylabel("Metric Value")
    plt.grid()
    plt.show()
