import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Custom Dataset for Time-Series Data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Normalize Data
def normalize_data(data):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler

# Split Data into Train-Test
def split_train_test(data, train_ratio=0.9):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Generate K-Folds
def generate_k_folds(data, k, seq_length):
    fold_size = len(data) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        if i == k - 1:  # Ensure the last fold gets the remaining data
            end = len(data)
        folds.append(TimeSeriesDataset(data[start:end], seq_length))
    return folds

# Prepare DataLoader
def prepare_dataloader(data, seq_length, batch_size):
    dataset = TimeSeriesDataset(data, seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Example Usage
if __name__ == "__main__":
    # Load Dataset
    data = np.genfromtxt('cleaned_dataset.csv', delimiter=',', skip_header=1)  # Replace with your dataset

    # Normalize Dataset
    data_normalized, scaler = normalize_data(data[:, 1:])  # first column is Date

    # Train-Test Split
    train_data, test_data = split_train_test(data_normalized, train_ratio=0.9)

    # Generate K-Folds from Training Data
    k = 5
    seq_length = 30
    folds = generate_k_folds(train_data, k, seq_length)

    # DataLoader for Test Set
    test_loader = prepare_dataloader(test_data, seq_length, batch_size=64)

    # Print Example
    print(f"Train Data Size: {len(train_data)}, Test Data Size: {len(test_data)}")
    print(f"Number of Folds: {len(folds)}, Fold Size: {[len(f) for f in folds]}")
