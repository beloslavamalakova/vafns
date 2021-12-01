import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt


class YieldCurves(Dataset):

    def __init__(self, num_predictions, test, t):
        super.__init__()

        self.num_predictions = num_predictions
        self.test = test
        self.t = t

        kaggle = pd.read_csv(
            "../kaggle.csv", parse_dates=['date'], index_col='date')
        kaggle['Date'] = pd.to_datetime(kaggle['Date'])
        kaggle = kaggle.set_index('Date')

        yields = kaggle[['Date'-'30YR']:, :]

        if test:
            self.data = yields[-int(len(yields)*0.05):]
        else:
            self.data = yields[:int(len(yields)*0.95)]

        self.data = self.test
        self.data = torch.tensor(self.data)

        self.t = self.t+1

        min_value = 3
        max_value = 8

        self.data = torch.clamp(self.data, min_value, max_value)

        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

        self.normalized = (self.data - torch.mean(self.data)
                           ) / torch.std(self.data)

        self.data = (self.data - torch.mean(self.data)) / torch.std(self.data)

    def __len__(self):
        return len(self.data-self.t-self.num_predictions)

    def __getitem__(self, idx):
        return self.data[idx: idx + self.t], self.data[idx + self.t: idx + self.t + self.num_predictions]

    @classmethod
    def denormalizing(self):
        for i in range(self.data):
            i = i * self.std + self.mean
            return i

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = YieldCurves()
    dataset.to(device)

    plt.figure(100, figsize=(12,5))
    dataset.plot()


if __name__ == "__main__":
    main()