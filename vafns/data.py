import torch
import pandas as pd
import numpy as np
from unicodedata import normalize
from torch.utils.data import Dataset


class Fed(Dataset):
    def __init__(self, t, test=False, num_predictions=1):

        self.num_predictions = num_predictions
        self.test = test
        self.t = t

        def clean(x):
            if isinstance(x, str):
                return normalize('NFKC', x).strip()
            else:
                return x

        yields = pd.read_html(
            "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldAll", match='30 yr')
        yields = yields.applymap(clean)
        yields.columns = yields.columns.to_series().apply(clean)

        col_type = {
            'Date': 'string',
            '1 mo': 'float',
            '2 mo': 'float',
            '3 mo': 'float',
            '6 mo': 'float',
            '1 yr': 'float',
            '2 yr': 'float',
            '3 yr': 'float',
            '5 yr': 'float',
            '7 yr': 'float',
            '10 yr': 'float',
            '20 yr': 'float',
            '30 yr': 'float',
        }

        clean_dict = {'%': '', '−': '-', '\(est\)': ''}
        yields = yields.replace(clean_dict, regex=True).replace(
            {'N/A': np.nan}).astype(col_type)

        self.train = yields[:int(len(self.yields)*0.95)]
        self.validation = yields[-int(len(self.yields)*0.05):]

        self.data = self.train
        self.data = torch.tensor(self.data)

        min_value = 3
        max_value = 8

        self.data = torch.clamp(self.data, min_value, max_value)

        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

        self.normalized = (self.data - torch.mean(self.data)
                           ) / torch.std(self.data)

        self.data = (self.data - torch.mean(self.data)) / torch.std(self.data)

        if test:
            self.data = self.validation
        else:
            self.data = self.train

    def __len__(self):
        return len(self.data-self.t-self.num_predictions)

    def __getitem__(self, idx):
        return self.data[idx: idx + self.t], self.data[idx + self.t: idx + self.t + self.num_predictions]

    @classmethod
    def denormalizing(self, x):
        x = x * self.std + self.mean


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dataset = Fed()
    dataset.head()


if __name__ == "__main__":
    main()
