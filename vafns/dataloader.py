import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from   sklearn.preprocessing  import StandardScaler
import numpy as np
import pandas as pd
import math
import matplotlib as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class YieldCurves(Dataset):

    def __init__(self, t): #to decide the agrs
        super.__init__()

        kaggle= pd.read_csv("../kaggle.csv", parse_dates=['date'], index_col='date')
        kaggle['Date'] = pd.to_datetime(kaggle['Date'])
        kaggle = kaggle.set_index('Date')

        kaggle_yields = kaggle['1 yr':'30YR']

        self.x=(dataset[1:, :])
        self.y=(dataset[[1-2]:, :])
        self.t=t

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __getitem__(self, ):
        return self.

    def __len__ (self):
        return self.

    def time(self, t):
        labels==inputs[t+1]

dataset = YieldCurves()
dataset.head(3)

def main():

    device: torch.device

    plt.figure(figsize=(11,4), dpi= 80)
    pd.plotting.autocorrelation_plot(data.loc['1997-03-01': '2019-09-01', 'Yield Curves'])
    
    # outlier process- filtering out the top and bottom 2% of the data; init



# cross-entropy= default function 

# splitting the dataset 90/10, train_test_split-- shuffle=False

if __name__ == "__main__":
    main()