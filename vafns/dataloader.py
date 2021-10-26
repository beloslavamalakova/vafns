import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from   sklearn.preprocessing  import StandardScaler
import pandas as pd
import torch.nn as nn
import math
import matplotlib as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class YieldCurves(Dataset):

    def __init__(self, t): #TODO to figure out the agrs
        super.__init__()

        kaggle= pd.read_csv("../kaggle.csv", parse_dates=['date'], index_col='date')
        kaggle['Date'] = pd.to_datetime(kaggle['Date'])
        kaggle = kaggle.set_index('Date')

        kaggle_yields = kaggle['1 yr':'30YR']

        self.x=(dataset[1:, :])
        self.y=(dataset[[1-2]:, :])
        self.t=t

        x_train = self.x
        y_train = StandardScaler.transform(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__ (self):
        return self.kaggle

    def __getitem__(self, index):
        if index + self.t <= len():
           return self.y_train[index : index + self.t], self.x_train[index : index + self.t]
        else:
           raise TypeError("Index must be within the range of t")

dataset = YieldCurves()
dataset.head(3)

def main():

    device: torch.device

    plt.figure(figsize=(11,4), dpi= 80)
    pd.plotting.autocorrelation_plot(dataset.loc['1997-03-01': '2019-09-01', '1yr':'30YR'])

    # outlier process- filtering out the top and bottom 2% of the data; init
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion() #the args in the parenthesis
    print(loss)

    # splitting the dataset 90/10, train_test_split-- shuffle=False

if __name__ == "__main__":
    main()