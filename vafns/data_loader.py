import pandas as pd
from pandas import read_html
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from math import sqrt, log
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error
from unicodedata import normalize

# labels === inputs[t+1]; normalize data with zero mean and unit variance; reverse/inverse normalize when outputting #inverse_transform(); checking for auto-correlation errors;
# outlier process- filtering out the top and bottom 2% of the data; normalizing and rescaling [0:1]; splitting the dataset 90/10; cross-entropy= entropy plus KL-divergence;
#  train_test_split-- shuffle=False; html parser

class Dataloader():
  def clean_normalize_whitespace(x):
    if isinstance(x, str):
        return normalize('NFKC', x).strip()
    else:
        return x

  table_yc= pd.read_html('../fs_yc.html')
  print(table_yc.head(5))

  table_yc = table_yc.applymap(clean_normalize_whitespace)
  table_yc.columns = table_yc.columns.to_series().apply(clean_normalize_whitespace)

  col_type = {
      'Date': 'int',
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
  table_yc = table_yc.replace(clean_dict, regex=True).replace({'-n/a ': np.nan}).astype(col_type)



  def __init__(self, table_yc, t):
        super(Dataloader, self).__init__()

        
        self.t = t
        self.table_yc=table_yc
        file_out = pd.read_csv(table_yc, header=0, index_col=0)
        x = file_out.iloc[:, 1].values
        y = file_out.iloc[:, [2, 12]].values

        sc = StandardScaler()
        x_train = x
        y_train = sc.transform(y)

        self.X_train = torch.tensor(x_train)
        self.y_train = torch.tensor(y_train)


  def __len__(self):
       return len(self.x_train)


  def __getitem__(self, idx):
      if idx + self.t <= len():
        return self.y_train[idx : idx + self.t], self.x_train[idx : idx + self.t]
      else:
        raise TypeError("Index must be within the range of t")


  def main():
    dataset=DataLoader()
    values = file_out.values
    values = values.reshape((len(values), 1))
    scaler = StandardScaler()
    scaler = scaler.fit(values)
    print("Mean: %f, StandardDeviation: %f" % (scaler.mean_, sqrt(scaler.var_)))

    normalized = scaler.transform(values)
    for i in range():
        print(normalized[i])
    reversed = scaler.inverse_transform(normalized)
    for i in range():
        print(reversed[i])

    X_train, y_train, X_test, y_test = train_test_split(
        x, y, test_size=0.1, shuffle=False
    )
    ee = OneClassSVM(nu=0.04)
    yhat = ee.fit_predict(X_train)
    yc = yhat != -1
    X_train, y_train = X_train[yc, :], y_train[yc]

    print(sm.graphics.tsa.asf(table_yc, nlags=40))  # auto-correlation check

    def cross_entropy_loss(yHat, y):
        if y == 1:
            return -np.log(yHat)
        else:
            return -np.log(1 - yHat)


