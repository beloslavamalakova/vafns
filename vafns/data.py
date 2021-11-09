import torch
import pandas as pd
import numpy as np
from unicodedata import normalize
import math
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Fed(Dataset):
    def __init__(self, t, test=False, num_predictions=1):

        self.num_predictions=num_predictions
        self.test=test
        self.t=t


        def clean(x):
            if isinstance(x, str):
                return normalize('NFKC', x).strip()
            else:
                return x


        yields= pd.read_html("https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldAll", match='30 yr')
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
        yields= yields.replace(clean_dict, regex=True).replace({'N/A': np.nan}).astype(col_type)


        self.train=yields[:int(len(self.yields)*0.95)]
        self.validation=yields[-int(len(self.yields)*0.05):]

        self.data=self.train

        self.train = (self.train - np.mean(self.train)) / np.std(self.train)
        self.validation = (self.validation - np.mean(self.validation)) / np.std(self.validation)

        self.train=StandardScaler.transform(self.train)
        self.validation=StandardScaler.transform(self.validation)

        min_value=3
        max_value=8

        torch.clamp(self.train, min_value, max_value)
        self.train=StandardScaler.transform(self.train)

        #to save somewehre the prior zero mean unit variance 
        #saving the values in
        #function to denormalize, *std and to add the u -- 
        

        
        if test:
            self.data=self.validation
        else:
            self.data=self.train


    def __len__(self):
        return len(self.data-self.t-self.num_predictions)

    def __getitem__(self, idx):
        return self.data[idx : idx + self.t], self.data[idx + self.t : idx + self.t + self.num_predictions]
        

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    dataset= Fed()
    dataset.head()

if __name__ == "__main__":
    main()