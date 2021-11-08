import torch
import pandas as pd
import numpy as np
from unicodedata import normalize
import math
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Fed(Dataset):
    def __init__(self, num_predictions, test, t):

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



        train=yields[:int(len(self.yields)*0.95)]
        validation=yields[-int(len(self.yields)*0.05):]

        train=StandardScaler.transform(train)
        validation=StandardScaler.transform(validation)

        num_predictions= 1 #logical value??

        min_value=0
        max_value=1
        torch.clamp(train, min_value, max_value)
        
        #assigned values for a_n, x and y?

        #if else for test boolean?

    def __len__(self):
        return self.len(self.yields-self.t-self.num_predictions)

    def __getitem__(self):
        # a_n, x, y, train_seq_lenght, test_seq_lenght
        pass



def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    dataset= Fed()
    dataset.head()

if __name__ == "__main__":
    main()
