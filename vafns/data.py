import torch
import pandas as pd
import numpy as np
from unicodedata import normalize
import math
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Fed(Dataset):
    def __init__(self):
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
            '2 mo': 'int',
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

        def cleaning(yields):
            clean_dict = {'%': '', '−': '-', '\(est\)': ''}
            yields= yields.replace(clean_dict, regex=True).replace({'N/A': np.nan}).astype(col_type)

        

    def __getitem__(self):
        pass

    def __len__(self):
        pass


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    dataset= Fed()
    yields.head()




if __name__ == "__main__":
    main()
