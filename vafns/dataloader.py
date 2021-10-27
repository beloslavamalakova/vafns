import torch
from torch.utils.data import Dataset
from   sklearn.preprocessing  import StandardScaler
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt


class YieldCurves(Dataset):

    def __init__(self, num_predictions, test, t): 
        super.__init__()

        kaggle= pd.read_csv("../kaggle.csv", parse_dates=['date'], index_col='date')
        kaggle['Date'] = pd.to_datetime(kaggle['Date'])
        kaggle = kaggle.set_index('Date')

        self.num_predictions=num_predictions
        self.test=test
        self.t=t

        self.yields=kaggle[['Date'-'30YR']:, :]

        num_predictions=1

        self.train=self.yields[:int(len(self.yields)*0.95)]
        self.validation=self.yields[-int(len(self.yields)*0.05):]

        train=StandardScaler.transform(self.train)
        validation=StandardScaler.transform(self.validation)


        if test == False:
            return 
        else:
            return #if true another dataset?

        #TODO outlier process- filtering out the top and bottom 2% of the data




    def __len__ (self):
        return self.len(self.yields-self.t-self.num_predictions)

    def __getitem__(self, index):
        if index + self.t < len(self.yields):
           return self.yields[index: index + self.t + self.num_predictions +1]
        else:
           raise TypeError("Index must be within the range of t")


def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dataset = YieldCurves()

    plt.figure(figsize=(11,4), dpi= 80)
    pd.plotting.autocorrelation_plot(dataset.loc['1997-03-01': '2019-09-01', '1yr':'30YR'])
 
    criterion = nn.CrossEntropyLoss()
    loss = criterion()
    print(loss)


if __name__ == "__main__":
    main()