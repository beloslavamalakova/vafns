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

        #self.yields -- all the yields; 
        yields=kaggle['1 yr':'30YR']
        yields=self.yields

        # num_predictions -- how many steps ahead to predict --1 or 2

        #default value test=false, if true another dataset, train_test_split

        #self.x=(dataset[1:, :])
        #self.y=(dataset[[1-2]:, :])

        x_train = self.x
        y_train = StandardScaler.transform(self.y)

    def __len__ (self):
        return self.len(self.yields-self.t-self.num_predictions)

    #getitem ot index do index pkus t; index +t+num_predictions+1 -- x pyrwite t; predictionite-the following num predictions
    def __getitem__(self, index):
        if index + self.t < len():
           return self.y_train[index : index + self.t], self.x_train[index : index + self.t]
        else:
           raise TypeError("Index must be within the range of t")

def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dataset = YieldCurves()

    plt.figure(figsize=(11,4), dpi= 80)
    pd.plotting.autocorrelation_plot(dataset.loc['1997-03-01': '2019-09-01', '1yr':'30YR'])

    #TODO outlier process- filtering out the top and bottom 2% of the data; init
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion() #the args in the parenthesis
    print(loss)

if __name__ == "__main__":
    main()