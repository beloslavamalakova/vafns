#for the kaggle dataset 
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from   sklearn.preprocessing  import StandardScaler #main?
import numpy as np
import math

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#abstract class
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

#reverse/inverse normalize when outputting inverse_transform(); 


#dataset class,  labels==inputs[t+1]
class YieldCurves(Dataset):

    def __init__(self, t):

        self.t=t
        xy=np.loadtxt('/home/beloslava/Projects/vafns/kaggle.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples= xy.shape[0]
        self.x_data=torch.from_numpy(xy[:, 1:])
        self.y_data=torch.from_numpy(xy[:, [0]]) #n_samples, 1

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__ (self):
        return self.n_samples

#creating the dataset
dataset = YieldCurves()
dataset.head(5)

first_data =dataset[0]
features, labels = first_data
print(features, labels)


def main():
    device: torch.device

# checking for auto-correlation errors;


# outlier process- filtering out the top and bottom 2% of the data; 

#cross-entropy= entropy plus KL-divergence;

# normalizing and rescaling [0:1]; splitting the dataset 90/10, train_test_split-- shuffle=False

if __name__ == "__main__":
    main()