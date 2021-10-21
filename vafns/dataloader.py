import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

#Dataset - location; initializing, reading the dataset with numpy
##the first column is the time (dates)

class YieldCurves(Dataset):

    def __init__(self):

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


first_data =dataset[0]
features, labels = first_data
print(features, labels)



#create the dataloader !do not forget -- shuffle=False