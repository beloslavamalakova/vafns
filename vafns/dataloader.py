import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class YieldCurves(Dataset):
    def __init__(self):
        xy=np.loadtxt('/home/beloslava/Projects/vafns/kaggle.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x=torch.from_numpy(xy[:, 1:])
        self.y=torch.from_numpy(xy[:, [0]]) #n_samples, 1
        self.n_samples= xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self,y[index]

    def __len__ (self):
        return self.n_samples
dataset = YieldCurves()
first_data =dataset[0]
features, labels = first_data
print(features, labels)
