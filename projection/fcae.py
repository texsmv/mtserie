import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data

class TimeSeriesAE(nn.Module):
    def __init__(self):
        super(TimeSeriesAE, self).__init__()
        self.conv1 = nn.Conv1d(5, 64, 9, padding = 4)
        self.conv2 = nn.Conv1d(64, 32, 5, padding = 2)
        self.conv3 = nn.Conv1d(12, 12, 5, padding = 2)
        self.conv4 = nn.Conv1d(12, 32, 5, padding = 2)
        self.conv5 = nn.Conv1d(32, 64, 9, padding = 4)
        self.conv6 = nn.Conv1d(64, 5, 9, padding = 4)
        self.maxPool1 = nn.MaxPool1d(2)
        self.maxPool2 = nn.MaxPool1d(3)

        self.upSample1 = nn.Upsample(scale_factor=3)
        self.upSample2 = nn.Upsample(scale_factor=2)
        # maxPool3 = torch.nn.MaxPool1d(3)
        self.linear = nn.Linear(160, 60)
        self.linearOut = nn.Linear(150, 150)
        
    def forward(self, X):
        X = F.relu(self.conv1(X))
        # print(X.shape)
        X = self.maxPool1(X)
        # print(X.shape)
        X = F.relu(self.conv2(X))
        # print(X.shape)
        X = self.maxPool2(X)
        # print(X.shape)
        # X = torch.flatten(X)
        X = X.view(-1, 160)
        # print(X.shape)
        X = self.linear(X)
        X = X.view(-1, 12, 5)
        # print(X.shape)
        X = F.relu(self.conv3(X))
        # print(X.shape)
        X = self.upSample1(X)
        # print(X.shape)
        X = F.relu(self.conv4(X))
        # print(X.shape)
        X = self.upSample2(X)
        # print(X.shape)
        X = F.relu(self.conv5(X))
        # print(X.shape)
        X = self.conv6(X)
        # print(X.shape)
        X = X.view(-1, 150)
        X = self.linearOut(X)
        X = X.view(-1, 5, 30)
        # print(X.shape)s
        # print(X)
        return X
    
class DatasetMTS(data.Dataset):
    def __init__(self, X, w, s):
        self.X = X
        self.len = len(X)
        self.w = w
        self.s = s
        print(self.len)
        
        super().__init__()
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.X[index]