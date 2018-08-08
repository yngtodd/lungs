import torch.nn as nn


class HyperParameters:

    def __init__(self, nfilters1, nfilters2, nfilters3,
                 kernel1=3, kernel2=3, kernel3=3):
        self.nfilters1 = nfilters1
        self.nfilters2 = nfilters2
        self.nfilters3 = nfilters3
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3

    def __repr__(self):
        return f'Model hyperparameters for MiniCNN.'


class MiniFeatures(nn.Module):

    def __init__(self, hyperparameters):
        super(MiniFeatures, self).__init__()
        self.hyperparameters = hyperparameters

        self.features = nn.Sequential(
            nn.Conv2d(3, self.nfilters1, self.kernel1),
            nn.ReLU(),
            nn.Conv2d(50, self.nfilters2, self.kernel2),
            nn.ReLU(),
            nn.Conv2d(100, self.nfilters3, self.kernel3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)


class MiniCNN(nn.Module):

    def __init__(self, features):
        super(MiniCNN, self).__init__()
        self.features = features
        self.fc1 = nn.Linear(200, 500)
        self.fc2 = nn.Linear(500, 14)

    def forward(self, x):
        x = self.features(x)
        x = self.fc1(x)
        x = self.fc2(x)
