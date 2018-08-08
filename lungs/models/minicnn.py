import torch.nn as nn


class HyperParameters:

    def __init__(self, nfilters1=100, nfilters2=100,
                 nfilters3=100, kernel1=3, kernel2=3, kernel3=3):
        self.nfilters1 = nfilters1
        self.nfilters2 = nfilters2
        self.nfilters3 = nfilters3
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3

    def __repr__(self):
        return f'Model hyperparameters for MiniCNN.'


class MiniFeatures(nn.Module):

    def __init__(self, hparams):
        super(MiniFeatures, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, hparams.nfilters1, hparams.kernel1),
            nn.ReLU(),
            nn.Conv2d(50, hparams.nfilters2, hparams.kernel2),
            nn.ReLU(),
            nn.Conv2d(100, hparams.nfilters3, hparams.kernel3),
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
        print(f'features output: {x.size()}')
        x = self.fc1(x)
        print(f'fc1 output: {x.size()}')
        x = self.fc2(x)
        print(f'fc2 output: {x.size()}')
