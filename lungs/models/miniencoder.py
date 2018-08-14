import torch.nn as nn


class Encoder(nn.Module):
    """
    To be transformed to mini cnn classifer.
    """
    def __init__(self):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        ) 

    def forward(self, x):
        return self.features(x)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, stride=3, padding=0),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=0),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MiniCNN(nn.Module):
    """
    CNN using the features from our autoencoder.
    """
    def __init__(self, encoder):
        super(MiniCNN, self).__init__()
        self.encoder = encoder 
        self.fc = nn.Linear(2, 14)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
