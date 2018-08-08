import torch.nn as nn


class Decoder60(nn.Module):

    def __init__(self, latent_size, intermediate_size):
        super(Decoder60, self).__init__()
        self.fc1 = nn.Linear(latent_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, 50176)
        self.deconv1 = nn.ConvTranspose2d(224, 224, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(224, 224, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(224, 224, kernel_size=2, stride=2, padding=0)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h3 = self.relu(self.fc1(z))
        out = self.relu(self.fc2(h3))
        print(f'fc output: {out.size()}')
        out = out.view(out.size(0), 1, 224, 224)
        print(f'fc out reshaped: {out.size()}')
        out = self.relu(self.deconv1(out))
        print(f'conv.T 1 out reshaped: {out.size()}')
        out = self.relu(self.deconv2(out))
        print(f'conv.T 2 out reshaped: {out.size()}')
        out = self.relu(self.deconv3(out))
        print(f'conv.T 3 out reshaped: {out.size()}')
        return out


class LinearDecoder(nn.Module):

    def __init__(self, latent_size, intermediate_size): #200704
        super(LinearDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, 3*224*224)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h3 = self.relu(self.fc1(z))
        out = self.relu(self.fc2(h3))
        out = out.view(z.size(0), 3, 224, 224)
        return out
