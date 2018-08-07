import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class TransferNet(nn.Module):

    def __init__(self, original_model, num_classes=14):
        super(TransferNet, self).__init__()
        self.model = nn.Sequential(*list(original_model.children())[:-1])
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        x = self.fc(x)
        return x
