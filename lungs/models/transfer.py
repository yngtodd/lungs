import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class TransferNet(nn.Module):

    def __init__(self, original_model, num_classes=14):
        super(TransferNet, self).__init__()
        self.model = nn.Sequential(*list(original_model.children())[:-1])
        self.model.classifier = nn.Sequential(
          OrderedDict([('linear', nn.Linear(7, num_classes)),
          ('sigmoid', nn.Sigmoid())])
        )

    def forward(self, x):
        x = self.model(x)
        return x
