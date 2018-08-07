import torch.nn as nn
from torchvision.models import DenseNet
from collections import OrderedDict


class LungXnet(nn.Module):
    """
    Parameters:
    ----------
    * `growth_rate` [int, default=32]
        How many filters to add each layer (`k` in paper)

    * `block_config` [tuple of ints, default=(6, 12, 24, 16)]
        How many layers in each pooling block

    * `num_init_features` [int, default=64]
        The number of filters to learn in the first convolution layer

    * `bn_size` [int, default=4]
        Multiplicative factor for number of bottle neck layers
            - (i.e. bn_size * k features in the bottleneck layer)

    * `drop_rate` [float, default=0.2]
        Dropout rate after each dense layer

    * `num_classes` [int, default=14]
        Number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.2, num_classes=14):
        super(LungXnet, self).__init__()

        self.model = DenseNet(growth_rate=growth_rate, block_config=block_config,
                              num_init_features=num_init_features, bn_size=bn_size,
                              drop_rate=drop_rate, num_classes=num_classes)

        self.n_filters = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
            OrderedDict([('linear', nn.Linear(self.n_filters, num_classes)),
                         ('sigmoid', nn.Sigmoid())])
        )

    def forward(self, x):
        print(f'n_filters = {self.n_filters}')
        x = self.model(x)
        return x
