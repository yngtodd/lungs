import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DenseNet121(nn.Module):

    def __init__(self, latentspace=200, pretrained=False):
        super(DenseNet121, self).__init__()
        original_model = models.densenet121(pretrained=pretrained)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = (nn.Linear(1024, latentspace))

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.avg_pool2d(f, kernel_size=7).view(f.size(0), -1)
        out = self.classifier(f)
        return out 


class DenseNet60(nn.Module):

    def __init__(self, block_config=(3, 6, 12, 8), latentspace=200):
        super(DenseNet60, self).__init__()
        original_model = models.DenseNet(block_config=block_config)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = (nn.Linear(516, latentspace))

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.avg_pool2d(f, kernel_size=7).view(f.size(0), -1)
        out = self.classifier(f)
        return out 
