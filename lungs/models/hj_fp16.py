import torch.nn as nn
from torch.autograd import Variable
import torch 
import torch.optim as optim
'''
Replicating HJ code which achieved 40TF on keras + tf 
'''
class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0), -1)

class hj_fp16(nn.Module):
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
    def __init__(self,num_layers, output_dim=14):
        super(hj_fp16, self).__init__()
        self.conv1_block = nn.Sequential(nn.Conv2d(1,num_layers,kernel_size=8,padding=0,stride=1),nn.ReLU(),nn.AdaptiveMaxPool2d(128),nn.Dropout2d(0.25))
        self.conv2_block = nn.Sequential(nn.Conv2d(num_layers,2*num_layers,kernel_size=8,padding=0,stride=1),nn.ReLU(),nn.AdaptiveMaxPool2d(64),nn.Dropout2d(0.25))
        self.conv3_block = nn.Sequential(nn.Conv2d(2*num_layers,4*num_layers,kernel_size=8,padding=0,stride=1),nn.ReLU(),nn.AdaptiveMaxPool2d(32),nn.Dropout(0.25))
        self.conv4_block = nn.Sequential(nn.Conv2d(4*num_layers,8*num_layers,kernel_size=8,padding=0,stride=1),nn.ReLU(),nn.AdaptiveMaxPool2d(16),nn.Dropout(0.25))
        self.conv5_block = nn.Sequential(nn.Conv2d(8*num_layers,16*num_layers,kernel_size=8,padding=0,stride=1),nn.ReLU(),nn.AdaptiveMaxPool2d(8),nn.Dropout(0.25))
        self.conv6_block = nn.Sequential(nn.Conv2d(16*num_layers,32*num_layers,kernel_size=8,padding=0,stride=1),nn.ReLU(),nn.AdaptiveMaxPool2d(16))
        self.fc1 = nn.Sequential(nn.Linear(524288,128),nn.ReLU(),nn.Dropout2d(0.5))
        self.fc2 = nn.Sequential(nn.Linear(128,14))
    '''
        self.model = torch.nn.Sequential()
        self.model.add_module("conv_1", torch.nn.Conv2d(1,num_layers,kernel_size=(8,8),stride=1,padding=0)) #1024
        self.model.add_module("maxpool_1", torch.nn.AdaptiveMaxPool2d(512))
        self.model.add_module("relu_1", torch.nn.ReLU())
        self.model.add_module("conv_2", torch.nn.Conv2d(num_layers,2*num_layers,kernel_size=(8,8),stride=1,padding=0)) #512
        self.model.add_module("maxpool_2", torch.nn.AdaptiveMaxPool2d(256))
        self.model.add_module("relu_2", torch.nn.ReLU())
        self.model.add_module("conv_3", torch.nn.Conv2d(2*num_layers,4*num_layers,kernel_size=(8,8),stride=1,padding=0)) #256
        self.model.add_module("maxpool_3", torch.nn.AdaptiveMaxPool2d(128))
        self.model.add_module("relu_3", torch.nn.ReLU())
        self.model.add_module("conv_4", torch.nn.Conv2d(4*num_layers,8*num_layers,kernel_size=(8,8),stride=1,padding=0)) #128
        self.model.add_module("maxpool_4", torch.nn.AdaptiveMaxPool2d(64))
        self.model.add_module("relu_4", torch.nn.ReLU())
        self.model.add_module("conv_5", torch.nn.Conv2d(8*num_layers,16*num_layers,kernel_size=(8,8),stride=1,padding=0)) #64
        self.model.add_module("maxpool_5", torch.nn.AdaptiveMaxPool2d(32))
        self.model.add_module("relu_5", torch.nn.ReLU())
        self.model.add_module("conv_6", torch.nn.Conv2d(16*num_layers,32*num_layers,kernel_size=(8,8),stride=1,padding=0)) #32
        self.model.add_module("maxpool_6", torch.nn.AdaptiveMaxPool2d(16))
        self.model.add_module("relu_6", torch.nn.ReLU())
        self.model.add_module("conv_7", torch.nn.Conv2d(32*num_layers,64*num_layers,kernel_size=(8,8),stride=1,padding=0)) #16
        self.model.add_module("maxpool_7", torch.nn.AdaptiveMaxPool2d(8))
        self.model.add_module("relu_7", torch.nn.ReLU())
        self.model.add_module("full_connected", torch.nn.Linear(8,output_dim))
        self.squash = nn.Sigmoid()
    '''
    def forward(self, x):
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x  = self.conv3_block(x)
        x  = self.conv4_block(x)
        x = self.conv5_block(x)
        x = self.conv6_block(x)
        x = x.view(x.size(0), -1)
        #print("x.size",x.size())
        x = self.fc1(x)
        #print("x.size",x.size())
        return self.fc2(x)


