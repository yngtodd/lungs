import torch.nn as nn
import torch.autograd import Variable
import torch 
import optim

class fp16_LungXnet(nn.Module):
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
    def __init__(self,num_layers, output_dim):,
        super(fp16_LungXnet, self).__init__()
		self.model = torch.nn.Sequential()
		self.model.add_module("conv_1", torch.nn.Conv2D(1,num_layers,kernel_size=(8,8),stride=1,padding=0)) #1024
		self.model.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
		self.model.add_module("relu_1", torch.nn.ReLU())
		self.model.add_module("conv_2", torch.nn.Conv2D(num_layers,2*num_layers,kernel_size=(8,8),stride=1,padding=0)) #512
        self.model.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
        self.model.add_module("relu_2", torch.nn.ReLU()
		sself.model.add_module("conv_3", torch.nn.Conv2D(2*num_layers,4*num_layers,kernel_size=(8,8),stride=1,padding=0)) #256
        self.model.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=2))
        self.model.add_module("relu_3", torch.nn.ReLU())
		self.model.add_module("conv_4", torch.nn.Conv2D(4*num_layers,8*num_layers,kernel_size=(8,8),stride=1,padding=0)) #128
        self.model.add_module("maxpool_4", torch.nn.MaxPool2d(kernel_size=2))
        self.model.add_module("relu_4", torch.nn.ReLU())
	    self.model.add_module("conv_5", torch.nn.Conv2D(8*num_layers,16*num_layers,kernel_size=(8,8),stride=1,padding=0)) #64
        self.model.add_module("maxpool_5", torch.nn.MaxPool2d(kernel_size=2))
        self.model.add_module("relu_5", torch.nn.ReLU())
	    self.model.add_module("conv_6", torch.nn.Conv2D(16*num_layers,32*num_layers,kernel_size=(8,8),stride=1,padding=0)) #32
        self.model.add_module("maxpool_6", torch.nn.MaxPool2d(kernel_size=2))
        self.model.add_module("relu_6", torch.nn.ReLU())
   	    self.model.add_module("conv_7", torch.nn.Conv2D(32*num_layers,64*num_layers,kernel_size=(8,8),stride=1,padding=0)) #16
        self.model.add_module("maxpool_7", torch.nn.MaxPool2d(kernel_size=2))
        self.model.add_module("relu_7", torch.nn.ReLU())
	    self.model.add_module("conv_8", torch.nn.Conv2D(128*num_layers,256*num_layers,kernel_size=(8,8),stride=1,padding=0)) #8
        self.model.add_module("maxpool_8", torch.nn.MaxPool2d(kernel_size=2))
        self.model.add_module("relu_8", torch.nn.ReLU())
		self.model.add_module("full_connected", torch.nn.Linear(256*num_layers,output_dim))

	def forward(self,x):
		x = self.model.forward(x)
		return x










)


    def forward(self, x):
        x = self.model(x)
        return x
