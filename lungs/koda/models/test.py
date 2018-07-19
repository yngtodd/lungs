import torch

from encoder import DenseNet121
from decoder import Decoder60, LinearDecoder


encoder = DenseNet121()
decoder1 = LinearDecoder(200, 300)
decoder2 = Decoder60(200, 300)

x = torch.randn(4, 3, 224, 224) 
latent = encoder(x)

out1 = decoder1(latent)
#out2 = decoder2(latent) 

print(f'Output1 has size {out1.size()}')
#print(f'Output1 has size {out2.size()}')
