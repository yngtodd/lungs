import torch

from encoder import DenseNet121
from decoder import Decoder60, LinearDecoder


def main():
    encoder = DenseNet121()
    decoder1 = LinearDecoder(200, 300)
    decoder2 = Decoder60(200, 300)

    x = torch.randn(4, 3, 224, 224)
    latent = encoder(x)
    out1 = decoder1(latent)

    print(f'Original image dimension: {x.size()}')
    print(f'Latent space is of size {latent.size()}')
    print(f'Output1 has size {out1.size()}')


if __name__=='__main__':
    main()
