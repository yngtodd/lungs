from lungs.data.loaders import XRayLoaders
from lungs.models.minicnn import MiniCNN, MiniFeatures, HyperParameters
from lungs.parser import parse_args

def main():
    args = parse_args()

    loaders = XRayLoaders(data_dir=args.data, batch_size=args.batch_size)
    trainloader = loaders.train_loader(imagetxt=args.traintxt)

    hyperparameters = HyperParameters()
    features = MiniFeatures(hyperparameters)
    model = MiniCNN(features)

    print(f'Model: {model}')

    for batch_idx, (data, target) in enumerate(trainloader)


if __name__=='__main__':
    main()

