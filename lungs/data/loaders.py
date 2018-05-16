from torchvision import transforms
from torch.utils.data import DataLoader

from lungs.data.data import ChestXrayDataSet


_image_list_files = {'train':'./chestX-ray14/labels/train_list.txt',
                    'test':'./chestX-ray14/labels/test_list.txt',
                    'val':'./chestX-ray14/labels/val_list.txt'}

def get_loaders(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    datatransforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
    }

    image_datasets = {x:ChestXrayDataSet(data_dir=args.data,
                                        image_list_file=_image_list_files[x],
                                        transform=datatransforms[x])
                     for x in ['train','test','val']}

    dataloaders = {x:DataLoader(dataset=image_datasets[x], batch_size=args.batch,
                              shuffle=False if x =='test' else True, num_workers=args.workers, pin_memory=True)
                   for x in ['train','test','val']}

    return dataloaders
