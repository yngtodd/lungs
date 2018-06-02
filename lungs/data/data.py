import os
from PIL import Image

import torch
from torch.utils.data import Dataset


class ChestXrayDataSet(Dataset):
    """
    NIH Chest X-Ray 14 dataset.

    Parameters:
    ----------
    data_dir : str
        Path to image directory.

    imagetxt:
        Path to the file containing images with corresponding labels.

    transform : Pytorch transform
        Optional transform to be applied on a sample.
    """
    def __init__(self, data_dir, imagetxt, transform=None):
        image_names = []
        labels = []
        with open(imagetxt, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names[:4096]
        self.labels = labels[:4096]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        """Get next image and label"""
        img = self.image_names[index]
        img = Image.open(img).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(label)
