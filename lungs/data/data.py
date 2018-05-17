import os
import cv2

import torch
from torch.utils.data import Dataset


class ChestXrayDataSet(Dataset):
    """
    NIH Chest X-Ray 14 dataset.

    Parameters:
    ----------
    data_dir : str
        Path to image directory.

    image_list_file:
        Path to the file containing images with corresponding labels.

    transform : Pytorch transform
        Optional transform to be applied on a sample.
    """
    def __init__(self, data_dir, image_list_file, transform=None):
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        """Get next image and label"""
        img = self.image_names[index]
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(label)
