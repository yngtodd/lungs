from torchvision import transforms
from torch.utils.data import DataLoader

from lungs.data.data import ChestXrayDataSet


class XRayLoaders:
    """
    Data loaders and transforms for Xray14 Data

    Parameters:
    ----------
    data_dir : str
        Path to the data base directory

    batch_size : int
        Batch size for data loaders.

    DataSet : Pytorch DataSet class, default: ChestXrayDataSet
        Handles reading of the data.

    pin_memory : bool, default: True

    num_workers : int, default: 4
        Number of threads to read in the data.

    train_transform : func, Optional
        Custom Pytorch transforms to be passed to the train set.

    val_transform : func, Optional
        Custom Pytorch transforms to be passed to the validation set.

    test_transform : func, Optional
        Custom Pytorch transforms to be passed to the test set.
    """

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_default = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda
        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda
        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])

    val_default = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda
        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda
        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])

    test_default = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda
        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda
        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])

    def __init__(self, data_dir, batch_size,
                 DataSet=ChestXrayDataSet, pin_memory=True, num_workers=4,
                 train_transform=None, val_transform=None, test_transform=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.DataSet = DataSet
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def train_loader(self, imagetxt, shuffle=True, transform=True):
        """
        Create trainloader with options for data transforms

        Parameters:
        ----------
        imagetxt : str
            Path to the train image file list. Contains image names and labels.

        shuffle : bool, default: True
            Whether to shuffle the data.

        Transform : bool, default: True
            Whether to transform the data. A few options here:
            - If False, no data transformations are made
            - If True and a train_transformer is given, train_transformer is used.
            - the class is not given a train_transformer, a
              default transform is used.
        """
        if not transform:
            # Instantiate the dataset with
            dataset = self.DataSet(
              data_dir=self.data_dir,
              imagetxt=imagetxt
             )
        elif self.train_transform is not None:
            dataset = self.DataSet(
              data_dir=self.data_dir,
              imagetxt=imagetxt,
              transform=self.train_transform
            )
        else:
            dataset = self.DataSet(
              data_dir=self.data_dir,
              imagetxt=imagetxt,
              transform=XRayLoaders.train_default
            )

        # Create data loader
        loader = DataLoader(
          dataset=dataset, batch_size=self.batch_size, shuffle=shuffle,
          num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        return loader

    def val_loader(self, imagetxt, shuffle=True, transform=True):
        """
        Create valloader with options for data transforms

        Parameters:
        ----------
        imagetxt : str
            Path to the train image file list. Contains image names and labels.

        shuffle : bool, default: True
            Whether to shuffle the data.

        Transform : bool, default: True
            Whether to transform the data. A few options here:
            - If False, no data transformations are made
            - If True and a val_transformer is given, train_transformer is used.
            - the class is not given a val_transformer, a
              default transform is used.
        """
        if not transform:
            # Instantiate the dataset with
            dataset = self.DataSet(
              data_dir=self.data_dir,
              imagetxt=imagetxt
             )
        elif self.val_transform is not None:
            dataset = self.DataSet(
              data_dir=self.data_dir,
              imagetxt=imagetxt,
              transform=self.val_transform
            )
        else:
            dataset = self.DataSet(
              data_dir=self.data_dir,
              imagetxt=imagetxt,
              transform=XRayLoaders.val_default
            )

        # Create data loader
        loader = DataLoader(
          dataset=dataset, batch_size=self.batch_size, shuffle=shuffle,
          num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        return loader

    def test_loader(self, imagetxt, shuffle=True, transform=True):
        """
        Create testloader with options for data transforms

        Parameters:
        ----------
        imagetxt : str
            Path to the train image file list. Contains image names and labels.

        shuffle : bool, default: True
            Whether to shuffle the data.

        Transform : bool, default: True
            Whether to transform the data. A few options here:
            - If False, no data transformations are made
            - If True and a test_transformer is given, train_transformer is used.
            - the class is not given a test_transformer, a
              default transform is used.
        """
        if not transform:
            # Instantiate the dataset with
            dataset = self.DataSet(
              data_dir=self.data_dir,
              imagetxt=imagetxt
             )
        elif self.test_transform is not None:
            dataset = self.DataSet(
              data_dir=self.data_dir,
              imagetxt=imagetxt,
              transform=self.test_transform
            )
        else:
            dataset = self.DataSet(
              data_dir=self.data_dir,
              imagetxt=imagetxt,
              transform=XRayLoaders.test_default
            )

        # Create data loader
        loader = DataLoader(
          dataset=dataset, batch_size=self.batch_size, shuffle=shuffle,
          num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        return loader
