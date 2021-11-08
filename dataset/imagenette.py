from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from typing import Any, Callable, Optional, Tuple

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ImageFolderWithIdx(ImageFolder):

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        image, target = super().__getitem__(index)
        return image, target, index

class ImageNetteDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, batch_size=128):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_path = r"data/imagenette2/imagenette2/"
        train_path = self.root_path + "train/"
        test_path = self.root_path + "test/"

        transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

        data = ImageFolderWithIdx(root=train_path, transform=transform)
        test_data = ImageFolderWithIdx(root=test_path, transform=transform)

        self.labels = data.classes
        self.classes = ['tench', 'English springer', 'cassette player', 'chain saw', 'church',
                        'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

        torch.manual_seed(43)
        train_dataset, val_dataset = random_split(data, [int(round(len(data) * 0.85)),
                                                            int(round(len(data) * 0.15))])

        self.train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        self.val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        self.test_data_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    def __len__(self):
        return len(self.data)

    def get_dataloaders(self, batch=10):
        return self.train_data_loader, self.val_data_loader, self.test_data_loader
