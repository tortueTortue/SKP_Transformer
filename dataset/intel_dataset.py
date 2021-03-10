# https://www.kaggle.com/puneet6060/intel-image-classification

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
import torch.utils.data as data

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

def load_intel_scene_data(batch=64, root_path="../images"):
    train_data = ImageFolder(root=root_path, transform=None)
    train_data_loader = DataLoader(train_data, batch_size=batch, shuffle=True,  num_workers=4)

    test_data = ImageFolder(root=root_path, transform=None)
    test_data_loader  = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=4)

    return train_data_loader, test_data_loader


class IntelSceneDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_folder, root_dir, transform=None):
        """
        Args:
            image_folder (string): Path to the folder with annotations.

        """
        pass

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        return sample