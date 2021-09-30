import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision import transforms
from typing import Any, Callable, Optional, Tuple



class ImageNetWithIndices(ImageNette):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        image, target = super().__getitem__(index)
        return image, target, index


class MinImageNetDataset:
    def __init__(self, batch_size=128): #TODO Add with indexes params and transform
        dataset = ImageNetWithIndices(root='data/', download=True, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]))
        test_dataset = ImageNetWithIndices(root='data/', train=False, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]))
        # dataset = CIFAR10WithIndices(root='data/', download=True, transform=ToTensor())
        # test_dataset = CIFAR10WithIndices(root='data/', train=False, transform=ToTensor())


        self.labels = dataset.classes
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']

        torch.manual_seed(43)
        train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * 0.85),
                                                            int(len(dataset) * 0.15)])

#         self.train_loader = DataLoader(
#             train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
#         self.val_loader = DataLoader(
#             val_dataset, batch_size, num_workers=4, pin_memory=True)
#         self.test_loader = DataLoader(
#             test_dataset, batch_size, num_workers=4, pin_memory=True)


        # TODO: Set num of workers to 0 to get a more meanigful error message
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size*2, num_workers=0, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size*2, num_workers=0, pin_memory=True)


        self.dataset = dataset

    def get_dataloaders(self, batch=10):
        return self.train_loader, self.val_loader, self.test_loader

    def __getitem__(self, id):
        return self.dataset.__getitem__(id), id