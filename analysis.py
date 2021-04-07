import torch

from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from dataset.cifar_data_loaders import Cifar10Dataset
from training.utils.utils import load_model, to_device, get_default_device, batches_to_device

def analysis():
    model_dir = "E:/Git/MAVTransformer/models/trained_models/"

    cifar10_data = Cifar10Dataset(batch_size=10)
    device = get_default_device()

    mavTXL = load_model(model_dir + "vitFirst_XL.pt")
    mavTXL.eval()

    vit = mavTXL.ViT
    to_device(mavTXL, device)
    to_device(vit, device)

    t,v,te = cifar10_data.get_dataloaders()

    batches_to_device(t, device)

    # torch.multiprocessing.freeze_support()
    for images, _ in t:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(make_grid(vit(images), nrow=16).permute((1, 2, 0)))
        plt.show()
        break

if __name__ == '__main__':
    analysis()