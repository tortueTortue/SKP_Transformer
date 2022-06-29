"""

"""


# from models.skp_Transformer import SKP_Transformer
from models.skp_vit import StochViT

from training.training_manager import train_model
from dataset.cifar_data_loaders import Cifar10Dataset
from training.utils.utils import save_model, load_model, load_checkpoint
from training.metrics.metrics import print_accuracy_per_class, print_accuracy, count_model_parameters
import copy

from pytorch_pretrained_vit import ViT




if __name__ == '__main__':
    project_name = "SKP_Transformer"
    # model_dir = f"/home/tortue/projects/def-mpederso/tortue/beluga/{project_name}/models/trained_models/"
    model_dir = f"E:/Git/{project_name}/models/trained_models/"
    model_name = ""
    # checkpoint_dir = f"/home/tortue/projects/def-mpederso/tortue/beluga/{project_name}/training/checkpoints/checkpoint_{model_name}.pt"
    checkpoint_dir = f"E:/Git/{project_name}/training/checkpoints/checkpoint_{model_name}.pt"
    batch_size = 2
    epochs = 100
    # cifar10_data = Cifar10Dataset(batch_size=batch_size)
    cifar10_debug_data = Cifar10Dataset(batch_size=batch_size, subset=True, subset_size=1000, test_subset_size=500)
    classes = cifar10_debug_data.classes

    """
    M A I N
    """

    # normalViT = ViT('B_16_imagenet1k', image_size=256, pretrained=False, num_classes=10)
    # train_model(epochs, normalViT, "normalViT", cifar10_data, batch_size, model_dir, with_indexes=False)

    stochViT = StochViT(num_classes=10, no_of_imgs_for_training=50000, image_size=256, sigma=1, num_layers=1) #, patches=4, num_classes=10, dim=64)
    # stochViT = StochViT(num_classes=10, no_of_imgs_for_training=50000, image_size=256, sigma=1) #, patches=4, num_classes=10, dim=64)
    save_model(train_model(epochs, stochViT, "stochViT", cifar10_debug_data, batch_size, model_dir, with_indexes=True), "stochViT", model_dir)
    stochViT = load_model(f"E:/Git/SKP_Transformer/models/trained_models/stochViT.pt")
    print_accuracy_per_class(stochViT, classes, batch_size, cifar10_debug_data.test_loader)
    print_accuracy(stochViT, classes, batch_size, cifar10_debug_data.test_loader)

    stochViTSigma01 = StochViT(num_classes=10, no_of_imgs_for_training=50000, image_size=256, sigma=0.1)
    stochViTSigma05 = StochViT(num_classes=10, no_of_imgs_for_training=50000, image_size=256, sigma=0.5)


 #   



