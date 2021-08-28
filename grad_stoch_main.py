"""

"""


# from models.skp_Transformer import SKP_Transformer
from models.skp_vit import StochViT

from training.training_manager import train_model
from dataset.cifar_data_loaders import Cifar10Dataset
from training.utils.utils import save_model, load_model, load_checkpoint
from training.metrics.metrics import print_accuracy_per_class, print_accuracy, count_model_parameters
import copy
import time

from pytorch_pretrained_vit import ViT




if __name__ == '__main__':
    # TODO Add as config
    project_name = "SKP_Transformer"
    model_dir = f"E:/Git/{project_name}/models/trained_models/"
    model_name = ""
    checkpoint_dir = f"E:/Git/{project_name}/training/checkpoints/checkpoint_{model_name}.pt"
    batch_size = 15
    epochs = 100
    cifar10_data = Cifar10Dataset(batch_size=batch_size)
    classes = cifar10_data.classes

    """
    M A I N
    """

    # Breakpoint before/after .backward() in training manager to inspect grads in debug mode.
    gradTestStochViT = StochViT(num_classes=10, no_of_imgs_for_training=50, num_layers=1) #, patches=4, num_classes=10, dim=64)
    print(f"Parameters {count_model_parameters(gradTestStochViT, False)}")
    start_time = time.time()
    save_model(train_model(epochs, gradTestStochViT, "gradTestStochViT", cifar10_data, batch_size, model_dir, with_indexes=True), "gradTestStochViT", model_dir)
    gradTestStochViT = load_model(f"E:/Git/SKP_Transformer/models/trained_models/gradTestStochViT.pt")
    print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    print_accuracy_per_class(gradTestStochViT, classes, batch_size, cifar10_data.test_loader)
    print_accuracy(gradTestStochViT, classes, batch_size, cifar10_data.test_loader)

 # TODO Update training configs with and without ids
 #   



