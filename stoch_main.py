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


    stochViT = StochViT(num_classes=10, no_of_imgs_for_training=50000) #, patches=4, num_classes=10, dim=64)
    print(f"Parameters {count_model_parameters(stochViT, False)}")
    start_time = time.time()
    save_model(train_model(epochs, stochViT, "stochViT", cifar10_data, batch_size, model_dir, with_indexes=True), "stochViT", model_dir)
    stochViT = load_model(f"E:/Git/SKP_Transformer/models/trained_models/stochViT.pt")
    print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    print_accuracy_per_class(stochViT, classes, batch_size, cifar10_data.test_loader)
    print_accuracy(stochViT, classes, batch_size, cifar10_data.test_loader)

 # TODO Update training configs with and without ids
 #   



