"""

"""


from models.skp_Transformer import SKP_Transformer
from models.normal_transformer import Transformer

from training.training_manager import train_model
from dataset.cifar_data_loaders import Cifar10Dataset
from training.utils.utils import save_model, load_model, load_checkpoint
from training.metrics.metrics import print_accuracy_per_class, print_accuracy, count_model_parameters
import copy
import time

if __name__ == '__main__':
    # TODO Add as config
    project_name = "SKP_Transformer"
    model_dir = f"E:/Git/{project_name}/models/trained_models/"
    model_name = ""
    checkpoint_dir = f"E:/Git/{project_name}/training/checkpoints/checkpoint_{model_name}.pt"
    batch_size = 15
    epochs = 36
    cifar10_data = Cifar10Dataset(batch_size=batch_size)
    classes = cifar10_data.classes

    """
    M A I N
    """

    # SKP TRANSFORMER 100 epochs
    skp = SKP_Transformer(8, 50000, len(classes))
    print(f"Parameters {count_model_parameters(skp, False)}")
    start_time = time.time()
    save_model(train_model(epochs, skp, "SKP", cifar10_data, batch_size, model_dir), "SKP", model_dir)
    print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    print_accuracy_per_class(skp, classes, batch_size, cifar10_data.test_loader)
    print_accuracy(skp, classes, batch_size, cifar10_data.test_loader)

    # SKP TRANSFORMER 100 epochs
    transformer = Transformer(8, 50000, len(classes))
    print(f"Parameters {count_model_parameters(skp, False)}")
    start_time = time.time()
    save_model(train_model(epochs, transformer, "SKP", cifar10_data, batch_size, model_dir), "SKP", model_dir)
    print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    print_accuracy_per_class(transformer, classes, batch_size, cifar10_data.test_loader)
    print_accuracy(transformer, classes, batch_size, cifar10_data.test_loader)


