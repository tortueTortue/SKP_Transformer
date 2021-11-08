"""

"""


# from models.skp_Transformer import SKP_Transformer
from models.normal_transformer import Transformer
from models.na_transformer import Transformer as NATransformer

from training.training_manager import train_model
from dataset.cifar_data_loaders import Cifar10Dataset
from dataset.imagenette import ImageNetteDataset
from training.utils.utils import save_model, load_model, load_checkpoint
from training.metrics.metrics import print_accuracy_per_class, print_accuracy, count_model_parameters
import copy
import time

from pytorch_pretrained_vit import ViT


if __name__ == '__main__':
    # TODO Add as config
    project_name = "SKP_Transformer"
    model_dir = f"/home/tortue/projects/def-mpederso/tortue/beluga/{project_name}/models/trained_models/"
    # model_dir = f"E:/Git/{project_name}/models/trained_models/"
    model_name = ""
    checkpoint_dir = f"/home/tortue/projects/def-mpederso/tortue/beluga/{project_name}/training/checkpoints/checkpoint_{model_name}.pt"
    # checkpoint_dir = f"E:/Git/{project_name}/training/checkpoints/checkpoint_{model_name}.pt"
    batch_size = 15
    epochs = 100
    imagenette_data = ImageNetteDataset(batch_size=batch_size)
    classes = imagenette_data.classes

    """
    M A I N
    """

    #TODO Try lr 1e-4 + momentum .99, diff dropout 0.5

    # # SKP TRANSFORMER 100 epochs
    # skp = SKP_Transformer(8, 50000, len(classes))
    # print(f"Parameters {count_model_parameters(skp, False)}")
    # start_time = time.time()
    # save_model(train_model(epochs, skp, "SKP", imagenette_data, batch_size, model_dir), "SKP", model_dir)
    # print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    # print_accuracy_per_class(skp, classes, batch_size, imagenette_data.test_loader)
    # print_accuracy(skp, classes, batch_size, imagenette_data.test_loader)
    #TODO Start over all tests
    # # TRANSFORMER 100 epochs
    # transformer = Transformer(2, 50000, len(classes))
    # print(f"Parameters {count_model_parameters(transformer, False)}")
    # start_time = time.time()
    # save_model(train_model(epochs, transformer, "Transformer", imagenette_data, batch_size, model_dir), "Transformer", model_dir)
    # transformer = load_model(f"E:/Git/SKP_Transformer/models/trained_models/Transformer.pt")
    # print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    # print_accuracy_per_class(transformer, classes, batch_size, imagenette_data.test_loader)
    # print_accuracy(transformer, classes, batch_size, imagenette_data.test_loader)
   
    # transformer = Transformer(4, 50000, len(classes))
    # print(f"Parameters {count_model_parameters(transformer, False)}")
    # start_time = time.time()
    # save_model(train_model(epochs, transformer, "Transformer", imagenette_data, batch_size, model_dir), "Transformer", model_dir)
    # transformer = load_model(f"E:/Git/SKP_Transformer/models/trained_models/Transformer.pt")
    # print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    # print_accuracy_per_class(transformer, classes, batch_size, imagenette_data.test_loader)
    # print_accuracy(transformer, classes, batch_size, imagenette_data.test_loader)
   
    # transformer = Transformer(8, 50000, len(classes))
    # print(f"Parameters {count_model_parameters(transformer, False)}")
    # start_time = time.time()
    # save_model(train_model(epochs, transformer, "Transformer", imagenette_data, batch_size, model_dir), "Transformer", model_dir)
    # transformer = load_model(f"E:/Git/SKP_Transformer/models/trained_models/Transformer.pt")
    # print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    # print_accuracy_per_class(transformer, classes, batch_size, imagenette_data.test_loader)
    # print_accuracy(transformer, classes, batch_size, imagenette_data.test_loader)
   

    # ViTPretrained = ViT('B_16_imagenet1k', pretrained=True, num_classes=10) #, patches=4, num_classes=10, dim=64)
    # print(f"Parameters {count_model_parameters(ViTPretrained, False)}")
    # start_time = time.time()
    # save_model(train_model(epochs, ViTPretrained, "ViTPretrained", imagenette_data, batch_size, model_dir), "ViTPretrained", model_dir)
    # ViTPretrained = load_model(f"E:/Git/SKP_Transformer/models/trained_models/ViTPretrained.pt")
    # print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    # print_accuracy_per_class(ViTPretrained, classes, batch_size, imagenette_data.test_loader)
    # print_accuracy(ViTPretrained, classes, batch_size, imagenette_data.test_loader)
   

    ViTFromScratch = ViT('B_16_imagenet1k', pretrained=False, num_classes=10) #, patches=4 dim=64)
    print(f"Parameters {count_model_parameters(ViTFromScratch, False)}")
    start_time = time.time()
    save_model(train_model(epochs, ViTFromScratch, "ViTFromScratch", imagenette_data, batch_size, model_dir), "ViTFromScratch", model_dir)
    ViTFromScratch = load_model(f"E:/Git/SKP_Transformer/models/trained_models/ViTFromScratch.pt")
    print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    print_accuracy_per_class(ViTFromScratch, classes, batch_size, imagenette_data.test_loader)
    print_accuracy(ViTFromScratch, classes, batch_size, imagenette_data.test_loader)

    # NA TRANSFORMER 100 epochs
    # model_name = "na_transformer"
    # na_transformer = NATransformer(8, 50000, len(classes))
    # print(f"Parameters {count_model_parameters(na_transformer, False)}")
    # start_time = time.time()
    # save_model(train_model(epochs, na_transformer, "na_transformer", imagenette_data, batch_size, model_dir), "na_transformer", model_dir)
    # na_transformer = load_model(f"E:/Git/SKP_Transformer/models/trained_models/na_transformer.pt")
    # print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    # print_accuracy_per_class(na_transformer, classes, batch_size, imagenette_data.test_loader)
    # print_accuracy(na_transformer, classes, batch_size, imagenette_data.test_loader)


