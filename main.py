import json
from training.training_manager import train_and_test_model
from dataset.cifar_data_loaders import Cifar10Dataset
from models.skp_vit import StochViT


if __name__ == '__main__':
    f = open('configs/default_configs.json')
    config = json.load(f)
    f.close()
    cifar10_debug_data = Cifar10Dataset(batch_size=config['hyperparameters']['batch_size'], subset=True, subset_size=1000, test_subset_size=500)
    classes = cifar10_debug_data.classes

    stochViT = StochViT(num_classes=10, no_of_imgs_for_training=50000, image_size=256, sigma=1, num_layers=1)

    train_and_test_model(classes=classes,
                         model=stochViT,
                         dataset=cifar10_debug_data,
                         end_of_epoch_routine=None,
                         config=config)