import json
from pyclbr import Function
import sys

from training.training_manager import train_and_test_model
from dataset.cifar_data_loaders import Cifar10Dataset
from models.transformer import StochViT, end_of_iteration_stoch_gaussian_ViT


if __name__ == '__main__':
    if len(sys.argv) > 1:
        f = open(f'configs/{sys.argv[1]}.json')
    else:
        f = open('configs/gaussian_configs.json')
        # f = open('configs/default_configs.json')

    config = json.load(f)
    f.close()

    dataset = Cifar10Dataset(batch_size=config['hyperparameters']['batch_size'],
                             subset=True,
                             subset_size=1000,
                             test_subset_size=500)
    classes = dataset.classes

    end_of_iteration_routine: Function = None

    if config['attention_type'] == 'Gaussian':
        model = StochViT(num_classes=10,
                         no_of_imgs_for_training=50000,
                         image_size=256,
                         sigma=1,
                         num_layers=1,
                         classifier="",
                         attention_type=config['attention_type'])
        end_of_epoch_routine = end_of_iteration_stoch_gaussian_ViT(config['hyperparameters']['attention_learning_rate'])
    
    elif config['attention_type'] == 'SamplingNetwork':
        model = StochViT(num_classes=10,
                         image_size=256,
                         num_layers=1,
                         classifier="",
                         attention_type=config['attention_type'])
    
    elif config['attention_type'] == 'Normal':
        model = StochViT(num_classes=10,
                         image_size=256,
                         num_layers=1,
                         attention_type=config['attention_type'])
    
    elif config['attention_type'] == 'None':
        model = StochViT(num_classes=10,
                         image_size=256,
                         num_layers=1,
                         attention_type=config['attention_type'])

    train_and_test_model(classes=classes,
                         model=model,
                         dataset=dataset,
                         end_of_epoch_routine=None,
                         end_of_iteration_routine = end_of_iteration_routine,
                         config=config)
