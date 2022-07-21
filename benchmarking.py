import copy
import json
from pyclbr import Function
import sys

from training.training_manager import train_and_test_model
from dataset.cifar_data_loaders import Cifar10Dataset
from models.transformer import StochViT, end_of_iteration_stoch_gaussian_ViT

BENCHMARK_TIME = {
                  'gen_qkv': 0,
                  'grid_sample_time' : 0 ,
                  'att_time' : 0,
                  'whole_time' : 0
                }

BENCHMARK_MAP_IN_S = {
                      'Gaussian' : copy.deepcopy(BENCHMARK_TIME),
                      'SampNet' : copy.deepcopy(BENCHMARK_TIME),
                      'ViT' : copy.deepcopy(BENCHMARK_TIME)
                     }

if __name__ == '__main__':
    if len(sys.argv) > 1:
        f = open(f'configs/{sys.argv[1]}.json')
    else:
        f = open('configs/gaussian_configs.json')

    config = json.load(f)
    f.close()

    config['hyperparameters']['batch_size'] = 128
    config['hyperparameters']['epoch'] = 1

    dataset = Cifar10Dataset(batch_size=config['hyperparameters']['batch_size'],
                             subset=True,
                             subset_size=1000,
                             test_subset_size=500)
    classes = dataset.classes

    end_of_iteration_routine: Function = None


    stoch_g = StochViT(num_classes=10,
                        no_of_imgs_for_training=50000,
                        image_size=256,
                        sigma=1,
                        num_layers=config['hyperparameters']['num_layers'],
                        classifier="",
                        attention_type='Gaussian')
    end_of_iteration_routine = end_of_iteration_stoch_gaussian_ViT(config['hyperparameters']['attention_learning_rate'])


    stoch_s = StochViT(num_classes=10,
                        image_size=256,
                        num_layers=config['hyperparameters']['num_layers'],
                        classifier="",
                        attention_type='SamplingNetwork')


    ViT = StochViT(num_classes=10,
                        image_size=256,
                        num_layers=config['hyperparameters']['num_layers'],
                        attention_type='Normal')

    print("Gaussian")
    train_and_test_model(classes=classes,
                         model=stoch_g,
                         dataset=dataset,
                         end_of_epoch_routine=None,
                         end_of_iteration_routine = end_of_iteration_routine,
                         config=config)

    print("SamplingNet")
    train_and_test_model(classes=classes,
                         model=stoch_s,
                         dataset=dataset,
                         end_of_epoch_routine=None,
                         end_of_iteration_routine = end_of_iteration_routine,
                         config=config)

    print("ViT")
    train_and_test_model(classes=classes,
                         model=ViT,
                         dataset=dataset,
                         end_of_epoch_routine=None,
                         end_of_iteration_routine = end_of_iteration_routine,
                         config=config)