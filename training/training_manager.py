"""
Here are the training methods.
"""
from distutils.log import debug
from pyclbr import Function
from statistics import mode
from models.skp_vit import StochViT
from models.skp_Transformer import SKP_Transformer
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD
import torch
import time

from training.utils.utils import batches_to_device, get_default_device, to_device, save_checkpoints
from training.metrics.metrics import accuracy, print_accuracy_per_class, print_accuracy, count_model_parameters
from training.utils.logger import start_training_logging
from datetime import datetime
from training.utils.utils import save_model, load_model

from optimizer.sam.sam import SAM


def end_of_epoch_routine(model=None):
    pass

def validation_step(model, batch, with_indices: bool = False):
    images, labels, i = batch
    if with_indices:
        images, labels, i = images.cuda(), labels.cuda(), i.cuda()
        out = model.forward(images, i)
    else :
        images, labels = images.cuda(), labels.cuda()
        out = model.forward(images)
    cross_entropy = CrossEntropyLoss()                  
    val_loss = cross_entropy(out, labels)

    return {'val_loss': val_loss.detach(), 'val_acc': accuracy(out, labels)}

def evaluate(model: Module, val_set: DataLoader, epoch: int, with_indices: bool = False):
    outputs = [validation_step(model, batch, with_indices) for batch in val_set]

    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'epoch' : epoch}

def train(epochs_no: int,
          model: Module,
          train_set: DataLoader,
          val_set: DataLoader, model_dir,
          logger, lr, with_sam_opt: bool = False,
          with_indices: bool = False, 
          end_of_epoch_routine: Function = end_of_epoch_routine,
          debug: bool = False):
    loss = CrossEntropyLoss()
    history = []

    if with_sam_opt:
        optimizer = SAM(model.parameters(), SGD(), lr=0.0001, momentum=0.9)
    else:
        optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)

    if debug:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f'runs/debug/{model.__class__.__name__}-{datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}')
        im, _, ids = next(iter(train_set))
        # writer.add_graph(model, (im.cuda(), ids.cuda()), verbose=True)
        import torchvision
        grid = torchvision.utils.make_grid(im)
        writer.add_image('images', grid, 0)
        writer.close()

    for epoch in range(epochs_no):
        start_time = time.time()
        """  Training Phase """ 
        for batch_index, batch in enumerate(train_set):
            optimizer.zero_grad()
            inputs, labels, indexes = batch
            if with_indices:
                inputs, labels, indexes = inputs.cuda(), labels.cuda(), indexes.cuda()
                curr_loss = loss(model(inputs, indexes), labels)
            else:
                inputs, labels = inputs.cuda(), labels.cuda()
                curr_loss = loss(model(inputs), labels)

            curr_loss.backward()

            if with_sam_opt:
                optimizer.first_step(zero_grad=True)

                # Second pass
                loss(model.forward(inputs), labels).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            if end_of_epoch_routine:
                end_of_epoch_routine(model=model)


        """ Validation Phase """
        result = evaluate(model, val_set, epoch, with_indices)
        print(result)
        print(f"Training time for {epoch} : {time.time() - start_time}")
        history.append(result)
        writer.add_scalar("Loss/val", result['val_loss'], epoch)
        writer.add_scalar("Accuracy/val", result['val_acc'], epoch)

        logger.info(str(result))
        if epoch % 10 == 0 :
            save_checkpoints(epoch, model, optimizer, loss, model_dir + f"checkpoint_{epoch}_{type(model).__name__}.pt")
    writer.flush()

    return history

def train_model(epochs_no: int,
                model_to_train: Module,
                model_name: str,
                dataset: Dataset,
                batch_size: int,
                model_dir: str,
                with_sam_opt: bool=False,
                with_indices: bool=False,
                end_of_epoch_routine: Function=None,
                learning_rate = 0.0001):
    model_to_train.train()
    device = get_default_device()
    logger = start_training_logging(model_name)

    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size)

    batches_to_device(train_loader, device)
    batches_to_device(val_loader, device)
    batches_to_device(test_loader, device)

    model = to_device(model_to_train, device)

    print(f"Parameter count {count_model_parameters(model_to_train, False)}")
    start_time = time.time()
    history = train(epochs_no, model, train_loader, val_loader, model_dir, logger, learning_rate, with_sam_opt=with_sam_opt, with_indices=with_indices, end_of_epoch_routine=end_of_epoch_routine)
    print(f"Training time for {epochs_no} epochs : {time.time() - start_time}")

    return model


def train_and_test_model(classes: list,
                         model: Module,
                         dataset: Dataset,
                         epochs: int,
                         config,
                         end_of_epoch_routine: Function = None):
    model_name=config['model_name']
    model_dir=config['model_dir']
    batch_size=config['hyperparameters']['batch_size']
    epochs=config['hyperparameters']['epochs']
    with_indices=config['dataset_with_indices']
    with_sam_opt=config['with_sam_opt']
    debug=config['debug']
    learning_rate=config['hyperparameters']['learning_rate']

    print(f"*********************************Training {model_name}*********************************")
    print(f"Parameters {count_model_parameters(model, False)}")
    start_time = time.time()
    trained_model = train_model(epochs, model, "model_name", dataset, batch_size,
                                model_dir, with_indices=with_indices, with_sam_opt=with_sam_opt,
                                learning_rate=learning_rate, debug=debug, end_of_epoch_routine=end_of_epoch_routine)
    save_model(trained_model, "model_name", model_dir)
    model = load_model(f"{model_dir}/{model_name}.pt")
    print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    print(f"*********************************Testing  {model_name}*********************************")
    print_accuracy_per_class(model, classes, batch_size, dataset.test_loader)
    print_accuracy(model, classes, batch_size, dataset.test_loader)