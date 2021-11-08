"""
Here are the training methods.
"""
from models.skp_vit import StochViT
from models.skp_Transformer import SKP_Transformer
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD
import torch

from training.utils.utils import batches_to_device, get_default_device, to_device, save_checkpoints
from training.metrics.metrics import accuracy
from training.utils.logger import start_training_logging



from optimizer.sam.sam import SAM

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

# TODO : Add configs for this one
PATH = ""

def validation_step(model, batch, with_indexes: bool = False):
    images, labels, i = batch
    if with_indexes:
        images, labels, i = images.cuda(), labels.cuda(), i.cuda()
        out = model.forward(images, i)
    else :
        images, labels = images.cuda(), labels.cuda()
        out = model.forward(images)
    cross_entropy = CrossEntropyLoss()                  
    val_loss = cross_entropy(out, labels)

    return {'val_loss': val_loss.detach(), 'val_acc': accuracy(out, labels)}

def evaluate(model: Module, val_set: DataLoader, epoch: int, with_indexes: bool = False):
    outputs = [validation_step(model, batch, with_indexes) for batch in val_set]

    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'epoch' : epoch}

def train(epochs_no: int, model: Module, train_set: DataLoader, val_set: DataLoader, model_dir, logger, lr, with_sam_opt: bool = False, with_indexes: bool = False):
    loss = CrossEntropyLoss()
    history = []

    if with_sam_opt:
        optimizer = SAM(model.parameters(), SGD(), lr=0.0001, momentum=0.9)
    else:
        optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(epochs_no):

        """  Training Phase """ 
        for batch_index, batch in enumerate(train_set):
            optimizer.zero_grad()
            inputs, labels, indexes = batch
            if with_indexes:
                inputs, labels, indexes = inputs.cuda(), labels.cuda(), indexes.cuda()
                # TODO try with this forward call
                curr_loss = loss(model(inputs, indexes), labels) #--> model.forward(inputs, indexes)
            else:
                inputs, labels = inputs.cuda(), labels.cuda()
                curr_loss = loss(model(inputs), labels)

            curr_loss.backward()

            # TODO ADD PARAM
            # if True:
            #     model.compute_gradients(curr_loss, indexes)

            if with_sam_opt:
                optimizer.first_step(zero_grad=True)

                # Second pass
                loss(model.forward(inputs), labels).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            #TODO Remove self.avgs[img_id][0] and self.std_devs[img_id][1] from GPU

            # TODO: A voir
            if  with_indexes:
                model.propagate_attention(lr, indexes, None)


        """ Validation Phase """
        result = evaluate(model, val_set, epoch, with_indexes)
        print(result)
        history.append(result)
        #writer.add_scalar("Loss/val", result['val_loss'], epoch)
        #writer.add_scalar("Accuracy/val", result['val_acc'], epoch)
        logger.info(str(result))
        if epoch % 10 == 0 :
            save_checkpoints(epoch, model, optimizer, loss, model_dir + f"checkpoint_{epoch}_{type(model).__name__}.pt")
    #writer.flush()

    return history

def train_model(epochs_no: int, model_to_train: Module, model_name: str, dataset: Dataset, batch_size: int, model_dir: str, with_sam_opt=False, with_indexes=False):
    model_to_train.train()
    device = get_default_device()
    logger = start_training_logging(model_name)

    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size)

    batches_to_device(train_loader, device)
    batches_to_device(val_loader, device)
    batches_to_device(test_loader, device)

    # TODO CLEAN UP THIS< VERY BAD PRACTICE
    if isinstance(model_to_train, StochViT):
        print("Loading ViT")
        model_to_train.load_on_gpu()
        model = model_to_train
    else:
        model = to_device(model_to_train, device) # TODO Add except param avg std

    

    lr = 0.0001

    history = train(epochs_no, model, train_loader, val_loader, model_dir, logger, lr, with_sam_opt=with_sam_opt, with_indexes=with_indexes)

    return model