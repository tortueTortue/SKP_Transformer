import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from torch.utils.data.dataloader import DataLoader
from ..utils.utils import get_default_device, batches_to_device, to_device
from ..dataset.cifar_data_loaders import train_loader, val_loader, test_loader
import torch.nn as nn
import torch

loss = nn.CrossEntropyLoss()

resnet101 = models.resnet101(pretrained=False, progress=True)
resnet101.train()

# class CIFAR10Model(ResNet):
#     def __init__(self):
#         super(CIFAR10Model, self).__init__(BasicBlock, [2, 2, 2, 2])
        
#     def forward(self, x):
#         # change forward here
#         x = self.conv1(x)
#         return x
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation_step(model, batch):
    images, labels = batch 
    images, labels = images.cuda(), labels.cuda()
    out = model.forward(images)
    cross_entropy = nn.CrossEntropyLoss()                  
    val_loss = cross_entropy(out, labels)

    return {'val_loss': val_loss.detach(), 'val_acc': accuracy(out, labels)}

def evaluate(model: nn.Module, val_set: DataLoader):
    outputs = [validation_step(model, batch) for batch in val_set]

    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def train(epochs_no, model: nn.Module, train_set: DataLoader, val_set: DataLoader):
    history = []
    
    # TODO Read about optimizer optimizer = opt_func(model.parameters(), lr)
    
    for epoch in range(epochs_no):
        """  Training Phase """ 
        for batch in train_set:
            inputs, labels = batch
            inputs, labels = inputs.cuda(), labels.cuda()
            curr_loss = loss(model.forward(inputs), labels)
            curr_loss.backward()
            #optimizer.step()
            #optimizer.zero_grad()
        """ Validation phase """
        result = evaluate(model, val_set)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def train_baseline():
    device = get_default_device()

    batches_to_device(train_loader, device)
    batches_to_device(val_loader, device)
    batches_to_device(test_loader, device)

    model = to_device(resnet101, device)

    train(5, model, train_loader, val_loader)
