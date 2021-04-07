import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from torch.utils.data.dataloader import DataLoader
# from ..utils.utils import get_default_device, batches_to_device, to_device
# from ..dataset.cifar_data_loaders import train_loader, val_loader, test_loader
import torch.nn as nn
import torch
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchvision.utils import make_grid

dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())

labels = dataset.classes

torch.manual_seed(43)
train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * 0.85),
                                                    int(len(dataset) * 0.15)])

batch_size=30

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)


def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() \
      else torch.device('cpu')

def to_device(data, device):
    return [to_device(x, device) for x in data] if isinstance(data, (list,tuple)) \
      else data.to(device, non_blocking=True)

def batches_to_device(data_loader, device):
    for batch in data_loader:
        yield to_device(batch, device)

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

if __name__ == "__main__":
    train_baseline()