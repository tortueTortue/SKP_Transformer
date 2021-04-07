"""
Here are the the methods computing the metrics used for evaluaton
"""
import torch
from torch.nn import Module

def accuracy(outputs, labels):
    """
    Accuracy = no_of_correct_preidctions / no_of_predictions

    *Note: Use this when the classes have about the amount of occurences.
    """
    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def print_accuracy(model: Module, classes: list, batch_size: int, test_loader):
    class_amount = len(classes)
    class_correct = 0
    class_total = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            i, l = batch
            i, l = i.cuda(), l.cuda()
            out = model(i)
            _, predicted = torch.max(out, 1)
            c = (predicted == l).squeeze()
            for i in range(l.shape[0]):
                label = l[i]
                class_correct += c[i].item()
                class_total += 1

    print('Accuracy of : %2d %%' % (100 * class_correct / class_total))

def print_accuracy_per_class(model: Module, classes: list, batch_size: int, test_loader):
    class_amount = len(classes)
    class_correct = list(0. for i in range(class_amount))
    class_total = list(0. for i in range(class_amount))
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            i, l = batch
            i, l = i.cuda(), l.cuda()
            out = model(i)
            _, predicted = torch.max(out, 1)
            c = (predicted == l).squeeze()
            for i in range(l.shape[0]):
                label = l[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(class_amount):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def count_model_parameters(model: Module, trainable: bool):
    """
    Returns the total amount of parameters in a model.
    
    args:
        model : model to count the parameters of
        trainable : whether to count the trainable params or not
    """
    return (sum(p.numel() for p in model.parameters()) ) if not trainable else \
           (sum(p.numel() for p in model.parameters() if p.requires_grad))
