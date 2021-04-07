import torch
from torch.nn.modules.loss import _Loss
from torch.optim import SGD
from torch.nn import Module

def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() \
      else torch.device('cpu')

def to_device(data, device):
    return [to_device(x, device) for x in data] if isinstance(data, (list,tuple)) \
      else data.to(device, non_blocking=True)

def batches_to_device(data_loader, device):
    for batch in data_loader:
        yield to_device(batch, device)

def save_checkpoints(epoch: int, model: Module, optimizer: SGD, loss: _Loss, path: str):
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, path)

def save_model(model: Module, model_name: str, dir: str):
  torch.save(model, f'{dir}{model_name}.pt')

def load_checkpoint(model: Module, checkpoint_path: str):
  model.load_state_dict(torch.load(checkpoint_path))
  model.eval()

def load_model(path: str):
  model = torch.load(path)
  model.eval()

  return model