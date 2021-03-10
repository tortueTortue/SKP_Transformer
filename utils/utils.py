import torch
# TODO check if it works
def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() \
      else torch.device('cpu')

def to_device(data, device):
    return [to_device(x, device) for x in data] if isinstance(data, (list,tuple)) \
      else data.to(device, non_blocking=True)

def batches_to_device(data_loader, device):
    for batch in data_loader:
        yield to_device(batch, device)