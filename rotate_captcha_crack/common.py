import torch

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count > 0:
        device = torch.device('cuda', device_count - 1)
    else:
        device = torch.device('cpu')
else:
    device = torch.device('cpu')