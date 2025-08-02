# loss_factory.py

import torch.nn.functional as F
import torch.nn as nn

def get_loss_fn(name: str):
    if name == "mse":
        return lambda x, y: nn.MSELoss()(x, y)
    elif name == "smooth":
        return F.smooth_l1_loss
    elif name == "l1":
        return lambda x, y: nn.L1Loss()(x, y)
    else:
        raise ValueError(f"Unknown loss function: {name}")
    
    
    
