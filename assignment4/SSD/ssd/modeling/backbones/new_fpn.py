import torch.nn as nn
import torchvision.ops as T
import torch.utils.model_zoo as zoo

class NewFPN(nn.Module):
    """
    A new backbone consisting of ResNet- with and FPN on top.
    The FPN is made using pytorch helper functions
    
    """
    
    def __init__(self, in_channels_list):
        super().__init__()