from typing import OrderedDict
import torch.nn as nn
import torchvision.ops as ops
from .resnet import ResNet

class FPN(nn.Module):
    """
    A new backbone consisting of ResNet-50 with and FPN on top.
    The FPN is made using pytorch helper functions
    
    """
    
    def __init__(self, out_channels):

        super().__init__()
        
        self.model = ResNet()
        self.out_channels = out_channels

        # Initializing FPN helperfunction using in and out channels from the provided paper
        self.fpn = ops.FeaturePyramidNetwork(in_channels_list =[64, 256, 512, 1024, 2048], out_channels = self.out_channels[0])


    def forward(self, x):
        resnet_features = self.model.forward(x)

        feature_map_dict = OrderedDict() 

        for idx, feature in enumerate(resnet_features):
            feature_map_dict[f"feat{idx}"] = feature

        output = self.fpn(feature_map_dict)
        outputs = output.values()
        out_features = list(outputs)

        return out_features