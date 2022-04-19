from typing import OrderedDict
import torch.nn as nn
import torchvision.ops as ops
import torchvision.models as models
from typing import Tuple, List

class NewFPN(nn.Module):
    """
    A new backbone consisting of ResNet-50 with and FPN on top.
    The FPN is made using pytorch helper functions
    
    """
    
    def __init__(self):

        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.fpn = ops.FeaturePyramidNetwork(in_channels_list =[64, 256, 512, 1024, 2048], out_channels = [256])

    def forward(self, x):
        feature_map_dict = OrderedDict()
        for i, feature in enumerate(self.model):
            x = feature(x)
            feature_map_dict[f"feat{i}"] = x

        output = self.fpn(feature_map_dict)
        print([(k, v.shape) for k, v in output.items()])
            
        return output
        
    
def main():
    ResNet = models.resnet50(pretrained=True)
    ResNet.eval()
    print(ResNet)
    

if __name__ == "__main__":
    main()