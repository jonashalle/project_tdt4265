from typing import OrderedDict
import torch.nn as nn
import torchvision.ops as ops
import torchvision.models as models
from typing import Tuple, List

class FPN(nn.Module):
    """
    A new backbone consisting of ResNet-50 with and FPN on top.
    The FPN is made using pytorch helper functions
    
    """
    
    def __init__(self, out_channels):

        super().__init__()
        
        self.model = models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 8)) #default for ResNet is output_size=(1, 1)
        self.out_channels = out_channels
        self.fpn = ops.FeaturePyramidNetwork(in_channels_list =[64, 256, 512, 1024, 2048], out_channels = self.out_channels[0])

    def forward(self, x):
        feature_map_dict = OrderedDict()

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        feature_map_dict["feat0"] = x

        x = self.model.layer1(x)
        x = self.model.maxpool(x)
        feature_map_dict[f"feat1"] = x

        x = self.model.layer2(x)
        feature_map_dict[f"feat2"] = x

        x = self.model.layer3(x)
        feature_map_dict[f"feat3"] = x

        x = self.model.layer4(x)
        feature_map_dict[f"feat4"] = x
        
        x = self.model.avgpool(x)
        feature_map_dict[f"feat5"] = x

        output = self.fpn(feature_map_dict)
        outputs = output.values()
        out_features = list(outputs)
        # print(f"Printing model: {self.model}")
        # print(f"Out_features: {out_features}")
        # print("Printing features")
        # print([(k, v.shape) for k, v in output.items()])
            
        return out_features
        
    
def main():
    ResNet = models.resnet50(pretrained=True)
    ResNet.eval()
    print(ResNet)
    

if __name__ == "__main__":
    main()