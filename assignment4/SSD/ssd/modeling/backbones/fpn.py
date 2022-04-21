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

    # def forward(self, x):
    #     #feature_map_dict = OrderedDict() 
    #     out_features = []


    #     x = self.model.conv1(x)
    #     x = self.model.bn1(x)
    #     x = self.model.relu(x)
    #     x = self.model.maxpool(x) 
    #     #feature_map_dict["feat0"] = x
    #     out_features.append(x)
    #     print(f"x1.shape {x.shape}")
    #     x = self.model.layer1(x)
    #     x = self.model.maxpool(x) # Choosing to use an exta MaxPool instead of messing with the architecture of layer1
    #     #feature_map_dict["feat1"] = x
    #     out_features.append(x)
    #     print(f"x2.shape {x.shape}")
    #     x = self.model.layer2(x)
    #     #feature_map_dict["feat2"] = x
    #     out_features.append(x)
    #     print(f"x3.shape {x.shape}")

    #     x = self.model.layer3(x)
    #     #feature_map_dict["feat3"] = x
    #     out_features.append(x)
    #     print(f"x4.shape {x.shape}")

    #     x = self.model.layer4(x)
    #     #feature_map_dict["feat4"] = x
    #     out_features.append(x)
    #     print(f"x5.shape {x.shape}") 

    #     x = self.model.avgpool(x)
    #     #feature_map_dict["feat5"] = x
    #     out_features.append(x)
    #     print(f"x6.shape {x.shape}")

    #     #output = self.fpn(feature_map_dict)
    #     #outputs = output.values()
    #     #out_features = list(outputs)
    #     # print(f"Printing model: {self.model}")
    #     # print(f"Out_features: {out_features}")
    #     # print("Printing features")
    #     # print([(k, v.shape) for k, v in output.items()])
            
    #     return out_features

    def forward(self, x):
        resnet_features = self.model.forward(x)

        feature_map_dict = OrderedDict() 

        for idx, feature in enumerate(resnet_features):
            feature_map_dict[f"feat{idx}"] = feature

        output = self.fpn(feature_map_dict)
        outputs = output.values()
        out_features = list(outputs)

        return out_features