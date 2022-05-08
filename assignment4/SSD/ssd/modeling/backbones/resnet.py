from typing import OrderedDict
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    """
    Module initializing ResNet-50 and extracting the 6 feature needed 
    to implement the Feature Pyramid Network
    
    """
    
    def __init__(self):

        super().__init__()
        
        self.model = models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 8)) #default for ResNet is output_size=(1, 1)

    def forward(self, x):

        out_features = []

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x) 

        out_features.append(x)

        x = self.model.layer1(x)
        x = self.model.maxpool(x) # Choosing to use an exta MaxPool instead of messing with the architecture of layer1

        out_features.append(x)
       
        x = self.model.layer2(x)

        out_features.append(x)
        

        x = self.model.layer3(x)
        out_features.append(x)

        x = self.model.layer4(x)
        out_features.append(x)

        x = self.model.avgpool(x)
        out_features.append(x)
            
        return out_features

model = ResNet()
print(dir(model))