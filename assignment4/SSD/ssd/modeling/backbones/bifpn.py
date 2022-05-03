import torch.nn as nn
import torchvision.models as models


class BiFPN(nn.Module):
    """
    Implementasion of BiFPN, based on the EfficiendDet paper
    """
    def __init__(self, lvl):
        super().__init__()

        # Choosing the baseline EfficientNet as backbone network
        self.model = EfficientNet(lvl)
        self.net_features = []

    def bifpn_layer(self):
        # Make a list of a length 7, and fill it from [-1] from [-1]
        df


class EfficientNet(nn.Module):
    """
    Implementaton of baseline EfficientNet
    """

    def __init__(self, lvl = 0):
        super().__init__()

        assert lvl < 7 and lvl >= 0, \
            f"Expected a number between 0 and 7, got {lvl}"
        
        if lvl == 0:
            self.model = models.efficientnet_b0(pretrained=True)
        elif lvl == 1:
            self.model = models.efficientnet_b1(pretrained=True)
        elif lvl == 2:
            self.model = models.efficientnet_b2(pretrained=True)
        elif lvl == 3:
            self.model = models.efficientnet_b3(pretrained=True)
        elif lvl == 4:
            self.model = models.efficientnet_b4(pretrained=True)
        elif lvl == 5:
            self.model = models.efficientnet_b5(pretrained=True)
        elif lvl == 6:
            self.model = models.efficientnet_b6(pretrained=True)

        

    def forward(self, x):
        out_features = []

        for idx in range():
            x = self.model.children[idx](x)


x = 1
model = EfficientNet(0)
#print(model.layer1)
print(dir(model))
for idx, child in enumerate(model.children()):
    print(child)
    print(idx)