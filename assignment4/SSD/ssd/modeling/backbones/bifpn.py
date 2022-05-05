import torch.nn as nn
import torchvision.models as models


class BiFPN(nn.Module):
    """
    Implementasion of BiFPN, based on the EfficiendDet paper.
    Code inspired by work by @tristandb, code can be found at https://github.com/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py 
    Inspiration is also taken from @kentaroy47 with code at https://github.com/kentaroy47/efficientdet.pytorch/blob/master/BiFPN.py
    """
    def __init__(self, phi):
        super().__init__()

        # Choosing the baseline EfficientNet as backbone network
        self.model = EfficientNet(phi)

        self.num_features = 6 # Number to take out of the BiFPN, is 5 in the paper, but 6 is chosen here
        self.num_channels = 64
        self.phi = phi
    
        self.conv_td = []
        self.conv_out = []

        # Making convolutional layers for upscale and downscale use
        self.conv_td.append(self.bifpn_conv(self.num_channels, self.num_channels))
        for _ in range(self.num_features - 1):
            self.conv_td.append(self.bifpn_conv(self.num_channels, self.num_channels))
            self.conv_out.append(self.bifpn_conv(self.num_channels, self.num_channels))

    def bifpn_layer(self, P_in):
        """
        Takes in a list that is 6 long with P_2 to P_7. 
        Since we have six levels in our SSD model, we choose to widen the BiFPN with one feature for practicality.
        """
        # Make a list of a length 7, and fill it from [-1] from [-1]
        # P_2_in, P_3_in, P_4_in, P_5_in, P_6_in, P_7 = in_features

        # Here we use one "forward" in each block
        #for feature in self.net_features:
        #    self.convs.append(self.bifpn_conv(in_channels=in_channels, out_channels=out_channels))

        layer_features = []
        td_features = []
        
        # Makes two lists of convs

        # Upsample network
        P_td = self.conv_td[-1](P_in[-1])

        for idx in range(self.num_features - 1):
            # Making the upscale feature list back wards
            td_features.insert(0, P_td) # The upscaled P_3/2 is the same as the output, so we don't save it as td
            i = idx + 2
            scale = (P_in[-i].size(3)/P_td.size(3))
            # Last pass through P_2_td = P_out
            P_td = self.conv_td[-i](P_in[-i] + nn.Upsample(scale_factor=scale, mode="bilinear")(P_td))

        P_out = P_td

        # Making the output feature list the right way around
        layer_features.append(P_out)
        # Downsample network
        for idx in range(self.num_features - 1):
            P_out = self.bifpn_conv()
            layer_features.append(0, P_out)

    def bifpn_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels), # To normalize weights in the fusion
        )

    # def conv_td(self, in_channels, out_channels):
    #     """
    #     Helper function making the intermediate convolutions
    #     """
    #     conv_td = []

    #     for _ in range(6):
    #         conv_td.append(self.bifpn_conv(in_channels, out_channels))

    #     return conv_td

    # def conv_out(self, in_channels, out_channels):
    #     """
    #     Helper function making the output convolution
    #     """
    #     conv_out = []

    #     for _ in range(5):
    #         conv_out.append(self.bifpn_conv(in_channels, out_channels))

    #     return conv_out
    def forward(self, x):
        input_features = self.model.forward(x)

        input_features = input_features[-6:] # We only need the six last features

        # Depth of the BiFPN is 3 + phi
        out_features = self.bifpn_layer(input_features) 
        out_features = self.bifpn_layer(input_features)
        out_features = self.bifpn_layer(input_features)

        for i in self.phi:
            out_features = self.bifpn_layer(input_features)

        return out_features

class EfficientNet(nn.Module):
    """
    Implementaton of baseline EfficientNet
    """

    def __init__(self, phi = 0):
        super().__init__()

        assert phi < 7 and phi >= 0, \
            f"Expected a number between 0 and 7, got {phi}"
        
        # backbones = [models.efficientnet_b0(pretrained=True), models.efficientnet_b1(pretrained=True), models.efficientnet_b2(pretrained=True),
        #             models.efficientnet_b3(pretrained=True), models.efficientnet_b4(pretrained=True), models.efficientnet_b5(pretrained=True), models.efficientnet_b6(pretrained=True)]
        
        backbones = [models.efficientnet_b0, models.efficientnet_b1, models.efficientnet_b2,
                     models.efficientnet_b3, models.efficientnet_b4, models.efficientnet_b5, models.efficientnet_b6]

        # self.model = nn.Sequential(*list(backbones[phi].children())[:-2])
        self.model = backbones[phi](pretrained=True)

    def forward(self, x):
        out_features = []

        for _, feature in enumerate(*list(self.model.children())[:-2]):
            x = feature(x)
            out_features.append(x)
        
        return out_features


    def trial(self):
        print("Here!")
        for idx, feature in enumerate(*list(self.model.children())[:-2]):
            print(f"Feature {idx}: {feature}")



model = EfficientNet(0)
model.trial()