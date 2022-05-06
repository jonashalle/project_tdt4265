from numpy import size
from torch import no_grad
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BiFPN(nn.Module):
    """
    Implementasion of BiFPN, based on the EfficiendDet paper.
    Code inspired by work by @tristandb, code can be found at https://github.com/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py 
    Inspiration is also taken from @kentaroy47 with code at https://github.com/kentaroy47/efficientdet.pytorch/blob/master/BiFPN.py
    """
    def __init__(self, phi, output_feature_sizes = [[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]]):
        super().__init__()


        # Choosing the baseline EfficientNet as backbone network
        self.model = EfficientNet(phi)

        self.output_feature_shape = output_feature_sizes
        self.num_features = 6 # Number to take out of the BiFPN, is 5 in the paper, but 6 is chosen here
        self.out_channels = [256, 256, 256, 256, 256, 256]
        num_channels = self.out_channels[0]
        
        if phi <= 1:
            self.in_channels = [40, 80, 112, 192, 320, 1280] # Works for EfficientNet-b0 and b1

        assert phi > 2 | phi <= 7, \
            f"BiFPN has not yet been implemented for {phi}"

        self.phi = phi
    
        self.conv_td = []
        self.conv_out = []
        self.scale_conv = []

        # Convolutional layers to convert the channels of the EfficientNet to the same number of channels to go into the BiFPN
        for in_ch, out_ch in zip(self.in_channels, self.out_channels):
            self.scale_conv.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, stride=1, bias=False).to("cuda"))

        # The scaling layers should only multiply by 1
        for conv in self.scale_conv:
            if isinstance(conv, nn.Conv2d):
                conv.weight.requires_grad = False # Making sure these convs don't learn
                nn.init.ones_(conv.weight)

        # Making convolutional layers for upscale and downscale use
        self.conv_td.append(self.bifpn_conv(num_channels, num_channels).to("cuda"))
        for _ in range(self.num_features - 1):
            self.conv_td.append(self.bifpn_conv(num_channels, num_channels).to("cuda"))
            self.conv_out.append(self.bifpn_conv(num_channels, num_channels).to("cuda"))

    def bifpn_layer(self, input_features):
        """
        Takes in a list that is 6 long with P_2 to P_7. 
        Since we have six levels in our SSD model, we choose to widen the BiFPN with one feature for practicality.
        """
        layer_features = []
        td_features = []
        
        # Upsample network
        P_td = self.conv_td[-1](input_features[-1])
        for idx in range(self.num_features - 1):
            i = idx + 2

            # Making the upscale feature list back wards
            td_features.insert(0, P_td) # The upscaled P_2 is the same as the output, so we don't save it as td
            scale = (input_features[-i].size(3)/P_td.size(3))
            P_td = self.conv_td[-i](input_features[-i] + nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)(P_td))

        # Last pass through P_2_td = P_2_out
        P_out = P_td

        # Making the output feature list the right way around,
        # Upsampling to fit default feature_sizes
        upscale = self.upscale()
        layer_features.append(upscale[-1](P_out))
        
        # Downsample network
        i = 2
        for p_in, p_td, conv_out in zip(input_features[1:], td_features, self.conv_out):
            P_out = conv_out(p_in + p_td + F.interpolate(P_out, p_td.size()[2:]))
            layer_features.append(upscale[-i](P_out))
            i += 1 # Good old fashoned iterator

        return layer_features

    def bifpn_conv(self, in_channels, out_channels):
        """
        Helper function for making BiFPN nodes
        """
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
    
    def upscale(self):
        """
        Helper function for scaling the feature maps of the BiFPN,
        to fit our desired feature sizes
        """
        upscale = []

        for i in range(self.num_features):
            upscale.append(nn.Upsample(size=(pow(2, i), pow(2, i+3)), mode="bilinear", align_corners=False))

        return upscale

    def forward(self, x):
        """
        Simple forward pulling 
        """

        P_in = []
        input_features = self.model.forward(x)

        input_features = input_features[-6:] # Only need the six last features

        #with no_grad: # It is not desired to train the weights, but simply change depth
        for conv2d, feature in zip(self.scale_conv, input_features):
            P_in.append(conv2d(feature)) 

        input_features = P_in


        # Depth of the BiFPN is 3 + phi
        out_features = self.bifpn_layer(input_features) 
        out_features = self.bifpn_layer(out_features)
        out_features = self.bifpn_layer(out_features)

        # Adding a depth layer for every number over 3
        for i in range(self.phi):
            out_features = self.bifpn_layer(out_features)

        # Assertion check for good measure
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"

        return out_features

class EfficientNet(nn.Module):
    """
    Implementaton of EfficientNet as backbone to the BiFPN
    """

    def __init__(self, phi = 0):
        super().__init__()

        assert phi < 7 and phi >= 0, \
            f"Expected a number between 0 and 7, got {phi}"
        
        backbones = [models.efficientnet_b0, models.efficientnet_b1, models.efficientnet_b2,
                     models.efficientnet_b3, models.efficientnet_b4, models.efficientnet_b5, models.efficientnet_b6]

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



# model = EfficientNet(0)
# model.trial()