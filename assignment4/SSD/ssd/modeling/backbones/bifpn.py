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
    def __init__(self, phi):
        super().__init__()

        # Choosing the baseline EfficientNet as backbone network
        self.model = EfficientNet(phi)

        self.num_features = 6 # Number to take out of the BiFPN, is 5 in the paper, but 6 is chosen here
        self.out_channels = [256, 256, 256, 256, 256, 256]
        num_channels = self.out_channels[0]
        
        if phi <= 1:
            self.in_channels = [40, 80, 112, 192, 320, 1280] # Works for EfficientNet-b0 and b1

        self.phi = phi
    
        self.conv_td = []
        self.conv_out = []
        self.scale_conv = []

        # Convolutional layers to convert the channels of the EfficientNet to the same number of channels to go into the BiFPN
        for in_ch, out_ch in zip(self.in_channels, self.out_channels):
            self.scale_conv.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, stride=1, bias=False))

        # The scaling layers should only multiply by 1
        for conv in self.scale_conv:
            if isinstance(conv, nn.Conv2d):
                conv.weight.requires_grad = False # Making sure these convs don't learn
                nn.init.ones_(conv.weight)

        # Making convolutional layers for upscale and downscale use
        self.conv_td.append(self.bifpn_conv(num_channels, num_channels))
        for _ in range(self.num_features - 1):
            self.conv_td.append(self.bifpn_conv(num_channels, num_channels))
            self.conv_out.append(self.bifpn_conv(num_channels, num_channels))

    def bifpn_layer(self, input_features):
        """
        Takes in a list that is 6 long with P_2 to P_7. 
        Since we have six levels in our SSD model, we choose to widen the BiFPN with one feature for practicality.
        """
        layer_features = []
        td_features = []
        

        # Upsample network
        print(f"Inputfeature: {input_features[-4].size(1)}")
        print(f"What is conv: {self.conv_td[-1]}")


        P_td = self.conv_td[-1](input_features[-1])
        print(f"Size of P_td: {P_td.shape}")
        for idx in range(self.num_features - 1):
            i = idx + 2

            # Making the upscale feature list back wards
            td_features.insert(0, P_td) # The upscaled P_2 is the same as the output, so we don't save it as td
            scale = (input_features[-i].size(3)/P_td.size(3))
            P_td = self.conv_td[-i](input_features[-i] + nn.Upsample(scale_factor=scale, mode="bilinear")(P_td))

        # Last pass through P_2_td = P_2_out
        P_out = P_td

        # Making the output feature list the right way around
        layer_features.append(P_out)

        # Downsample network
        for p_in, p_td, conv_out in zip(input_features[1:], td_features, self.conv_out):
            P_out = conv_out(p_in + p_td + F.interpolate(P_out, p_td.size()[2:]))
            layer_features.append(P_out)

        return layer_features

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

    def forward(self, x):
        P_in = []
        input_features = self.model.forward(x)

        input_features = input_features[-6:] # Only need the six last features

        #with no_grad: # It is not desired to train the weights, but simply change depth
        for conv2d, feature in zip(self.scale_conv, input_features):
            P_in.append(conv2d(feature)) 

        input_features = P_in

        for idx, features in enumerate(input_features):
            print(f"Feature size {idx}: {features.shape}")

        """
        In order to make the BiFPN work we need to
        convert all the channels from the input_features to one channel depth
        This can be done using a convolutional layer with kernel size 1 and no bias or padding.
        these layers must be initiated with ones as weights and will be made new for very forward pass.
        """

        # Depth of the BiFPN is 3 + phi
        out_features = self.bifpn_layer(input_features) 
        out_features = self.bifpn_layer(out_features)
        out_features = self.bifpn_layer(out_features)

        for i in self.phi:
            out_features = self.bifpn_layer(out_features)

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