import torch.nn as nn
from typing import Tuple, List


def relu_conv_layer(in_channel_size, out_channel_size, num_filters, conv_kernel_size, conv_padding):
    return nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channel_size,
                out_channels=num_filters,
                kernel_size = conv_kernel_size,
                stride = 1,
                padding = conv_padding
            ),
            nn.ReLU(),            
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=out_channel_size,
                kernel_size = conv_kernel_size,
                stride = 2,
                padding = conv_padding
            ),
            nn.ReLU(),
        )
class BasicModel(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.conv_kernel_size = 3
        self.conv_padding = 1
        self.pool_kernel_size = 2
        self.pool_stride = 2


        
        # output_channels[0]
        self.feature_extractor = nn.ModuleList([nn.Sequential(
            nn.Conv2d(
                in_channels = image_channels,
                out_channels = 32,
                kernel_size = self.conv_kernel_size,
                stride = 1,
                padding = self.conv_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_kernel_size, stride = self.pool_stride),
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = self.conv_kernel_size,
                stride = 1,
                padding = self.conv_padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 64,
                out_channels = 64,
                kernel_size = self.conv_kernel_size,
                stride = 1,
                padding = self.conv_padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 64,
                out_channels = output_channels[0],
                kernel_size = self.conv_kernel_size,
                stride = 2,
                padding = self.conv_padding
            ),
            nn.ReLU(),
        )])
        self.feature_extractor.append(relu_conv_layer(output_channels[0], output_channels[1], 128, self.conv_kernel_size, self.conv_padding))

        self.feature_extractor.append(relu_conv_layer(output_channels[1],output_channels[2],256,self.conv_kernel_size,self.conv_padding))
        
        self.feature_extractor.append(relu_conv_layer(output_channels[2],output_channels[3],128,self.conv_kernel_size,self.conv_padding))
        self.feature_extractor.append(relu_conv_layer(output_channels[3],output_channels[4],128,self.conv_kernel_size,self.conv_padding))
        # self.feature_extractor.append(nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(
        #         in_channels=output_channels[3],
        #         out_channels=output_channels[4],
        #         kernel_size = self.conv_kernel_size,
        #         stride = 1,
        #         padding = self.conv_padding
        #     ),
        #     nn.ReLU(),            
        #     nn.Conv2d(
        #         in_channels=output_channels[3],
        #         out_channels=128,
        #         kernel_size = self.conv_kernel_size,
        #         stride = 2,
        #         padding = self.conv_padding
        #     ),
        #     nn.ReLU(),
        # ))
        self.feature_extractor.append(nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[4],
                out_channels=128,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_kernel_size, stride = self.pool_stride),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
        ))
        

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for feature in self.feature_extractor:
            x = feature(x)
            out_features.append(x)
        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)



        # a comment to test something