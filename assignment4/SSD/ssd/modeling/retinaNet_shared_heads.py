import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms
import numpy as np

class RetinaNetSharedHeads(nn.Module):
    def __init__(self, 
            feature_extractor: nn.Module,
            anchors,
            loss_objective,
            num_classes: int,
            subnet_init = "xavier"):
        super().__init__()
        """
            Implements the SSD network, consisting of a five layer classification head and a parallel regression head.
            These two heads are shared across all six resolutions of the feature extractor.
            Backbone outputs a list of features, which are gressed to SSD output with regression/classification heads.
        """

        self.feature_extractor = feature_extractor
        self.loss_func = loss_objective
        self.num_classes = num_classes
        self.num_boxes = anchors.num_boxes_per_fmap
        box_num = self.num_boxes[0]
        out_ch = self.feature_extractor.out_channels[0]
        
        # Set 
        self.classification_heads = self.subnet(out_ch, box_num * self.num_classes)
        self.regression_heads = self.subnet(out_ch, 4 * box_num)

        self.anchor_encoder = AnchorEncoder(anchors)
        self._init_weights(subnet_init)

    def _init_weights(self, subnet_init):
        layers = [*self.regression_heads, *self.classification_heads]
        
        if subnet_init == "xavier": # Original xavier_uniform parameter initialization
            for layer in layers:
                for param in layer.parameters():
                    if param.dim() > 1: 
                        nn.init.xavier_uniform_(param)

        if subnet_init == "gaussian": # Our weight and bias initialization based on the Focal loss paper
            for layer in layers:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0.0,std=0.01)
                    nn.init.zeros_(layer.bias)
            p = 0.99        
            bias = np.log(p * (self.num_classes-1)/(1-p))
            nn.init.constant_(self.classification_heads[-1].bias[:self.num_boxes[0]], bias)
                
    def regress_boxes(self, features):
        locations = []
        confidences = []
        for x in features:
            bbox_delta = self.regression_heads(x).view(x.shape[0], 4, -1)
            bbox_conf = self.classification_heads(x).view(x.shape[0], self.num_classes, -1)
            locations.append(bbox_delta)
            confidences.append(bbox_conf)
        bbox_delta = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()
        return bbox_delta, confidences

    def subnet(self, in_channels, out_channels):
        """
        Helper function for making the classification and regression heads.
        This is a direct implementation of the one that can be found in the paper "Focal loss for dense object detection"
        """
        return nn.Sequential(
        nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        # A Softmax is added at the end of the head in the loss function 
        )    
    
    def forward(self, img: torch.Tensor, **kwargs):
        """
            img: shape: NCHW
        """
        if not self.training:
            return self.forward_test(img, **kwargs)
        features = self.feature_extractor(img)
        return self.regress_boxes(features)
    
    def forward_test(self,
            img: torch.Tensor,
            imshape=None,
            nms_iou_threshold=0.5, max_output=200, score_threshold=0.05):
        """
            img: shape: NCHW
            nms_iou_threshold, max_output is only used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx], confs[img_idx],
                nms_iou_threshold, max_output, score_threshold)
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))
        return predictions

 
def filter_predictions(
        boxes_ltrb: torch.Tensor, confs: torch.Tensor,
        nms_iou_threshold: float, max_output: int, score_threshold: float):
        """
            boxes_ltrb: shape [N, 4]
            confs: shape [N, num_classes]
        """
        assert 0 <= nms_iou_threshold <= 1
        assert max_output > 0
        assert 0 <= score_threshold <= 1
        scores, category = confs.max(dim=1)

        # 1. Remove low confidence boxes / background boxes
        mask = (scores > score_threshold).logical_and(category != 0)
        boxes_ltrb = boxes_ltrb[mask]
        scores = scores[mask]
        category = category[mask]

        # 2. Perform non-maximum-suppression
        keep_idx = batched_nms(boxes_ltrb, scores, category, iou_threshold=nms_iou_threshold)

        # 3. Only keep max_output best boxes (NMS returns indices in sorted order, decreasing w.r.t. scores)
        keep_idx = keep_idx[:max_output]
        return boxes_ltrb[keep_idx], category[keep_idx], scores[keep_idx]

       

    
    
  
        
        

