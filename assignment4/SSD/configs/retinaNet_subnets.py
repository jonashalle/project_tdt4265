from tops.config import LazyCall as L
from ssd.modeling import RetinaNet, RetinaFocalLoss, AnchorBoxes
from .retinaNet_focal_loss import train, optimizer, schedulers, backbone, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [124, 400]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    # In order to share heads we need to share aspect ratios

    # For shared heads to work, we need to have the same number of boxes for every resolution,
    # to benchmark multiple subnets, we choose to use the same anchors as this
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],

    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

loss_objective = L(RetinaFocalLoss)(anchors="${anchors}")

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,
    subnet_init="xavier" 
)