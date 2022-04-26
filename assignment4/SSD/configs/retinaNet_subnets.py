from tops.config import LazyCall as L
from ssd.modeling import RetinaNet
from .retinaNet_focal_loss import train, anchors, optimizer, schedulers, backbone, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

subnet_init = "uniform"

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,
    subnet_init="uniform" 
)