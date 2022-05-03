from tops.config import LazyCall as L
from ssd.modeling import RetinaNetSharedHeads
from .retinaNet_subnets import anchors, loss_objective, train, optimizer, schedulers, backbone, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform


model = L(RetinaNetSharedHeads)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,
    subnet_init="xavier" 
)