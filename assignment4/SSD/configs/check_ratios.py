from tops.config import LazyCall as L
from ssd.modeling import RetinaNetSharedHeads, RetinaFocalLoss 
from .retinaNet_shared_heads import anchors, model, loss_objective, train, optimizer, schedulers, backbone, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

alpha = [0.01, 1, 1, 1, 1, 1, 1, 1, 1]

loss_objective = L(RetinaFocalLoss)(anchors="${anchors}", alpha = alpha)

model = L(RetinaNetSharedHeads)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,
    subnet_init="gaussian"
)