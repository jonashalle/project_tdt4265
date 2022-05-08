from tops.config import LazyCall as L
from ssd.modeling.ssd_multibox_loss import RetinaFocalLoss 
from .retinaNet_FPN import train, anchors, optimizer, schedulers, model, backbone, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

alpha = [0.01, 1, 1, 1, 1, 1, 1, 1, 1]  # Default alpha from project assignment

loss_objective = L(RetinaFocalLoss)(anchors="${anchors}", alpha = alpha)