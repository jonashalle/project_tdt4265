from tops.config import LazyCall as L
from ssd.modeling import RetinaFocalLoss
from .retinaNet_initialization import model, anchors, train, optimizer, schedulers, backbone, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

alpha = [0.01, 1, 10, 10, 10, 10, 10, 3, 5]

loss_objective = L(RetinaFocalLoss)(anchors="${anchors}", alpha = alpha)
