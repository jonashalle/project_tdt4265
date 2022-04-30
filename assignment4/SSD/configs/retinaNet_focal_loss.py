from tops.config import LazyCall as L
from ssd.modeling.ssd_multibox_loss import RetinaFocalLoss 
from .retinaNet_FPN import train, anchors, optimizer, schedulers, model, backbone, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

loss_objective = L(RetinaFocalLoss)(anchors="${anchors}")

# class_names = ("background", "car", "truck", "bus", "motorcycle", "bicycle", "scooter", "person", "rider")