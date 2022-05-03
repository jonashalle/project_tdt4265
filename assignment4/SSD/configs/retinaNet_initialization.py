# from tops.config import LazyCall as L
# from ssd.modeling import retinaNet
from .retinaNet_subnets import train, anchors, optimizer, schedulers, backbone, model, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

model.subnet_init = "gaussian"

