from tops.config import LazyCall as L
from .bifpn_0 import model, anchors, loss_objective, backbone, train, optimizer, schedulers, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

backbone.phi = 6