from tops.config import LazyCall as L
from ssd.modeling import backbones
from .tdt4265 import train, anchors, optimizer, schedulers, model, data_train, data_val, loss_objective, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

train["epochs"] = 50

backbone = L(backbones.FPN)(out_channels = [256, 256, 256, 256, 256, 256])