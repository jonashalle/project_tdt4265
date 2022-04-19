import torchvision
import torch
from tops.config import LazyCall as L
from ssd.modeling import backbones
from .tdt4265 import train, anchors, optimizer, schedulers, model, data_train, data_val, loss_objective, label_map

train["epochs"] = 50

backbone = L(backbones.NewFPN)()