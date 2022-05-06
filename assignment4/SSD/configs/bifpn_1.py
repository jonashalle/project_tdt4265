from ssd.modeling.backbones import BiFPN
from ssd.modeling.backbones.bifpn import EfficientNet
from tops.config import LazyCall as L
from .bifpn_0 import model, anchors, loss_objective, backbone, train, optimizer, schedulers, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform


backbone = L(EfficientNet)(
    phi=0, 
    output_feature_sizes="${anchors.feature_sizes}"
    )
model.backbone = backbone