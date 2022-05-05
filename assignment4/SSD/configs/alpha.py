from tops.config import LazyCall as L
from ssd.modeling import retinaNet, AnchorBoxes, RetinaFocalLoss
from .retinaNet_aspectRatios import model, anchors, train, optimizer, schedulers, backbone, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

# kjøring mens dere var å spiste alpha = [0.01, 1, 10, 10, 10, 10, 10, 3, 4]  # Default alpha from project assignment
alpha = [0.01, 1, 10, 9, 15, 7, 8, 3, 6]
print('alpha = ', alpha)
loss_objective = L(RetinaFocalLoss)(anchors="${anchors}", alpha = alpha)



# config 1 alpha = [0.01, 1, 10, 10, 10, 10, 10, 3, 4]
# config 2 alpha = [0.01, 1, 10, 9, 15, 7, 8, 3, 6]