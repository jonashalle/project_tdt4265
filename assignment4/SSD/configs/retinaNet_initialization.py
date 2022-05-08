from .retinaNet_deeper_heads import train, anchors, optimizer, schedulers, backbone, model, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

model.subnet_init = "gaussian"

