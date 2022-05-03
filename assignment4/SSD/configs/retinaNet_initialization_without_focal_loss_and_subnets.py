from .retinaNet_FPN import train, anchors, model, optimizer, schedulers, backbone, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

model.subnet_init = "gaussian"