from tops.config import LazyCall as L
from ssd.modeling import RetinaFocalLoss 
from ssd.data import TDT4265Dataset
from .utils import get_dataset_dir
from .retinaNet_subnets import anchors, loss_objective, train, optimizer, schedulers, backbone, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

alpha = [0.01, 1, 1, 1, 1, 1, 1, 1, 1]

loss_objective = L(RetinaFocalLoss)(anchors="${anchors}", alpha = alpha)


data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022_updated/train_annotations.json"))
data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022_updated/val_annotations.json"))
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform