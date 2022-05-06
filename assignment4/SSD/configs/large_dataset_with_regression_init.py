import torch
torch.cuda.empty_cache()
from tops.config import LazyCall as L
from ssd.data import TDT4265Dataset
from .utils import get_dataset_dir
from .retinaNet_initialization import model, anchors, loss_objective, train, optimizer, schedulers, backbone, loss_objective, data_train, data_val, label_map, train_cpu_transform, val_cpu_transform, gpu_transform

data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022_updated/train_annotations.json"))
data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022_updated/val_annotations.json"))