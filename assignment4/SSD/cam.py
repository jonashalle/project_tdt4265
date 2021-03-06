
import torchvision
import torch
import tqdm
import click
import numpy as np
import tops
from ssd import utils
from tops.config import instantiate
from PIL import Image
from tops.checkpointer import load_checkpoint
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import cv2
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image



import requests
import torchvision
from PIL import Image


def draw_b(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
    return image


@torch.no_grad()
@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))

@click.option("-s", "--score_threshold", type=click.FloatRange(min=0, max=1), default=.85)
def cam(config_path: Path, score_threshold: float):
    cfg = utils.load_config(config_path)
    model = tops.to_cuda(instantiate(cfg.model))
    model.eval()
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])
    dataset_to_visualize = "train" # or "val"
    cfg.train.batch_size = 1
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        if cfg.data_train.dataset._target_ == torch.utils.data.ConcatDataset:
            for dataset in cfg.data_train.dataset.datasets:
                dataset.transform.transforms = dataset.transform.transforms[:-1]
        else:
            cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        dataset = instantiate(cfg.data_train.dataloader)
        gpu_transform = instantiate(cfg.data_train.gpu_transform)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        dataset = instantiate(cfg.data_val.dataloader) 
        gpu_transform = instantiate(cfg.data_val.gpu_transform)
    cpu_transform = instantiate(cfg.data_val.dataset.transform)
    gpu_transform = instantiate(cfg.data_val.gpu_transform)

    first_video = 0 
    for j in range(3): 
        video_number = j+first_video 
        for image_number in range(100):
            image_name = 'data/tdt4265_2022/images/train/trip007_glos_Video000{}_{}.png'.format(video_number,image_number)
            orig_img = np.array(Image.open(image_name).convert("RGB"))
        
            height, width = orig_img.shape[:2]
            img = cpu_transform({"image": orig_img})["image"].unsqueeze(0)
            img = tops.to_cuda(img)
            img = gpu_transform({"image": img})["image"]
            boxes, labels, scores = model(img,score_threshold=score_threshold)[0]        
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
            boxes, labels, scores = [_.cpu().numpy() for _ in [boxes, labels, scores]]
 


            input_tensor = img
            image_float_np = np.float32(orig_img)/255  
            classes = ("background","car", "truck", "bus", "motorcycle", "bicycle", "scooter", "person", "rider")
            fpn_model = model.feature_extractor
            target_layers = [fpn_model.model.model.layer3]
        
            
            targets = [FasterRCNNBoxScoreTarget(labels, boxes,0.8)] 
            cam = EigenCAM(model,               #nn.Module
                        target_layers,          # List[nn.Module]
                        use_cuda=torch.cuda.is_available()
                        )
            cam.uses_gradients = False                                   
            grayscale_cam = cam(input_tensor, targets=targets)             
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)            
            image_with_bounding_boxes = draw_b(boxes, labels, classes, cam_image)
            Image.fromarray(image_with_bounding_boxes)
            plt.imshow(image_with_bounding_boxes)
 
            path = "CAM_figures/video{}/img{}".format(video_number,image_number)
            plt.savefig(path)
        

if __name__ == '__main__':
    cam()

    