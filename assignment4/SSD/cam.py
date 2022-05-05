
import torchvision
import torch
import tqdm
import click
import numpy as np
import tops
from ssd import utils
from tops.config import instantiate
from PIL import Image
from vizer.draw import draw_boxes
from tops.checkpointer import load_checkpoint
from pathlib import Path
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt1

@torch.no_grad()
@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))

@click.option("-s", "--score_threshold", type=click.FloatRange(min=0, max=1), default=.65)
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
    
    image_name = 'data/tdt4265_2022/images/train/trip007_glos_Video00000_7.png'
    orig_img = np.array(Image.open(image_name).convert("RGB"))
    
    height, width = orig_img.shape[:2]
    img = cpu_transform({"image": orig_img})["image"].unsqueeze(0)
    img = tops.to_cuda(img)
    img = gpu_transform({"image": img})["image"]
    boxes, categories, scores = model(img,score_threshold=score_threshold)[0]
    print(scores)
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    boxes, categories, scores = [_.cpu().numpy() for _ in [boxes, categories, scores]]
    drawn_image = draw_boxes(
        orig_img, boxes, categories, scores).astype(np.uint8)
    im = Image.fromarray(drawn_image)
    
    print('scores len',len(scores))




    
    plt1.imshow(orig_img)
    plt1.show()
    
    plt1.imshow(im)
    plt1.show()

if __name__ == '__main__':
    cam()
