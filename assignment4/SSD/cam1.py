import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image

import requests
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

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

def predict(class_names, input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    print('class names---------', class_names)
    print('outputs---------', type(outputs))
    pred_classes = [class_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    
    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices

# def draw_boxes(boxes, labels, classes, image):
#     COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
#     for i, box in enumerate(boxes):
#         color = COLORS[labels[i]]
#         cv2.rectangle(
#             image,
#             (int(box[0]), int(box[1])),
#             (int(box[2]), int(box[3])),
#             color, 2
#         )
#         cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
#                     lineType=cv2.LINE_AA)
#     return image




@torch.no_grad()
@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))

@click.option("-s", "--score_threshold", type=click.FloatRange(min=0, max=1), default=.65)
def cam(config_path: Path, score_threshold: float):
    cfg = utils.load_config(config_path)
    

    class_names = ("background", "car", "truck", "bus", "motorcycle", "bicycle", "scooter", "person", "rider")

    # This will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    cpu_transform = instantiate(cfg.data_val.dataset.transform)
    gpu_transform = instantiate(cfg.data_val.gpu_transform)
    image_name = 'data/tdt4265_2022/images/train/trip007_glos_Video00000_7.png'
    image = np.array(Image.open(image_name).convert("RGB"))
    image_float_np = np.float32(image) / 255
    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    
    input_tensor = transform(image)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)
    

    # cfg = utils.load_config(config_path)
    # model = tops.to_cuda(instantiate(cfg.model))
    # model.eval()
    model = tops.to_cuda(instantiate(cfg.model))
    model.eval().to(device)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model.eval().to(device)
    height, width = image.shape[:2]
    img = cpu_transform({"image": image})["image"].unsqueeze(0)
    img = tops.to_cuda(img)
    img = gpu_transform({"image": img})["image"]
    boxes, categories, scores = model(img,score_threshold=score_threshold)[0]

    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    boxes, categories, scores = [_.cpu().numpy() for _ in [boxes, categories, scores]]

    drawn_image = draw_boxes(
    image, boxes, categories, scores).astype(np.uint8)
    im = Image.fromarray(drawn_image)
    # Run the model and display the detections
    # outputs = []
    # boxes, classes, labels, indices = predict(class_names, input_tensor, model, device, 0.7)
    # image = draw_boxes(boxes, labels, classes, image)

    # Show the image:
    #Image.fromarray(image)
    plt1.imshow(im)
    plt1.show()



if __name__ == '__main__':
    cam()