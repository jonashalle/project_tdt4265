import sys
assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import functools
import time
import click
import torch
import pprint
import tops
import tqdm
from pathlib import Path
from ssd.evaluate import evaluate
from ssd import utils
from tops.config import instantiate
from tops import logger, checkpointer
from torch.optim.lr_scheduler import ChainedScheduler
from omegaconf import OmegaConf
torch.backends.cudnn.benchmark = True

def class_activation_map(model, image):
    with torch.cuda.amp.autocast(enabled=tops.AMP()):
            bbox_delta, confs = model(batch["image"])
            loss, to_log = model.loss_func(bbox_delta, confs, batch["boxes"], batch["labels"])

    prediction = model.forward(image)



@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def cam(config_path):
    logger.logger.DEFAULT_SCALAR_LEVEL = logger.logger.DEBUG
    cfg = utils.load_config(config_path)
    print_config(cfg)

    cocoGt = dataloader_val.dataset.get_annotations_as_coco()
    model = tops.to_cuda(instantiate(cfg.model))
    optimizer = instantiate(cfg.optimizer, params=utils.tencent_trick(model))
    scheduler = ChainedScheduler(instantiate(list(cfg.schedulers.values()), optimizer=optimizer))
    checkpointer.register_models(
        dict(model=model, optimizer=optimizer, scheduler=scheduler))
    total_time = 0
    tops.print_module_summary(model, (dummy_input,))

    if checkpointer.has_checkpoint():
        train_state = checkpointer.load_registered_models(load_best=False)
        total_time = train_state["total_time"]
        logger.log(f"Resuming train from: epoch: {logger.epoch()}, global step: {logger.global_step()}")
    
    tops.print_module_summary(model, (dummy_input,))

    return 1