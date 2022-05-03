from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import json

def get_config(config_path,batch_size: int=1):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = batch_size
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_something(dataloader, cfg):
    for batch in tqdm(dataloader):
        # Remove the two lines below and start analyzing :D
        print("The keys in the batch are:", batch.keys())
        exit()

def load_annotation_file(cfg , set_type: str):
    if set_type == "train":
        _file = cfg["data_train"]["dataset"]["annotation_file"]
    elif set_type == "val":
        _file = cfg["data_val"]["dataset"]["annotation_file"]
    else:
        raise ValueError(f"Unknown set_type: {set_type}")
    if _file.endswith(".json"):
        with open(_file, "r") as f:
            return json.load(f)
    else:
        raise NotImplementedError("Only json are supported!")

def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
