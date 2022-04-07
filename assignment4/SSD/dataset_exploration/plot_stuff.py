from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm

#from ssd.data import TDT4265Dataset
#from .utils import get_dataset_dir

def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
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

def plot_something(dataloader, cfg):
    plot_list = []
    for batch in tqdm(dataloader):
        boxes = batch["boxes"]
        labels = batch["labels"]
        # Remove the two lines below and start analyzing :D
        print("The keys in the batch are:", batch.keys())
        #print("The keys in the batch are:", batch[""])
        zipped = zip(batch["boxes"], batch["labels"])
        plot_list.append(list(zipped))

        print("Batch boxes", batch["boxes"])
        print("Batch labels", batch["labels"])
        
        print("Plot_list", plot_list)

        exit()
        

def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    # img_folder = "data/images/train"
    # annotation_file = "data/train_annotations.json"

    # all_data = TDT4265Dataset(img_folder. annotate_file)

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    plot_something(dataloader, cfg)
    #analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
