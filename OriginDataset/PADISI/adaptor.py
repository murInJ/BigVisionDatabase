import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from Config.setting import GetOrigindataConfig
from OriginDataset.PADISI.loadDataset import PADISI_Origin_Dataset
from OriginDataset.base import BaseAdaptor

__plugin_name__ = "PADISI"

class Adaptor(BaseAdaptor):
    def __init__(self):
        config = GetOrigindataConfig()
        self.dataset = ConcatDataset([PADISI_Origin_Dataset(config=config, split="train"), PADISI_Origin_Dataset(config=config, split="test")])

    def __getitem__(self, idx):
        data = self.dataset[idx]
        images = {
            "RGB":   {
                "image": data["images"]["RGB"]["image"],
                "dataset_name": data["dataset"],
                "modality": "RGB",
                "extra":{}
            },
            "DEPTH": {
                "image": data["images"]["DEPTH"]["image"],
                "dataset_name": data["dataset"],
                "modality": "DEPTH",
                "extra":{}
            },
            "IR":    {
                "image": data["images"]["IR"]["image"],
                "dataset_name": data["dataset"],
                "modality": "IR",
                "extra":{}
            },
        }
        relation = {
            "FAS":{
                "image_names": ["RGB", "DEPTH", "IR"],
                "task_type": "classification",
                "annotation": {
                    "label": data["label"],
                }
            }
        }
        protocol = {
            "PADISI_train" if data["split"] == "train" else "PADISI_test" : ["FAS"]
        }
        adapted_data = {
            'images': images,
            'relation': relation,
            'protocol':protocol,
        }
        return adapted_data


if __name__ == '__main__':
    config = GetOrigindataConfig()
    dataset = PADISI_Origin_Dataset(config=config, split="train")
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset=dataset)
    for idx, sample in tqdm(enumerate(dataloader),total=len(dataloader)):
        pass