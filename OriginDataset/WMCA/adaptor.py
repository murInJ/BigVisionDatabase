import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from Config.setting import GetOrigindataConfig
from OriginDataset.WMCA.loadDataset import WMCA_Origin_Dataset
from OriginDataset.base import BaseAdaptor

__plugin_name__ = "WMCA"

class Adaptor(BaseAdaptor):
    def __init__(self):
        config = GetOrigindataConfig()
        self.dataset = ConcatDataset([WMCA_Origin_Dataset(config=config, split="train"), WMCA_Origin_Dataset(config=config, split="test")])

    def __getitem__(self, idx):
        data = self.dataset[idx]
        images = {
            "RGB": data['images']["RGB"]["image"],
            "DEPTH": data['images']["DEPTH"]["image"],
            "IR": data['images']["IR"]["image"],
        }
        relation = {
            "FAS":{
                "image_names": ["RGB", "DEPTH", "IR"],
                "task_type": "classification",
                "annotation": {
                    "label": data["label"],
                    "dataset": data["dataset"],
                }
            }
        }
        protocol = {
            "WMCA_train" if data["split"] == "train" else "WMCA_test" : ["FAS"]
        }
        adapted_data = {
            'images': images,
            'relation': relation,
            'protocol':protocol,
        }
        return adapted_data


if __name__ == '__main__':
    config = GetOrigindataConfig()
    dataset = WMCA_Origin_Dataset(config=config, split="train")
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset=dataset)
    for idx, sample in tqdm(enumerate(dataloader),total=len(dataloader)):
        pass