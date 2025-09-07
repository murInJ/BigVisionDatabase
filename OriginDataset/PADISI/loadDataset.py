import os

import pandas as pd
import cv2
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from Config.setting import GetOrigindataConfig


class PADISI_Origin_Dataset(Dataset):
    def __init__(self,config="config.yaml",split='train'):
        super(PADISI_Origin_Dataset, self).__init__()
        self.split = split
        self.root = config.get("PADISI_root")
        self.protocol = config.get("PADISI_train_protocol") if split == "train" else config.get("PADISI_test_protocol")

        self.len = 0
        self.dir_list = []

        self.load_PADISI(self.protocol)

    def __len__(self):
        return len(self.dir_list)

    def load_PADISI(self, protocol):
        """USC中spoofing_label为0的是假脸，为1的是真脸"""
        landmarks_frame = pd.read_csv(protocol, delimiter='	', header=None)
        for idx in range(len(landmarks_frame)):
            video_name = str(landmarks_frame.iloc[idx, 0]).split('.npz')[0]

            rgb_path = video_name + '/color.jpg'
            depth_path = video_name + '/depth.jpg'
            ir_path = video_name + '/ir.jpg'

            rgb_path = os.path.join(self.root, rgb_path)
            depth_path = os.path.join(self.root, depth_path)
            ir_path = os.path.join(self.root, ir_path)

            spoofing_label = landmarks_frame.iloc[idx, 1]

            """WMCA中spoofing_label为0的是假脸，为1的是真脸"""
            self.dir_list.append({
                "RGB_path": rgb_path,
                "DEPTH_path": depth_path,
                "IR_path": ir_path,
                "label": "Spoof" if spoofing_label == 0 else "Real Face",
                "possible_label": ["Spoof", "Real Face"],
                "dataset": 'PADISI',
                "split": self.split,
            })


    def __getitem__(self, idx):
        spoofing_label = self.dir_list[idx]['label']
        image_x = self.get_single_image_x(self.dir_list[idx]['RGB_path'])
        image_x_depth = self.get_single_image_x(self.dir_list[idx]['DEPTH_path'])
        image_x_ir = self.get_single_image_x(self.dir_list[idx]['IR_path'])

        sample = {
            'images': {
                "RGB": {
                    "image": image_x,
                    "path": self.dir_list[idx]['RGB_path']
                },
                "DEPTH": {
                    "image": image_x_depth,
                    "path": self.dir_list[idx]['DEPTH_path']
                },
                "IR": {
                    "image": image_x_ir,
                    "path": self.dir_list[idx]['IR_path']
                }
            },
            'label': spoofing_label,
            'possible_label': ",".join(self.dir_list[idx]['possible_label']),
            'dataset': self.dir_list[idx]['dataset'],
            'split': self.dir_list[idx]['split'],
            'task_type':"classification"
        }

        return sample

    def get_single_image_x(self, image_path):

        # RGB
        image_x = cv2.imread(image_path)

        # cv2.imwrite('temp.jpg', image_x_temp)
        image_x = cv2.resize(image_x, (224, 224))

        return image_x

if __name__ == '__main__':
    config = GetOrigindataConfig()
    dataset = PADISI_Origin_Dataset(config=config, split="train")
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset=dataset)
    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        pass