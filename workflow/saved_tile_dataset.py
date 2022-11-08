import os
from typing import Any, List
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import cv2

class SavedTilesDataset(Dataset):
    ''' dataset for saved tiles

    Parameters
    ----------
    dirs : List[str]
        path to the directory with directories of tiles
    transforms : Any, optional
        transformation applied to images and labels, by default None
    '''
    def __init__(self, dirs: List[str], transforms: Any = None):
        self.img_list = []
        self.label_list = []
        self.dirs = dirs
        self.transforms = transforms

        for directory in dirs:
            file_names = [file for file in os.listdir(directory)
                          if os.path.isfile(os.path.join(directory, file))]
            self.img_list.extend([os.path.join(directory, filename) for filename in file_names])
            self.label_list.extend([os.path.join(
                directory, 'labels', filename.split('.')[0] + '_label.tif'
            ) for filename in file_names])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_list[idx]), cv2.COLOR_BGR2RGB)
        img = ToTensor()(img)
        label_img = cv2.imread(self.label_list[idx], cv2.IMREAD_UNCHANGED)
        label_img = torch.Tensor(label_img)
        if self.transforms:
            img = self.transforms(img)
            label_img = self.transforms(label_img)
        return img, label_img
