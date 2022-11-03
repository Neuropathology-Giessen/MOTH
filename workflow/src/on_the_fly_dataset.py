''' create on-the-fly dataset for pytorch segmentation tasks (random tiles)'''
from typing import List, Dict, Tuple, Any
import random
from random import randint

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from paquo.images import QuPathProjectImageEntry
from mothi.tiling_projects import QuPathTilingProject


class OTFQPDataset(Dataset):
    ''' Custom Dataset holding random QuPath annotations for Segmentation tasks

    Parameters
    ----------
    qp_project : QuPathTilingProject
        project holding images and annotations
    images : List[int]
        QuPath project images (by id) used for dataset
    size : Tuple[int, int]
        size of the tiles
    tile_count : int
        number of tiles in the dataset (number of random tiles per epoch)
    transforms : Any, optional
        transformation to apply on images and labels, by default None
    downsample_level : int, optional
        level for downsampling, by default 0
    random_image : bool, optional
        True: sample qupath image random
        False: compute image by __getitem__ index
    '''
    def __init__(self,
            qp_project_path: str,
            images: List[int],
            size: Tuple[int, int],
            tile_count: int,
            *,
            transforms: Any = None,
            downsample_level: int = 0,
            random_image=True) -> None:

        ## set attributes
        self.qp_project: QuPathTilingProject = QuPathTilingProject(qp_project_path)
        self.imgs: List[int] = images
        self.size: Tuple[int, int] = size
        self.tile_count: int = tile_count
        self.downsample_level: int = downsample_level
        self.transforms = transforms
        self.random_image = random_image
        ## future attributes
        self.img_max_location: Dict[int, Tuple[int, int]]

        ## compute and save maximal location tuple for each image
        img_max_location: Dict[int, Tuple[int, int]] = {}
        img_id: int
        for img_id in self.imgs:
            qp_img: QuPathProjectImageEntry = self.qp_project.images[img_id]
            width: int = qp_img.width
            height: int = qp_img.height
            max_location: Tuple[int, int] = (width - size[0], height - size[1])
            img_max_location[img_id] = max_location
        self.img_max_location = img_max_location


    def __len__(self):
        return self.tile_count


    def __getitem__(self, idx: int):
        image: torch.Tensor
        label: torch.Tensor
        qp_img: int
        rand_location: Tuple[int, int]
        # get (random) image from image list
        if self.random_image:
            qp_img = random.choice(self.imgs)
        else:
            qp_img = self.imgs[idx % (len(self.imgs))]
        # get max location for image
        max_location: Tuple[int, int] = self.img_max_location[qp_img]
        rand_location = (randint(0, max_location[0]), randint(0, max_location[1]))
        image = ToTensor()(
            self.qp_project.get_tile(img_id=qp_img,
                location=rand_location,
                size=self.size,
                downsample_level=self.downsample_level)
            )
        label = torch.from_numpy(
            self.qp_project.get_tile_annot_mask(img_id=qp_img,
                location=rand_location,
                size=self.size,
                downsample_level=self.downsample_level)
        )
        if self.transforms:
            image = self.transforms(image)
            label = self.transforms(label)
        return image, label
