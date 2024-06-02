""" on-the-fly datasets for pytorch segmentation tasks """

import random
from abc import ABC, abstractmethod
from bisect import bisect_left
from random import randint
from typing import Any, Optional, cast

import torch
from paquo.images import QuPathProjectImageEntry
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from mothi.projects import MaskParameter, QuPathTilingProject


class QPDataset(Dataset, ABC):
    """abstract dataset with general functionality. \n
    In case of inheritance the method __getitem__ must be overwritten.

    Parameters
    ----------
    qp_project_path : str
        path to project holding images and annotations
    images : list[int]
    QuPath project images (by id) used for dataset
    size : tuple[int, int]
        (width, height) of the tiles
    transforms : Any, optional
        transformation applied to images and labels, by default None
    downsample_level : int, optional
        level for downsampling, by default 0
    tile_count : int, optional
        number of tiles in the dataset, by default 0
    """

    def __init__(
        self,
        qp_project_path: str,
        images: list[int],
        size: tuple[int, int],
        *,
        transforms: Any = None,
        downsample_level: int = 0,
        downsample_base: Optional[int] = None,
        tile_count: int = 0,
    ) -> None:

        ## set attributes
        self.qp_project: QuPathTilingProject = QuPathTilingProject(qp_project_path)
        self.imgs: list[int] = images
        # create dict: index of image_id -> image_id
        self.idx_img_to_img_id: dict[int, int] = dict(enumerate(self.imgs))
        self.size: tuple[int, int] = size
        self.downsample_level: int = downsample_level
        self.transforms: Any = transforms
        self.tile_count: int = tile_count
        self.downsample_base: Optional[int] = downsample_base

    @abstractmethod
    def __getitem__(self, index): ...

    def __len__(self):
        return self.tile_count

    def get_level_0_size(
        self,
        img_id: int,
        base: Optional[int] = None,
    ) -> tuple[int, int]:
        """calculate level 0 tile size

        Parameters
        ----------
        img_id : int
            id of the image from which tiles are to be taken
        base : Optional[int]
            base for the calculation of the downsample factor, by default None

        Returns
        -------
        tuple[int, int]
            level 0 tile size
        """
        downsample_factor: float = self.qp_project.get_downsample_factor(
            self.downsample_level,
            img_id=img_id,
            base=base,
        )
        level_0_size: tuple[int, int] = cast(
            tuple[int, int], tuple(downsample_factor * item for item in self.size)
        )
        return level_0_size

    def _get_and_transform_tile(
        self, img_id: int, location: tuple[int, int]
    ) -> tuple[Tensor, Tensor]:
        """get tile and label for tile

        Parameters
        ----------
        img_id : int
            id of the image from which tiles are to be taken
        location : tuple[int, int]
            location in the image, from which tiles are to be taken

        Returns
        -------
        tuple[Tensor, Tensor]
            image and label as pytorch Tensor
        """
        # get image
        image = ToTensor()(
            self.qp_project.get_tile(
                img_id=img_id,
                location=location,
                size=self.size,
                downsample_level=self.downsample_level,
            )
        )
        # get label
        label = torch.from_numpy(
            self.qp_project.get_tile_annotation_mask(
                MaskParameter(
                    img_id=img_id,
                    location=location,
                    downsample_level=self.downsample_level,
                ),
                size=self.size,
            )
        )
        # transform image and label
        if self.transforms:
            image = self.transforms(image)
            label = self.transforms(label)
        return image, label


class TiledQPDataset(QPDataset):
    """Custom Dataset holding tiled QuPath annotations for Segmentation tasks

    Parameters
    ----------
    qp_project_path : str
        path to project holding images and annotations
    images : list[int]
        QuPath project images (by id) used for dataset
    size : tuple[int, int]
        (width, height) of the tiles
    transforms : Any, optional
        transformation applied to images and labels, by default None
    downsample_level : int, optional
        level for downsampling, by default 0
    """

    def __init__(
        self,
        qp_project_path: str,
        images: list[int],
        size: tuple[int, int],
        *,
        transforms: Any = None,
        downsample_level: int = 0,
        downsample_base: Optional[int] = None,
    ) -> None:

        super().__init__(
            qp_project_path,
            images,
            size,
            transforms=transforms,
            downsample_level=downsample_level,
            downsample_base=downsample_base,
        )
        ## future attributes
        self.img_max_tiles_width: dict[int, int] = {}
        self.tile_count_borders: list[int] = []

        ## compute and save number of tiles per image
        img_tile_count: int = 0
        img_id: int
        for img_id in self.imgs:
            qp_img: QuPathProjectImageEntry = self.qp_project.images[img_id]
            ## compute number of tiles
            # compute number of tiles fitting with downsample factor
            level_0_size: tuple[int, int] = self.get_level_0_size(
                img_id=img_id, base=downsample_base
            )
            nmbr_tiles_width: int = qp_img.width // level_0_size[0]
            nmbr_tiles_height: int = qp_img.height // level_0_size[1]
            self.img_max_tiles_width[img_id] = nmbr_tiles_width
            img_tile_count = nmbr_tiles_width * nmbr_tiles_height
            # add number to total count
            self.tile_count += img_tile_count
            # save computed number
            self.tile_count_borders.append(self.tile_count - 1)

    def __getitem__(self, idx: int):
        return self._get_and_transform_tile(*self.get_location_by_index(idx))

    def get_location_by_index(self, index) -> tuple[int, tuple[int, int]]:
        """docstring to be added"""
        qp_img_id: int
        location: tuple[int, int]
        DOWNSAMPLE_FACTOR: int = 2**self.downsample_level
        level_0_size: tuple[int, int] = cast(
            tuple[int, int], tuple(DOWNSAMPLE_FACTOR * item for item in self.size)
        )

        ## get the image for the given index (idx)
        # bisect idx on border list => index of image_id in self.imgs
        # map: index of image_id -> image_id with self.idx_img_to_img_id
        border_item_idx: int = bisect_left(self.tile_count_borders, index)
        qp_img_id = self.idx_img_to_img_id[border_item_idx]

        ## get the location for the given index (idx)
        # to get the location_id in the image substract id of last border
        location_idx: int
        if border_item_idx == 0:
            location_idx = index
        else:
            # first index in the image is the last index of the previous picture + 1
            first_idx_img: int = self.tile_count_borders[border_item_idx - 1] + 1
            location_idx = index - (first_idx_img)
        ## compute location
        # width = location_idx % width, height = location_idx // width
        maxmul_width: int = self.img_max_tiles_width[qp_img_id]
        location_fac: tuple[int, int] = (
            location_idx % maxmul_width,
            location_idx // maxmul_width,
        )
        location = (
            level_0_size[0] * location_fac[0],
            level_0_size[1] * location_fac[1],
        )

        return qp_img_id, location


class RandomTiledQPDataset(QPDataset):
    """Custom Dataset holding random QuPath annotations for Segmentation tasks

    Parameters
    ----------
    qp_project_path : str
        path to project holding images and annotations
    images : list[int]
        QuPath project images (by id) used for dataset
    size : tuple[int, int]
        (width, height) of the tiles
    tile_count : int
        number of tiles in the dataset (number of random tiles per epoch)
    transforms : Any, optional
       transformation applied to images and labels, by default None
    downsample_level : int, optional
        level for downsampling, by default 0
    choose_random_image : bool, optional
        True: sample qupath image random
        False: compute image by __getitem__ index, by default True
    """

    def __init__(
        self,
        qp_project_path: str,
        images: list[int],
        size: tuple[int, int],
        tile_count: int,
        *,
        transforms: Any = None,
        downsample_level: int = 0,
        downsample_base: Optional[int] = None,
    ) -> None:

        super().__init__(
            qp_project_path,
            images,
            size,
            transforms=transforms,
            downsample_level=downsample_level,
            tile_count=tile_count,
        )
        self.img_max_location: dict[int, tuple[int, int]] = {}

        ## compute and save maximal location tuple for each image
        img_id: int
        for img_id in self.imgs:
            qp_img: QuPathProjectImageEntry = self.qp_project.images[img_id]
            level_0_size: tuple[int, int] = self.get_level_0_size(
                img_id=img_id, base=downsample_base
            )
            width: int = qp_img.width
            height: int = qp_img.height
            max_location: tuple[int, int] = (
                width - level_0_size[0],
                height - level_0_size[1],
            )
            self.img_max_location[img_id] = max_location

    def __getitem__(self, idx: Optional[int] = None):
        qp_img_id: int
        rand_location: tuple[int, int]
        # get (random) image from image list
        if idx is None:
            qp_img_id = random.choice(self.imgs)
        else:
            qp_img_id = self.imgs[idx % (len(self.imgs))]
        # get random location in bound of the maximal location
        max_location: tuple[int, int] = self.img_max_location[qp_img_id]
        rand_location = (randint(0, max_location[0]), randint(0, max_location[1]))

        return self._get_and_transform_tile(qp_img_id, rand_location)
