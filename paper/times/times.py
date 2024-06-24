""" Get the execution times of different operations """

import argparse
import re
import timeit
from math import floor
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import tifffile
from numpy import int32
from numpy.typing import NDArray

from mothi.projects import MaskParameter, QuPathTilingProject

IMAGE_DIR = Path("files")
QP_PROJECT_PATH = Path("time_project/Time_QuPath")
SAVED_FILES_PATH = Path("groovy_export")
REPETITIONS: int = 1000


def extract_tiles(
    project: QuPathTilingProject, img_id: int, tile_size: tuple[int, int]
):
    image_size: tuple[int, int] = (
        project.images[img_id].width,
        project.images[img_id].height,
    )

    for x_coord in [
        x * tile_size[0] for x in range(0, floor(image_size[0] / tile_size[0]))
    ]:
        for y_coord in [
            y * tile_size[1] for y in range(0, floor(image_size[1] / tile_size[1]))
        ]:
            tile: NDArray[int32] = project.get_tile(
                img_id, (x_coord, y_coord), tile_size, ret_array=True
            )
            mask: NDArray[int32] = project.get_tile_annotation_mask(
                MaskParameter(img_id, (x_coord, y_coord)), tile_size
            )
            print(np.unique(mask))
            tifffile.imwrite(IMAGE_DIR / f"tile(x={x_coord};y={y_coord}).tiff", tile)
            tifffile.imwrite(IMAGE_DIR / f"mask(x={x_coord};y={y_coord}).tiff", mask)


def read_region_annotations(
    project: QuPathTilingProject,
    img_id: int,
    bounding_box: tuple[int, int, int, int],
    tile_size: tuple[int, int],
):
    for x in range(0, floor((bounding_box[2] - bounding_box[0]) / tile_size[0])):
        x_coord: int = bounding_box[0] + x * tile_size[0]
        for y in range(0, floor((bounding_box[3] - bounding_box[1]) / tile_size[1])):
            y_coord: int = bounding_box[1] + y * tile_size[1]
            project.get_tile_annotation_mask(
                MaskParameter(img_id, (x_coord, y_coord)), tile_size
            )


def load_saved_annotations(path_to_annotations: Path):
    for file_path in Path.iterdir(path_to_annotations):
        groovy_im = cv2.imread(str(file_path))
        np.array(cv2.cvtColor(groovy_im, cv2.COLOR_BGR2GRAY), dtype=np.int32)


def import_masks(path: Path, img_id: int, project: QuPathTilingProject):
    files: list[Path] = [path for path in Path.iterdir(path) if "mask" in path.name]
    print(files)
    for filename in files:
        print(str(filename))
        positions: Union[re.Match[str], None] = re.search(
            r"(?<=x=)(\d*).*(?<=y=)(\d*)", str(filename)
        )
        if not positions:
            raise TypeError("None for positions")
        location: tuple[int, ...] = tuple(
            int(position) for position in positions.groups()
        )
        if not len(location) == 2:
            raise TypeError("Length of Location does not fit")
        mask = tifffile.imread(filename)
        project.save_mask_annotations(mask, MaskParameter(img_id, location))
        project.save()


def merge_annotations(project: QuPathTilingProject):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task", choices=["export", "import", "merge", "to_numpy_time", "load_saved"]
    )
    args: argparse.Namespace = parser.parse_args()

    time: float
    qp_project = QuPathTilingProject(QP_PROJECT_PATH, mode="a")
    if args.task == "load_saved":
        time = timeit.timeit(
            lambda: load_saved_annotations(SAVED_FILES_PATH), number=REPETITIONS
        )
        print(
            f"*** Average time to load saved annotations from the test project: {time / REPETITIONS}"
        )
    if args.task == "to_numpy_time":
        init_time_rep = 1
        time_prep = timeit.timeit(
            lambda: qp_project._update_img_annotation_dict(1), number=init_time_rep
        )
        time = timeit.timeit(
            lambda: read_region_annotations(
                qp_project, 1, (60375, 100375, 63000, 103000), (375, 375)
            ),
            number=REPETITIONS,
        )
        print(f"*** Time to prepare the project: {time_prep / init_time_rep}")
        print(f"*** Average time to convert the project to numpy: {time / REPETITIONS}")
    if args.task == "export":
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        time = timeit.timeit(
            lambda: extract_tiles(qp_project, 0, (128, 128)),
            number=REPETITIONS,
        )
        print(
            f"*** Average time to export tiles and masks from the test project: {time / REPETITIONS}"
        )
    elif args.task == "import":
        print("import path")
        time = timeit.timeit(
            lambda: import_masks(IMAGE_DIR, 1, qp_project),
            number=REPETITIONS,
        )
        print(
            f"*** Average time to import masks into the test project: {time / REPETITIONS}"
        )
