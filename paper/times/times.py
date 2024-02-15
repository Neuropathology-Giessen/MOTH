""" Get the execution times of different operations """

import argparse
import re
import timeit
from math import floor
from pathlib import Path
from typing import Union

import numpy as np
import tifffile
from numpy import int32
from numpy.typing import NDArray

from mothi.tiling_projects import QuPathTilingProject

IMAGE_DIR = Path("files")
QP_PROJECT_PATH = Path("times_project")
REPETITIONS: int = 10


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
            mask: NDArray[int32] = project.get_tile_annot_mask(
                img_id, (x_coord, y_coord), tile_size
            )
            print(np.unique(mask))
            tifffile.imwrite(IMAGE_DIR / f"tile(x={x_coord};y={y_coord}).tiff", tile)
            tifffile.imwrite(IMAGE_DIR / f"mask(x={x_coord};y={y_coord}).tiff", mask)


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
        project.save_mask_annotations(img_id, mask, location)
        project.save()


def merge_annotations(project: QuPathTilingProject):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["export", "import", "merge"])
    args: argparse.Namespace = parser.parse_args()

    time: float
    qp_project = QuPathTilingProject(QP_PROJECT_PATH, mode="a")
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
