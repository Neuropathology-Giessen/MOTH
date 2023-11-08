""" Get the execution times of different operations """
from math import floor
from pathlib import Path

from mothi.tiling_projects import QuPathTilingProject


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
            project.get_tile(img_id, (x_coord, y_coord), tile_size)
            project.get_tile_annot_mask(img_id, (x_coord, y_coord), tile_size)


def import_tiles(path: Path, project: QuPathTilingProject):
    pass


def merge_annotations(project: QuPathTilingProject):
    pass


if __name__ == "__main__":
    pass
