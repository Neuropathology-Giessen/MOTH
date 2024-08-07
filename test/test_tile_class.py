""" Test the QuPathTilingProject class and its functions """

import os
import pickle
import shutil
import unittest

import numpy as np
from numpy.typing import NDArray
from paquo.classes import QuPathPathClass
from paquo.images import QuPathProjectImageEntry
from shapely.geometry import Polygon
from shapely.strtree import STRtree

import moth
from moth.projects import MaskParameter, QuPathTilingProject

QUPATH_PATH: str = (
    "test/test_projects/qp_project/project.qpproj"  # generated by create_qp_project.ipynb
)
TEMPORARY_QP_PATH: str = "test/test_projects/temporary_project"
SLIDE_PATH: str = "test/test_projects/slides/white-4096.tif"
EXPECTED_MASK_PATH: str = "test/expected_masks.pkl"


class TestTileImportExportCycle(unittest.TestCase):
    """
    export, import in new project, then export to prove the circle outcome
    *QuPath drawn annotations*
    *no downsampling*
    """

    def setUp(self):
        self.qp_project: QuPathTilingProject = QuPathTilingProject(QUPATH_PATH, "r")
        self.temporary_project: QuPathTilingProject = QuPathTilingProject(
            TEMPORARY_QP_PATH, "x+"
        )
        self.temporary_project.add_image(SLIDE_PATH)
        self.temporary_project.path_classes = self.qp_project.path_classes

    def test_import_export_cycle(self):
        first_export: NDArray[np.int32] = self.qp_project.get_tile_annotation_mask(
            MaskParameter(0, (10, 10)), (450, 450)
        )
        self.temporary_project.save_mask_annotations(
            first_export, MaskParameter(0, (1000, 1000))
        )
        second_export: NDArray[np.int32] = (
            self.temporary_project.get_tile_annotation_mask(
                MaskParameter(0, (1000, 1000)), (450, 450)
            )
        )
        self.assertTrue(np.array_equal(first_export, second_export))

    def test_import_export_cycle_multichannel(self):
        first_export: NDArray[np.int32] = self.qp_project.get_tile_annotation_mask(
            MaskParameter(0, (10, 10), multichannel=True), (450, 450)
        )
        self.temporary_project.save_mask_annotations(
            first_export, MaskParameter(0, (1000, 1000), multichannel=True)
        )
        second_export: NDArray[np.int32] = (
            self.temporary_project.get_tile_annotation_mask(
                MaskParameter(0, (1000, 1000), multichannel=True), (450, 450)
            )
        )
        self.assertTrue(np.array_equal(first_export, second_export))

    def tearDown(self):
        # cleanup new QuPath project
        test_project_path: str = TEMPORARY_QP_PATH
        file: str
        for file in os.listdir(test_project_path):
            file_path: str = os.path.join(test_project_path, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)


class TestTileExport(unittest.TestCase):
    """test export related functions"""

    def setUp(self):
        self.qp_project: QuPathTilingProject = QuPathTilingProject(
            path=QUPATH_PATH, mode="r"
        )
        self.qp_project.update_image_paths(try_relative=True)
        with open(EXPECTED_MASK_PATH, "rb") as mask_file:
            expected_masks = pickle.load(mask_file)
        self.expected_singlemask: NDArray[np.uint8] = expected_masks[0]
        self.expected_multimask: NDArray[np.uint8] = expected_masks[1]

    def test_get_tile(self):
        expected_tile: NDArray[np.int_] = (np.ones((50, 50, 3)) * 255).astype(np.int_)
        tile: NDArray[np.int_] = self.qp_project.get_tile(
            0, (500, 500), (50, 50), ret_array=True
        )
        self.assertTrue(np.array_equal(expected_tile, tile))
        tile_arr: NDArray[np.int_] = self.qp_project.get_tile(
            0, (500, 500), (50, 50), ret_array=True
        )
        self.assertTrue(np.array_equal(expected_tile, tile_arr))

    def test_get_tile_annotation(self):
        # use custom annotation to know how the tiled annotation should look like
        expected_polygons: list[Polygon] = [
            Polygon([(510, 550), (550, 550), (550, 500), (510, 500)]),
            Polygon([(510, 510), (500, 510), (500, 550), (510, 550)]),
        ]
        tile_intersections: list[tuple[Polygon, str]] = (
            self.qp_project.get_tile_annotation(0, (500, 500), (50, 50))
        )
        polys: list[Polygon] = [intersection[0] for intersection in tile_intersections]
        i: int
        poly: Polygon
        self.assertTrue(len(polys) == 2)
        for i, poly in enumerate(polys):
            self.assertTrue(poly.equals(expected_polygons[i]))

    def test_get_tile_annotation_mask(self):
        single_mask: NDArray[np.int_] = self.qp_project.get_tile_annotation_mask(
            MaskParameter(0, (500, 500), multichannel=False), (50, 50)
        )
        self.assertTrue(np.array_equal(self.expected_singlemask, single_mask))
        multi_mask: NDArray[np.int_] = self.qp_project.get_tile_annotation_mask(
            MaskParameter(0, (500, 500), multichannel=True), (50, 50)
        )
        self.assertTrue(np.array_equal(self.expected_multimask, multi_mask))

    def tearDown(self):
        pass  # no files created


class TestTileImport(unittest.TestCase):
    """test import related functions"""

    def setUp(self):
        self.qp_project: QuPathTilingProject = QuPathTilingProject(
            path=QUPATH_PATH, mode="r"
        )
        self.temp_qp_project: QuPathTilingProject = QuPathTilingProject(
            TEMPORARY_QP_PATH, "x+"
        )
        self.temp_qp_project.add_image(SLIDE_PATH)
        self.temp_qp_project.path_classes = self.qp_project.path_classes
        with open(EXPECTED_MASK_PATH, "rb") as mask_file:
            expected_masks = pickle.load(mask_file)
        self.expected_singlemask: NDArray[np.uint8] = expected_masks[0]
        self.expected_multimask: NDArray[np.uint8] = expected_masks[1]

    def test_save_and_merge_annotations(self):
        # test by exporting two times size (25, 25), importing and merge those tiles,
        # export tile
        export_1: NDArray[np.int32] = self.qp_project.get_tile_annotation_mask(
            MaskParameter(0, (500, 500)), (25, 25)
        )
        export_2: NDArray[np.int32] = self.qp_project.get_tile_annotation_mask(
            MaskParameter(0, (500, 525)), (25, 25)
        )
        export_complete_area: NDArray[np.int_] = (
            self.qp_project.get_tile_annotation_mask(
                MaskParameter(0, (500, 500)), (25, 50)
            )
        )
        self.temp_qp_project.save_mask_annotations(
            export_1, MaskParameter(0, (500, 500))
        )
        self.temp_qp_project.save_mask_annotations(
            export_2, MaskParameter(0, (500, 525))
        )
        self.temp_qp_project.merge_near_annotations(0, max_dist=2)
        export_merged_area = self.temp_qp_project.get_tile_annotation_mask(
            MaskParameter(0, (500, 500)), (25, 50)
        )
        self.assertTrue(np.array_equal(export_complete_area, export_merged_area))

    def test_merge_near_annotations(self):
        # compare merged annotations to original annotations
        # no downsample!
        pass

    def tearDown(self) -> None:
        # cleanup new QuPath project
        test_project_path: str = TEMPORARY_QP_PATH
        file: str
        for file in os.listdir(test_project_path):
            file_path: str = os.path.join(test_project_path, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)


class TestTileUtils(unittest.TestCase):
    """test util functions"""

    def setUp(self):
        self.white_image_id: int = 0
        self.qp_project: QuPathTilingProject = QuPathTilingProject(
            path=QUPATH_PATH, mode="r+"
        )
        self.white_image: QuPathProjectImageEntry = self.qp_project.images[
            self.white_image_id
        ]
        self.original_path_classes: tuple[QuPathPathClass, ...] = (
            self.qp_project.path_classes
        )

    def test_update_path_classes(self):
        new_path_classes: tuple[QuPathPathClass, ...] = (
            QuPathPathClass("class_a"),
            QuPathPathClass("class_b"),
            QuPathPathClass("class_c"),
        )
        class_dict: dict[int, QuPathPathClass] = {
            0: new_path_classes[0],
            1: new_path_classes[1],
            2: new_path_classes[2],
        }

        self.qp_project.path_classes = new_path_classes
        self.assertEqual(self.qp_project.path_classes, new_path_classes)
        self.assertEqual(self.qp_project._class_dict, class_dict)

    def test_update_img_annotation_dict(self):
        # use own random annotations
        self.assertEqual(self.qp_project.img_annotation_dict, {})
        self.qp_project._update_img_annotation_dict(self.white_image_id)
        search_tree: STRtree
        search_tree_dict: dict[int, tuple[int, str]]
        search_tree, search_tree_dict = self.qp_project.img_annotation_dict[
            self.white_image_id
        ]
        self.assertIsInstance(search_tree, STRtree)
        self.assertEqual(
            len(search_tree_dict),
            len(self.qp_project.images[self.white_image_id].hierarchy),
        )

    def tearDown(self):
        self.qp_project.path_classes = self.original_path_classes


if __name__ == "__main__":
    print(f"Run test on {moth}")
    unittest.main()
