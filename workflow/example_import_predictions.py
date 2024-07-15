"""Example script for importing predictions into a QuPath project."""

import os
import re
import shutil
from typing import Optional

import cv2
import numpy as np

from moth.projects import MaskParameter, QuPathTilingProject

TEMP_QUPATH_DIR = os.path.join("data", "temp_project")
QUPATH_DIR = os.path.join("..", "test", "test_projects", "qp_project")
# QUPATH_IMAGE_DIR = '/workspaces/GBM_QuPath_tiles/test/test_projects/slides'
PREDICTION_DIR = "data/export/white-4096/labels/"
SLIDE_PATH = "../test/test_projects/slides/white-4096.tif"

# clear temp proj
shutil.rmtree(TEMP_QUPATH_DIR)

# create new project for saving predictions
if not os.path.isdir(TEMP_QUPATH_DIR):
    os.mkdir(TEMP_QUPATH_DIR)
qp_project = QuPathTilingProject(QUPATH_DIR)
temp_qp_proj = QuPathTilingProject(TEMP_QUPATH_DIR, mode="x+")
temp_qp_proj.path_classes = qp_project.path_classes
temp_qp_proj.add_image(SLIDE_PATH)

# save and merge tiles
for filename in os.listdir(PREDICTION_DIR):
    # get meta
    location_x_match: Optional[re.Match[str]] = re.search(r"(?<=x=)(\d+)", filename)
    location_y_match: Optional[re.Match[str]] = re.search(r"(?<=y=)(\d+)", filename)
    if not location_x_match or not location_y_match:
        continue
    location_x = int(location_x_match.group(0))
    location_y = int(location_y_match.group(0))
    # size = tuple(map(int, re.search('(?<=size=\()(\d+, ?\d+)', filename).group(0).split(',')))
    prediction = np.array(
        cv2.imread(  # pylint: disable=no-member
            os.path.join(PREDICTION_DIR, filename),
            cv2.IMREAD_UNCHANGED,  # pylint: disable=no-member
        ),
        dtype=np.uint8,
    )
    temp_qp_proj.save_mask_annotations(
        prediction, MaskParameter(0, (location_x, location_y))
    )
temp_qp_proj.merge_near_annotations(0, 2)
temp_qp_proj.save()
