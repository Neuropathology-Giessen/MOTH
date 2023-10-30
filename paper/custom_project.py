""" Script to create an empty qupath project with a white 256x256 px image """
import os
import shutil
from pathlib import Path

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from paquo.projects import QuPathProject
from tifffile import imwrite

QP_PROJECT_PATH = Path("qp_project")
QP_MODE = "x"

# clear existing project
if os.path.isdir(QP_PROJECT_PATH):
    shutil.rmtree(QP_PROJECT_PATH)

# create new QuPath project
qp_project = QuPathProject(QP_PROJECT_PATH, QP_MODE)

# create blank image
slide_path: Path = QP_PROJECT_PATH / Path("Slides", "white.tiff")
slide_path.parent.mkdir(parents=True, exist_ok=True)
image_data: NDArray[float64] = np.ones((256, 256, 3)) * 255
imwrite(slide_path, image_data)

# add image to project
qp_project.add_image(slide_path)
qp_project.save()
