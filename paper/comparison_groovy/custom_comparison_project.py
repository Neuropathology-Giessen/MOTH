""" Script to create an empty qupath project with a white 256x256 px image """

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from paquo.classes import QuPathPathClass
from paquo.projects import QuPathProject
from tifffile import imwrite

QP_PROJECT_PATH = Path("comp_project")
QP_MODE = "x"

# clear existing project
if os.path.isdir(QP_PROJECT_PATH):
    shutil.rmtree(QP_PROJECT_PATH)

# create new QuPath project
qp_project = QuPathProject(QP_PROJECT_PATH, QP_MODE)

# create blank image
slide_path: Path = QP_PROJECT_PATH / Path("Slides", "white.tiff")
slide_path.parent.mkdir(parents=True, exist_ok=True)
image_data: NDArray[float64] = np.ones((32, 32, 3)) * 255
imwrite(slide_path, image_data, shape=(32, 32, 3))


# add image to project
qp_project.add_image(slide_path)
tumor_path_class = QuPathPathClass("Tumor")
qp_project.path_classes = (tumor_path_class,)


qp_project.save()

subprocess_command: list[str] = [
    "QuPath",
    "script",
    "-s",
    "-p",
    str(QP_PROJECT_PATH / Path("project.qpproj")),
    "-i",
    "white.tiff",
    "create_compproject_rois.groovy",
]
try:
    subprocess.run(subprocess_command, check=True)
except FileNotFoundError as err:
    print("Failed to run the Groovy script")
    qupath_exec_path: str = str(Path(os.environ["PAQUO_QUPATH_DIR"], "bin"))
    os.environ["PATH"] += os.pathsep + qupath_exec_path
    subprocess.run(["chmod", "u+x", str(Path(qupath_exec_path, "QuPath"))], check=True)
    print("Added QuPath to PATH and made it executable")
    subprocess.run(subprocess_command, check=True)
