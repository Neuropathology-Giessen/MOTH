""" Script to create an empty qupath project with a white 256x256 px image """

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from paquo.classes import QuPathPathClass
from paquo.hierarchy import QuPathPathObjectHierarchy
from paquo.projects import QuPathProject
from shapely import Polygon
from tifffile import imwrite

QP_PROJECT_PATH = Path("custom_time_project")
QP_MODE = "x"
MOTH_DATA_PATH = Path("../moth/moth_data.json")

# clear existing project
if os.path.isdir(QP_PROJECT_PATH):
    shutil.rmtree(QP_PROJECT_PATH)

# create new QuPath project
qp_project = QuPathProject(QP_PROJECT_PATH, QP_MODE)

# create blank image
slide_path: Path = QP_PROJECT_PATH / Path("Slides", "white.tiff")
slide_path_2: Path = QP_PROJECT_PATH / Path("Slides", "white2.tiff")
slide_path.parent.mkdir(parents=True, exist_ok=True)
image_data: NDArray[float64] = np.ones((256, 256, 3)) * 255
imwrite(slide_path, image_data, shape=(256, 256, 3))
imwrite(slide_path_2, image_data, shape=(256, 256, 3))

# add image to project
qp_project.add_image(slide_path)
qp_project.add_image(slide_path_2)
tumor_path_class = QuPathPathClass("Tumor")
qp_project.path_classes = (tumor_path_class,)

# add moth polygon
image_hierarchy: QuPathPathObjectHierarchy = qp_project.images[0].hierarchy
with open(MOTH_DATA_PATH, "r") as moth_data_file:
    moth_data = json.load(moth_data_file)
    moth = Polygon(
        tuple(
            map(
                lambda coordinate: (int(coordinate[0]), int(coordinate[1])),
                moth_data.get("coordinates")[0],
            )
        )
    )
    image_hierarchy.add_annotation(moth, tumor_path_class)

qp_project.save()

subprocess_command: list[str] = [
    "QuPath",
    "script",
    "-s",
    "-p",
    str(QP_PROJECT_PATH / Path("project.qpproj")),
    "-i",
    "white.tiff",
    "create_timesproject_rois.groovy",
]
try:
    subprocess.run(subprocess_command, check=True)
except FileNotFoundError as err:
    print("Failed to run the Groovy script. Retrying with QuPath added to PATH")
    qupath_exec_path: str = str(Path(os.environ["PAQUO_QUPATH_DIR"], "bin"))
    os.environ["PATH"] += os.pathsep + qupath_exec_path
    subprocess.run(["chmod", "u+x", str(Path(qupath_exec_path, "QuPath"))], check=True)
    print("Added QuPath to PATH and made it executable")
    subprocess.run(subprocess_command, check=True)
