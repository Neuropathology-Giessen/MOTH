import os, shutil, re
import cv2
from qp_tiling.tile_class import QuPathOperations

TEMP_QUPATH_DIR = os.path.join('..', 'data', 'temp_project')
QUPATH_DIR = os.path.join('..', '..', 'test', 'test_projects', 'qp_project')
QUPATH_IMAGE_DIR = '/workspaces/GBM_QuPath_tiles/test/test_projects/slides'
PREDICTION_DIR = '/workspaces/GBM_QuPath_tiles/workflow/data/export/white-4096/labels/'
SLIDE_PATH = '/workspaces/GBM_QuPath_tiles/test/test_projects/slides/white-4096.tif'

# clear temp proj
shutil.rmtree(TEMP_QUPATH_DIR)

# create new project for saving predictions
if not os.path.isdir(TEMP_QUPATH_DIR):
    os.mkdir(TEMP_QUPATH_DIR)
qp_project = QuPathOperations(QUPATH_DIR)    
temp_qp_proj = QuPathOperations(TEMP_QUPATH_DIR, mode = 'x+')
temp_qp_proj.update_path_classes(qp_project.path_classes)
temp_qp_proj.add_image(SLIDE_PATH)

# save and merge tiles
for filename in os.listdir(PREDICTION_DIR):
    # get meta
    location_x = int(re.search('(?<=x=)(\d+)', filename).group(0))
    location_y = int(re.search('(?<=y=)(\d+)', filename).group(0))
    # size = tuple(map(int, re.search('(?<=size=\()(\d+, ?\d+)', filename).group(0).split(',')))
    prediction = cv2.imread(os.path.join(PREDICTION_DIR, filename), cv2.IMREAD_UNCHANGED)
    temp_qp_proj.save_mask_annotations(0, prediction, (location_x, location_y))
temp_qp_proj.merge_near_annotations(0, 2)
temp_qp_proj.save()