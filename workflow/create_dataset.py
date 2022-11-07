import os, pickle

from  workflow.src.saved_tile_dataset import QPDataset
from qp_tiling.tile_class import QuPathOperations
import numpy as np
import cv2
from progress.bar import Bar

# config
QUPATH_DIR = os.path.join('..', '..', 'test', 'test_projects', 'qp_project')
DATA_DIR = os.path.join('..', 'data')
SAVE_DIR = os.path.join(DATA_DIR, 'export')
QUPATH_IMAGE_DIR = '/workspaces/GBM_QuPath_tiles/test/test_projects/slides'
EXPORT_IMG_IDS = [0] # numpy indexing

TILE_SIZE = (250, 250) # (width, height)
DOWNSAMPLE_LEVEL = 0

# constants
DOWNSAMPLE_FACTOR = 2 ** DOWNSAMPLE_LEVEL
LEVEL_0_TILE_SIZE = tuple(map(lambda size: size * DOWNSAMPLE_FACTOR, TILE_SIZE))
LEVEL_0_WIDTH = LEVEL_0_TILE_SIZE[0]
LEVEL_0_HEIGHT = LEVEL_0_TILE_SIZE[1]


# extract tiles
qp_project = QuPathOperations(QUPATH_DIR)
img_meta = np.array(qp_project.images)[EXPORT_IMG_IDS]
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

for qupath_img in img_meta:
    img_id = int(qupath_img.entry_id) - 1
    # name to save tiles with
    save_tile_img_name = qupath_img.image_name.split('.')[0]
    # create subdirectory for each tiled image 
    save_tiles_dir = os.path.join(SAVE_DIR, save_tile_img_name)
    if not os.path.exists(save_tiles_dir):
        os.mkdir(save_tiles_dir)
        os.mkdir(os.path.join(save_tiles_dir, 'labels'))

    ### check for existing exports
    if len(os.listdir(save_tiles_dir)) > 1:
        print(save_tile_img_name + ' skipped: already exported')
        continue
    ###

    print(save_tile_img_name + ': start export')
    y_steps = int(qupath_img.height / LEVEL_0_HEIGHT)
    with Bar('tile export: ' + save_tile_img_name, max = y_steps, suffix = '%(percent)d%%') as bar:
        for y_step in range(y_steps):
            location_y = y_step * LEVEL_0_TILE_SIZE[1]

            for x_step in range(int(qupath_img.width / LEVEL_0_WIDTH)):
                location_x = x_step * LEVEL_0_TILE_SIZE[0]
                tilename = f'{save_tile_img_name}[x={location_x},y={location_y},size={TILE_SIZE}].tif'
                tile_path_name = os.path.join(save_tiles_dir, tilename)
                tile_mask_path_name = os.path.join(save_tiles_dir, 'labels',
                                                tilename.split('.')[0] + '_label.tif')

                tile = qp_project.get_tile(QUPATH_IMAGE_DIR, img_id, (location_x, location_y), TILE_SIZE, 
                                        downsample_level = DOWNSAMPLE_LEVEL)
                tile = np.array(tile)[:, :, :3] # drop 4th dimension

                tile_mask = qp_project.get_tile_annot_mask(img_id, (location_x, location_y), TILE_SIZE,
                                                        downsample_level = DOWNSAMPLE_LEVEL)
                
                # write tile
                cv2.imwrite(tile_path_name, tile)
                cv2.imwrite(tile_mask_path_name, tile_mask)
            bar.next()


# create and save dataset
dataset = QPDataset([os.path.join(SAVE_DIR, img) for img in os.listdir(SAVE_DIR)])
dataset_dir = os.path.join(DATA_DIR, 'dataset')
if not os.path.isdir(dataset_dir):
    os.mkdir(dataset_dir)
with open(os.path.join(dataset_dir, 'dataset.pkl'), 'wb') as dump_file:
    pickle.dump(dataset, dump_file)