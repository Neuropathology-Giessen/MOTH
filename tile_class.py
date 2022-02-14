from __future__ import annotations
from paquo.projects import QuPathProject

import numpy as np
from shapely.geometry import Polygon
from openslide import OpenSlide
import PIL
import cv2

import os


class QuPathOperations(QuPathProject):
    def __init__(self, path, mode = 'r'):
        """ load or create a new qupath project

        Parameters:

            path:
                path to `project.qpproj` file, or its parent directory
            mode:
                'r' --> readonly, error if not there
                'r+' --> read/write, error if not there
                'a' = 'a+' --> read/write, create if not there, append if there
                'w' = 'w+' --> read/write, create if not there, truncate if there
                'x' = 'x+' --> read/write, create if not there, error if there

        """
        super().__init__(path, mode)


    def set_path_classes(self, path_classes):
        self.path_classes = path_classes


    def get_tile(self, img_id, location_x, location_y, size):
        ''' get tile between (x|y) and (x + size| y + size)

        Parameters:

            img_id: id of image to operate
            location_x: x coordinate for tile begin
            location_y: y coordinate for tile begin
            size:   size of tile (tile shape = (size, size))       

        Returns:

            tile:   tile image 
        '''
        slide = self.images[img_id]
        with OpenSlide(os.path.join("Qupath_data", "Slides zum Training", slide.image_name)) as slide_data:
            tile = slide_data.read_region((location_x, location_y), 0, (size, size))
        return(tile)


    def get_tile_annot(self, img_id, location_x, location_y, size):
        ''' get tile annotations between (x|y) and (x + size| y + size)

        Parameters:

            img_id: id of image to operate
            location_x: x coordinate for tile begin
            location_y: y coordinate for tile begin
            size:   size of tile (tile shape = (size, size))       

        Returns:

            tile_intersections: list of annotations in tile
        '''
        slide = self.images[img_id]
        hier_data = slide.hierarchy.to_geojson()
        polygon_tile = Polygon(([location_x, location_y], [location_x + size, location_y], [location_x + size, location_y + size], [location_x, location_y + size]))
        tile_intersections = []

        for geojson in hier_data:
            try:
                annot_class = geojson["properties"]["classification"]["name"]
            except KeyError:
                continue
            poly_data = geojson['geometry']['coordinates']
            try:    
                polygon_annot = Polygon(poly_data[0], poly_data[1:]) if len(poly_data) > 1 else Polygon(poly_data[0])
            except (ValueError, AssertionError) as e:
                poly_data_squeezed = []
                for i in range(len(poly_data)):
                    poly_data_squeezed.append(np.array(poly_data[i]).squeeze())
                try:
                    polygon_annot = Polygon(poly_data_squeezed[0], poly_data_squeezed[1:]) if len(poly_data_squeezed) > 1 else Polygon(poly_data_squeezed[0])
                except ValueError:
                    polygon_annot = Polygon(poly_data_squeezed[0][0], poly_data_squeezed[1:]) if len(poly_data_squeezed) > 1 else Polygon(poly_data_squeezed[0][0])
            
            intersection = polygon_annot.intersection(polygon_tile)
            if not intersection.is_empty:
                tile_intersections.append((annot_class, intersection))

        return(tile_intersections)


    def save_tile(self, filename: str, img, label_img):
        ''' save tile with annotations in QuPathProject

        Parameters:

            filename (str): name to save tile in project
            img:            tile to save
            label_img:      annotations of the tile
        '''
        slides_path = '{}/slides/'.format(os.path.split(self.path)[0])
        class_dict = {}
        for i, ann in enumerate(self.path_classes):
            class_dict[i] = ann
        if len(os.listdir(os.path.split(self.path)[0])) == 0:
            self.save()
            os.mkdir(slides_path)
        img_path = '{}/slides/{}.tif'.format(os.path.split(self.path)[0], filename)
        img.save(img_path)
        slide = self.add_image(img_path)

        poly_annot_list = self.label_img_to_polys(label_img)
        for poly, annot in poly_annot_list:
            slide.hierarchy.add_annotation(poly, path_class= class_dict[annot])


    @classmethod
    def label_img_to_polys(cls, label_img):
        ''' convert label mask to list of Polygons

        Parameters:

            label_img: mask [H, W] with values between 0 and highest label class

        Returns:

            poly_labels: list of Polygon and label tuple [(polygon, label), ...]
        '''
        label_img = label_img.astype(np.uint8)
        poly_labels = []
        for i in range(1, np.max(label_img)+1):
            label_img_copy = label_img.copy()
            it_img = np.where(label_img_copy == i, 1, 0).astype(np.uint8)
            contours, _ = cv2.findContours(it_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                continue
            contours = map(np.squeeze, contours)
            polygons = map(Polygon, contours)
            polygons = map(lambda x: x.simplify(0), polygons)
            poly_label = map(lambda poly: (poly, i), polygons)
            
            poly_labels.extend(
                list(poly_label)
            )
            
        return poly_labels