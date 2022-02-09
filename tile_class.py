from __future__ import annotations
from typing import Sequence
from paquo.projects import QuPathProject

import numpy as np
from shapely.geometry import Polygon
from openslide import OpenSlide
import PIL


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