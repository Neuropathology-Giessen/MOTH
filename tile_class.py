from paquo.projects import QuPathProject

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.strtree import STRtree
import shapely
from openslide import OpenSlide
import cv2

import os


class QuPathOperations(QuPathProject):
    def __init__(self, path, mode = 'r'):
        ''' load or create a new qupath project

        Parameters:

            path:
                path to `project.qpproj` file, or its parent directory
            mode:
                'r' --> readonly, error if not there
                'r+' --> read/write, error if not there
                'a' = 'a+' --> read/write, create if not there, append if there
                'w' = 'w+' --> read/write, create if not there, truncate if there
                'x' = 'x+' --> read/write, create if not there, error if there

        '''
        super().__init__(path, mode)
        self._class_dict = {}
        for i, ann in enumerate(self.path_classes):
            self._class_dict[i] = ann
        self._inverse_class_dict = {value.id: key for key, value in self._class_dict.items()}
        self.img_annot_dict = {}


    def update_path_classes(self, path_classes):
        ''' update the annotation classes and annotation dictionaries of the project
        
        Parameters:
            path_classes: annotation classes to set
        '''
        self.path_classes = path_classes
        for i, ann in enumerate(self.path_classes):
            self._class_dict[i] = ann
        self._inverse_class_dict = {value.id: key for key, value in self._class_dict.items()}


    def get_tile(self, img_dir, img_id, location, size, downsample_level = 0):
        ''' get tile starting at x|y (slide level 0) with given size  

        Parameters:

            img_dir:    directory containing the image
            img_id:     id of image to operate
            location:   (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
            size:       (width, height) tuple containing the tile size
            downsample_level: level for downsampling

        Returns:
            tile:   tile image 
        '''
        slide = self.images[img_id]
        with OpenSlide(os.path.join(img_dir, slide.image_name)) as slide_data:
            tile = slide_data.read_region(location, downsample_level, size)
        return(tile)


    def get_tile_annot(self, img_id, location, size, class_filter = None):
        ''' get tile annotations between (x|y) and (x + size| y + size)

        Parameters:

            img_id:     id of image to operate
            location:   (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
            size:       (width, height) tuple containing the tile size
            class_filter:   list of annotationclass names or ids to filter by
                            if None no filter is applied

        Returns:
            tile_intersections: list of annotations (shapely polygons) in tile 
        '''
        slide = self.images[img_id]
        hier_data = slide.hierarchy.to_geojson()
        location_x, location_y = location
        width, height = size
        polygon_tile = Polygon(([location_x, location_y], [location_x + width, location_y], [location_x + width, location_y + height], [location_x, location_y + height]))
        tile_intersections = []

        if img_id in self.img_annot_dict:
            ann_tree, class_by_id = self.img_annot_dict[img_id]
            near_polys = [o for o in ann_tree.query(polygon_tile)]
            near_poly_classes = [class_by_id[id(o)] for o in near_polys]
            for poly, annot_class in zip(near_polys, near_poly_classes):
                intersection = poly.intersection(polygon_tile)
                if intersection.is_empty:
                    continue
                
                filter_bool = (not class_filter) or (annot_class in class_filter) or (self._inverse_class_dict[annot_class] in class_filter)
                if filter_bool and isinstance(intersection, MultiPolygon): # filter applies and polygon is a multipolygon
                    for inter in intersection.geoms:                        
                        tile_intersections.append((annot_class, inter))
                
                elif filter_bool: # filter applies and is not a multipolygon
                    tile_intersections.append((annot_class, intersection))

        else:
            img_ann_list = []
            for geojson in hier_data: # only use polygons with classes
                if 'properties' in geojson.keys() and 'classification' in geojson['properties'].keys():
                    annot_class = geojson['properties']['classification']['name']
                else:
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
                    except ValueError: # multiple annotation areas for same annotation
                        try:
                            polygon_annot = Polygon(poly_data_squeezed[0][0], poly_data_squeezed[1:]) if len(poly_data_squeezed) > 1 else Polygon(poly_data_squeezed[0][0])
                        except: # incorrect data, continue with next annotation
                            continue

                img_ann_list.append((annot_class, polygon_annot)) # save all Polygons in list to create a cache. Now the json only has to be converted ones per image

                intersection = polygon_annot.intersection(polygon_tile)
                if intersection.is_empty:
                    continue

                filter_bool = (not class_filter) or (annot_class in class_filter) or (self._inverse_class_dict[annot_class] in class_filter)  

                if filter_bool and isinstance(intersection, MultiPolygon): # filter applies and polygon is a multipolygon
                    for inter in intersection.geoms:
                        tile_intersections.append((annot_class, inter))

                elif filter_bool: # filter applies and is not a multipolygon
                    tile_intersections.append((annot_class, intersection))

            img_ann_transposed = np.array(img_ann_list).transpose() # [list(annot_classes), list(polygons)]
            class_by_id = dict((id(ann_poly), img_ann_transposed[0][i]) for i, ann_poly in enumerate(img_ann_transposed[1]))
            img_ann_tree = STRtree(img_ann_transposed[1])
            self.img_annot_dict[img_id] = (img_ann_tree, class_by_id)

        return tile_intersections

    
    def get_tile_annot_mask(self, img_id, location, size, downsample_level = 0, multilabel = False, class_filter = None):
        ''' get tile annotations mask between (x|y) and (x + size| y + size)

        Parameters:

            img_id:     id of image to operate
            location:   (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
            size:       (width, height) tuple containing the tile size
            multilabel: if True annotation mask contains boolean image for each class ([num_classes, width, height])
            class_filter:   list of annotationclass names to filter by
                            if None no filter is applied
            downsample_level: level for downsampling

        Returns:
            annot_mask: mask [height, width] with an annotation class for each pixel
                        or [num_class, height, width] for multilabels
                        background class is ignored for multilabels ([0, height, width] shows mask for the first annotation class)
        '''
        location_x, location_y = location
        width, height = size
        downsample_factor = 2 ** downsample_level 
        level_0_size = map(lambda x: x* downsample_factor, size) # level_0_size needed to get all Polygons in downsampled area
        tile_intersections = self.get_tile_annot(img_id, location, level_0_size, class_filter)

        if multilabel:
            num_classes = len(self.path_classes) -1 
            annot_mask = np.zeros((num_classes, height, width))

        else:
            # sort intersections descending by area. Now we can not accidentally overwrite polys with other poly holes
            sorted_intersections = sorted(tile_intersections, key = lambda tup: tup[1].exterior.area, reverse=True)
            tile_intersections = sorted_intersections
            annot_mask = np.zeros((height, width))
        

        for inter_class, intersection in tile_intersections:
            class_num = self._inverse_class_dict[inter_class]
            if multilabel: # first class should be on the lowest level for multilabels
                class_num -= 1

            trans_inter = shapely.affinity.translate(intersection, location_x * -1, location_y * -1)
            # apply downsampling by scaling the Polygon down
            scale_inter = shapely.affinity.scale(trans_inter, xfact = 1/downsample_factor, yfact = 1/downsample_factor, origin = (0,0)) 

            int_coords = lambda coords: np.array(coords).round().astype(np.int32)
            exteriors = [int_coords(scale_inter.exterior.coords)]
            interiors = [int_coords(poly) for poly in scale_inter.interiors]
                
            if multilabel:
                cv2.fillPoly(annot_mask[class_num], exteriors, 1)
                cv2.fillPoly(annot_mask[class_num], interiors, 0)

            else:
                cv2.fillPoly(annot_mask, exteriors, class_num)
                cv2.fillPoly(annot_mask, interiors, 0)

        return annot_mask


    def save_tile(self, filename: str, img, label_img):
        ''' save tile with annotations in QuPathProject

        Parameters:

            filename (str): name to save tile in project
            img:            tile to save
            label_img:      annotations of the tile
        '''
        slides_path = '{}/slides/'.format(os.path.split(self.path)[0])
        # check if QuPath project is initilazed, if not save project before filling folder
        if len(os.listdir(os.path.split(self.path)[0])) == 0:
            self.save()
            os.mkdir(slides_path)
        img_path = '{}/slides/{}.tif'.format(os.path.split(self.path)[0], filename)
        img.save(img_path)
        slide = self.add_image(img_path)

        poly_annot_list = self.label_img_to_polys(label_img)
        for poly, annot in poly_annot_list:
            slide.hierarchy.add_annotation(poly, path_class= self._class_dict[annot])


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