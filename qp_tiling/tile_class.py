from paquo.projects import QuPathProject

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, CAP_STYLE, JOIN_STYLE
from shapely.strtree import STRtree
from shapely.validation import make_valid
from shapely.ops import unary_union
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
        self._class_dict = {}
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
        hier_data = slide.hierarchy.annotations
        location_x, location_y = location
        width, height = size
        polygon_tile = Polygon(([location_x, location_y], [location_x + width - 1, location_y], [location_x + width - 1, location_y + height - 1], [location_x, location_y + height - 1]))
        tile_intersections = []

        if img_id in self.img_annot_dict:
            ann_tree, index_and_class = self.img_annot_dict[img_id]
            near_polys = [poly for poly in ann_tree.query(polygon_tile)]
            near_poly_classes = [index_and_class[id(poly)][1] for poly in near_polys]
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
            for annot in hier_data:
                if not annot.path_class:
                    continue
                annot_class = annot.path_class.id
                polygon_annot = annot.roi
                img_ann_list.append((polygon_annot, annot_class)) # save all Polygons in list to create a cache.

                intersection = polygon_annot.intersection(polygon_tile)
                if intersection.is_empty:
                    continue

                filter_bool = (not class_filter) or (annot_class in class_filter) or (self._inverse_class_dict[annot_class] in class_filter)  

                if filter_bool and isinstance(intersection, MultiPolygon): # filter applies and polygon is a multipolygon
                    for inter in intersection.geoms:
                        tile_intersections.append((annot_class, inter))

                elif filter_bool: # filter applies and is not a multipolygon
                    tile_intersections.append((annot_class, intersection))

            img_ann_transposed = np.array(img_ann_list, dtype = object).transpose() # [list(rois), list(annotation_classes)]
            class_by_id = dict((id(ann_poly), (i, img_ann_transposed[1][i])) for i, ann_poly in enumerate(img_ann_transposed[0]))
            img_ann_tree = STRtree(img_ann_transposed[0])
            self.img_annot_dict[img_id] = (img_ann_tree, class_by_id)

        return tile_intersections

    
    def get_tile_annot_mask(self, img_id, location, size, downsample_level = 0, multilabel = False, class_filter = None):
        ''' get tile annotations mask between (x|y) and (x + size| y + size)

        Parameters:

            img_id:     id of image to operate
            location:   (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
            size:       (width, height) tuple containing the tile size
            downsample_level: level for downsampling
            multilabel: if True annotation mask contains boolean image for each class ([num_classes, width, height])
            class_filter:   list of annotationclass names to filter by
                            if None no filter is applied

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
            sorted_intersections = sorted(tile_intersections, key = lambda tup: Polygon(tup[1].exterior).area, reverse=True)
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
            interiors = [int_coords(poly.coords) for poly in scale_inter.interiors]
                
            if multilabel:
                cv2.fillPoly(annot_mask[class_num], exteriors, 1)
                cv2.fillPoly(annot_mask[class_num], interiors, 0)

            else:
                cv2.fillPoly(annot_mask, exteriors, class_num)
                cv2.fillPoly(annot_mask, interiors, 0)

        return annot_mask


    def merge_near_annotations(self, img_id, max_dist):
        ''' merge nearby annotations with equivalent annotation class

        Parameters:
            img_id:     id of image to operate
            max_dist:   maximal distance between annotations to merge
        '''
        hierarchy = self.images[img_id].hierarchy
        annotations = hierarchy.annotations
        self.update_img_annot_dict(img_id)
        already_merged = [] # save merged indicies
        ann_tree, class_by_id = self.img_annot_dict[img_id]

        for index, annot in enumerate(annotations):
            if index in already_merged:
                annotations.discard(annot)
                continue
            annot_poly = annot.roi
            annot_poly_class = annot.path_class
            annot_poly_buffered = annot_poly.buffer(max_dist)

            annotations_to_merge = [annot_poly_buffered]

            nested_annotations = [annot_poly_buffered]
            while len(nested_annotations) > 0:
                annot_poly_buffered = nested_annotations.pop(0)
                near_polys = [poly for poly in ann_tree.query(annot_poly_buffered)]
                near_poly_index_and_classes = [class_by_id[id(poly)] for poly in near_polys]

                while len(near_polys) > 0:
                    near_poly = near_polys.pop(0)
                    near_poly_index, near_poly_annotation_class = near_poly_index_and_classes.pop(0)

                    if near_poly_index in already_merged:
                        continue
                    if index == near_poly_index: # tree query will always return the polygon from the same annotation
                        continue
                    if annot_poly_class != near_poly_annotation_class:
                        continue

                    near_poly_buffered = near_poly.buffer(max_dist)
                    intersects = near_poly_buffered.intersects(annot_poly_buffered)
                    if intersects:     
                        annotations_to_merge.append(near_poly_buffered)
                        nested_annotations.append(near_poly_buffered)
                        already_merged.append(near_poly_index)

            if len(annotations_to_merge) > 1:
                merged_annot = unary_union(annotations_to_merge).buffer(-max_dist)
                hierarchy.add_annotation(merged_annot, annot_poly_class)
                annotations.discard(annot)


    def update_img_annot_dict(self, img_id):
        ''' update annotation rois tree for faster shapely queries

        Parameters:
            img_id:     id of image to operate
        '''
        slide = self.images[img_id]
        annotations = slide.hierarchy.annotations
        img_ann_list = [(annot.roi, annot.path_class.id) for annot in annotations]

        img_ann_transposed = np.array(img_ann_list, dtype = object).transpose() # [list(rois), list(annot_classes)]
        class_by_id = dict((id(ann_poly), (i, img_ann_transposed[1][i])) for i, ann_poly in enumerate(img_ann_transposed[0]))
        img_ann_tree = STRtree(img_ann_transposed[0])
        self.img_annot_dict[img_id] = (img_ann_tree, class_by_id)


    def save_mask_annotations(self, img_id, annot_mask, location = (0,0), downsample_level = 0, min_polygon_area = 0, multilabel = False):
        ''' saves a mask as annotations to QuPath

        Parameters:
            img_id:             id of image to operate
            annot_mask:         mask with annotations
            location:           (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
            downsample_level:   level for downsampling
            min_polygon_area:   minimal area for polygons to be saved
            multilabel:         if True annotation mask contains boolean image for each class ([num_classes, width, height])
        '''
        slide = self.images[img_id]
        poly_annot_list = self.label_img_to_polys(annot_mask, downsample_level, min_polygon_area, multilabel)
        for annot_poly, annot_class in poly_annot_list:
            poly_to_add = shapely.affinity.translate(annot_poly, location[0], location[1])
            slide.hierarchy.add_annotation(poly_to_add, self._class_dict[annot_class])


    @classmethod
    def label_img_to_polys(cls, label_img, downsample_level = 0, min_polygon_area = 0, multilabel = False):
        ''' convert label mask to list of Polygons

        Parameters:
            label_img:          mask [H, W] with values between 0 and highest label class
            downsample_level:   level for downsampling
            min_polygon_area:   minimal area for polygons to be saved
            multilabel:         if True annotation mask contains boolean image for each class ([num_classes, width, height])

        Returns:
            poly_labels: list of Polygon and label tuple [(polygon, label), ...]
        '''
        downsample_factor = 2 ** downsample_level
        label_img = label_img.astype(np.uint8)
        poly_labels = []

        if multilabel:
            iter_range = range(len(label_img))
        else:
            iter_range = range(1, np.max(label_img)+1)

        for i in iter_range:
            if multilabel:
                it_img = label_img[i]
                i += 1
            else:
                label_img_copy = label_img.copy()

                it_img = np.where(label_img_copy == i, 1, 0).astype(np.uint8)
                
            contours, hierarchy = cv2.findContours(it_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contours = list(map(np.squeeze, contours))
            if len(contours) == 0:
                continue
            # iterate hierarchy
            next_id = 0                                 # start with first contour
            while next_id != -1:
                current_id = next_id                    # previous next id is now our current id
                next_id = hierarchy[0][current_id][0]
                child_id = hierarchy[0][current_id][2]

                if len(contours[current_id]) < 3:       # linear ring possible?
                    continue
                
                if child_id == -1:
                    poly = Polygon(contours[current_id])
                else:
                    holes = []
                    hole_poly = Polygon(contours[child_id]).buffer(
                        -1,
                        join_style= JOIN_STYLE.mitre,
                        cap_style= CAP_STYLE.square
                    )
                    holes.append(list(map(
                        lambda coords: np.array(coords).round().astype(np.int32),
                        hole_poly.exterior.coords
                    )))
                    next_child_id = hierarchy[0][child_id][0]
                    while next_child_id != -1:
                        hole_poly = Polygon(contours[child_id]).buffer(
                            -1,
                            join_style= JOIN_STYLE.mitre,
                            cap_style= CAP_STYLE.square
                        )
                        holes.append(list(map(
                            lambda coords: np.array(coords).round().astype(np.int32),
                            hole_poly.exterior.coords
                        )))
                        next_child_id = hierarchy[0][next_child_id][0]
                    poly = Polygon(contours[current_id], holes)

                poly = shapely.affinity.scale(poly, xfact = 1*downsample_factor, yfact = 1*downsample_factor, origin = (0,0))
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.area > min_polygon_area:
                    poly_labels.append((poly, i))

        return poly_labels