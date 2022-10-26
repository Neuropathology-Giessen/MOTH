import pathlib
import platform
from typing import List, Dict, Tuple, Union, Literal, Iterable, Optional

import numpy as np
import cv2
import openslide
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.strtree import STRtree
from shapely.ops import unary_union
from PIL.Image import Image

from paquo.projects import QuPathProject
from paquo.classes import QuPathPathClass
from paquo.images import QuPathProjectImageEntry
from paquo.hierarchy import PathObjectProxy
from mothi.utils import label_img_to_polys, _round_polygon

# import openSlide (https://openslide.org/api/python/#installing)
OPENSLIDE_PATH = 'C:\\Program Files\\openslide-win64-20220811\\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

ProjectIOMode = Literal["r", "r+", "w", "w+", "a", "a+", "x", "x+"]

class QuPathTilingProject(QuPathProject):
    ''' load or create a new qupath project

        Parameters
        ----------
        path:
            path to `project.qpproj` file, or its parent directory
        mode:
            'r' --> readonly, error if not there
            'r+' --> read/write, error if not there
            'a' = 'a+' --> read/write, create if not there, append if there
            'w' = 'w+' --> read/write, create if not there, truncate if there
            'x' = 'x+' --> read/write, create if not there, error if there
        '''
    def __init__(self,
                 path: Union[str, pathlib.Path],
                 mode: ProjectIOMode  = 'r'):
        super().__init__(path, mode)
        self._class_dict: Dict[int, QuPathPathClass] = {}
        self._inverse_class_dict: Dict[str, int] = {}

        for i, ann in enumerate(self.path_classes):
            self._class_dict[i] = ann
        self._inverse_class_dict = {
            value.id: key for key, value in self._class_dict.items()
        }

        ## create dictonary to handle one Strtree per image
        # {img_id: (annotationTree, {roi_id: (annotation_id, path_class)})}
        # each image has an own query tree
        # roi's in the tree are identified by their id
        # each roi_id has it's own enumerated annotation_id and path_class
        self.img_annot_dict: Dict[int, Tuple[STRtree, Dict[int, Tuple[int, str]]]] = {}


    def update_path_classes(self, path_classes: Iterable[QuPathPathClass]) -> None:
        ''' update the annotation classes and annotation dictionaries of the project

        Parameters
        ----------
        path_classes:
            annotation classes to set
        '''
        self.path_classes = path_classes
        # overwrite _class_dict and _inverse_class_dict
        self._class_dict = {}
        for i, ann in enumerate(self.path_classes):
            self._class_dict[i] = ann
        self._inverse_class_dict = {value.id: key for key, value in self._class_dict.items()}


    def update_img_annot_dict(self, img_id: int) -> None:
        ''' update annotation rois tree for faster shapely queries

        Parameters
        ----------
        img_id:
            id of image to operate
        '''
        slide: QuPathProjectImageEntry = self.images[img_id]
        annotations: PathObjectProxy = slide.hierarchy.annotations
        img_ann_list: List[Tuple[Polygon, str]] = [
            (annot.roi, annot.path_class.id)
            for annot in annotations
        ]

        # [list(rois), list(annot_classes)]
        img_ann_transposed: np.ndarray = np.array(img_ann_list, dtype = object).transpose()
        class_by_id: Dict[int, Tuple[int, str]] = dict(
            (id(ann_poly), (i, img_ann_transposed[1][i]))
            for i, ann_poly in enumerate(img_ann_transposed[0])
        )
        img_ann_tree: STRtree = STRtree(img_ann_transposed[0])
        self.img_annot_dict[img_id] = (img_ann_tree, class_by_id)


    def get_tile(self,
                 img_id: int,
                 location: Tuple[int, int],
                 size: Tuple[int, int],
                 downsample_level: Optional[int] = 0) -> Image:
        ''' get tile starting at x|y (slide level 0) with given size

        Parameters
        ----------
        img_id:
            id of image to operate
        location:
            (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
        size:
            (width, height) tuple containing the tile size
        downsample_level:
            level for downsampling

        Returns
        -------
        :
            requested tile
        '''
        slide: QuPathProjectImageEntry = self.images[img_id]
        slide_url: str = slide.uri.removeprefix('file://')
        # remove leading '/' on windows systems '/C:/...' -> 'C:/...'
        if platform.system() == 'Windows':
            slide_url = slide_url.removeprefix('/')
        with openslide.open_slide(slide_url) as slide_data:
            tile: Image = slide_data.read_region(location, downsample_level, size)
        return tile


    def get_tile_annot(self,
                       img_id: int,
                       location: Tuple[int, int],
                       size: Tuple[int, int],
                       class_filter: Optional[List[Union[int, str]]] = None):
        ''' get tile annotations between (x|y) and (x + size| y + size)

        Parameters
        ----------
        img_id:
            id of image to operate
        location:
            (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
        size:
            (width, height) tuple containing the tile size
        class_filter:
            list of annotationclass names or ids to filter by
            if None no filter is applied

        Returns
        -------
        :
            list of annotations (shapely polygons) in tile
        '''
        location_x: int
        location_y: int
        width: int
        height: int
        location_x, location_y = location
        width, height = size
        polygon_tile: Polygon = Polygon((
            [location_x, location_y],
            [location_x + width, location_y],
            [location_x + width, location_y + height],
            [location_x, location_y + height]
        ))
        tile_intersections: List[Tuple[str, Polygon]] = []

        if img_id not in self.img_annot_dict:
            self.update_img_annot_dict(img_id)

        ann_tree, index_and_class = self.img_annot_dict[img_id]
        near_polys = [poly for poly in ann_tree.query(polygon_tile)]
        near_poly_classes = [index_and_class[id(poly)][1] for poly in near_polys]
        for poly, annot_class in zip(near_polys, near_poly_classes):
            intersection = poly.intersection(polygon_tile)
            if intersection.is_empty:
                continue

            filter_bool = ((not class_filter) or
                (annot_class in class_filter) or
                (self._inverse_class_dict[annot_class] in class_filter))

            # filter applies and polygon is a multipolygon
            if (filter_bool and
                    isinstance(intersection, (MultiPolygon, GeometryCollection))):
                for inter in intersection.geoms:
                    if isinstance(inter, Polygon):
                        tile_intersections.append((annot_class, inter))

            # filter applies and is a polygon
            elif filter_bool and isinstance(intersection, Polygon):
                tile_intersections.append((annot_class, intersection))

        return tile_intersections


    def get_tile_annot_mask(self, img_id, location, size, downsample_level=0, multilabel = False, class_filter = None):
        ''' get tile annotations mask between (x|y) and (x + size| y + size)

        Parameters
        ----------
        img_id:
            id of image to operate
        location:
            (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
        size:
            (width, height) tuple containing the tile size
        downsample_level:
            level for downsampling
        multilabel:
            if True annotation mask contains boolean image for each class ([num_classes, width, height])
        class_filter:
            list of annotationclass names to filter by

        Returns
        -------
        annot_mask: _
            mask [height, width] with an annotation class for each pixel
            or [num_class, height, width] for multilabels
            background class is ignored for multilabels ([0, height, width] shows mask for the first annotation class)
        '''
        location_x, location_y = location
        width, height = size
        downsample_factor = 2 ** downsample_level
        # level_0_size needed to get all Polygons in downsampled area
        level_0_size = tuple(map(lambda x: x* downsample_factor, size))
        tile_intersections = self.get_tile_annot(img_id, location, level_0_size, class_filter)

        if multilabel:
            num_classes = len(self.path_classes) -1
            annot_mask = np.zeros((num_classes, height, width), dtype = np.uint8)

        else:
            # sort intersections descending by area.
            # Now we can not accidentally overwrite polys with other poly holes
            sorted_intersections = sorted(tile_intersections,
                                          key = lambda tup: Polygon(tup[1].exterior).area,
                                          reverse=True)
            tile_intersections = sorted_intersections
            annot_mask = np.zeros((height, width), dtype = np.uint8)


        for inter_class, intersection in tile_intersections:
            class_num = self._inverse_class_dict[inter_class]
            if multilabel: # first class should be on the lowest level for multilabels
                class_num -= 1

            trans_inter = affinity.translate(intersection, location_x * -1, location_y * -1)
            # apply downsampling by scaling the Polygon down
            scale_inter = affinity.scale(
                trans_inter,
                xfact = 1/downsample_factor,
                yfact = 1/downsample_factor,
                origin = (0,0)
            )

            exteriors, interiors = _round_polygon(scale_inter)

            if multilabel:
                cv2.fillPoly(annot_mask[class_num], [exteriors], 1)
                cv2.fillPoly(annot_mask[class_num], interiors, 0)

            else:
                cv2.fillPoly(annot_mask, [exteriors], class_num)
                cv2.fillPoly(annot_mask, interiors, 0)

        return annot_mask


    def save_mask_annotations(self, img_id, annot_mask, location = (0,0), downsample_level = 0, min_polygon_area = 0, multilabel = False):
        ''' saves a mask as annotations to QuPath

        Parameters
        ----------
        img_id:
            id of image to operate
        annot_mask:
            mask with annotations
        location:
            (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
        downsample_level:
            level for downsampling
        min_polygon_area:
            minimal area for polygons to be saved
        multilabel:
            if True annotation mask contains boolean image for each class ([num_classes, width, height])
        '''
        slide = self.images[img_id]
        poly_annot_list = label_img_to_polys(annot_mask,
                                             downsample_level,
                                             min_polygon_area,
                                             multilabel)
        for annot_poly, annot_class in poly_annot_list:
            poly_to_add = affinity.translate(annot_poly, location[0], location[1])
            slide.hierarchy.add_annotation(poly_to_add, self._class_dict[annot_class])


    def merge_near_annotations(self, img_id, max_dist):
        ''' merge nearby annotations with equivalent annotation class

        Parameters
        ----------
        img_id:
            id of image to operate
        max_dist:
            maximal distance between annotations to merge
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
            annot_poly_class = annot.path_class.id
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
                    # tree query will always return the polygon from the same annotation
                    if index == near_poly_index:
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
                hierarchy.add_annotation(
                    merged_annot,
                    self._class_dict[self._inverse_class_dict[annot_poly_class]]
                )
                annotations.discard(annot)
