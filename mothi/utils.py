from typing import Any, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import cv2
from shapely import affinity
from shapely.validation import make_valid
from shapely.geometry import Polygon, CAP_STYLE, JOIN_STYLE
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import InteriorRingSequence


def label_img_to_polys(label_img: NDArray[np.uint8],
        downsample_level: int = 0,
        min_polygon_area: Union[float, int] = 0,
        multilabel: bool = False) -> List[Tuple[Union[Polygon, BaseGeometry], int]]:
    ''' convert label mask to list of Polygons

    Parameters
    ----------
        label_img:
            mask [height, width] with an annotation class for each pixel
            or [num_class, height, width] for multilabels
            background class is ignored for multilabels
        downsample_level:
            level for downsampling
        min_polygon_area:
            minimal area for polygons to be saved
        multilabel:
            if True annotation mask contains boolean image for each class
    Returns
    -------
        :
            list of Polygon and label number tuple [(polygon, label_nmbr), ...]
    '''
    downsample_factor: int = 2 ** downsample_level
    poly_labels: List[Tuple[Union[Polygon, BaseGeometry], int]] = []

    iter_range: range = range(len(label_img)) if multilabel else range(1, np.max(label_img) + 1)

    class_id: int
    for class_id in iter_range:
        it_img: NDArray[np.uint8]
        if multilabel:
            it_img = label_img[class_id]
            class_id += 1
        else:
            label_img_copy: NDArray[np.uint8] = label_img.copy()
            it_img = np.where(label_img_copy == class_id, 1, 0).astype(np.uint8)

        contours: Tuple[NDArray[np.int32], ...]
        hierarchy: NDArray[np.int32]
        contours, hierarchy = cv2.findContours(it_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours = tuple(map(np.squeeze, contours))
        if len(contours) == 0:
            continue
        # iterate hierarchy
        next_id: int = 0                                 # start with first contour
        while next_id != -1:
            current_id: int = next_id                    # previous next id is now our current id
            child_id: int = hierarchy[0][current_id][2]
            next_id = hierarchy[0][current_id][0]

            if len(contours[current_id]) < 3:       # linear ring possible?
                continue

            # if contour has childs, thoose are the holes
            # -1 -> no childs
            poly: Union[Polygon, BaseGeometry] # final Polygon per iteration
            if child_id == -1:
                poly = Polygon(contours[current_id])
                poly = Polygon(*_round_polygon(poly, export = False))
            else:
                holes: List[NDArray[np.int32]] = []
                hole_poly: Polygon = Polygon(Polygon(contours[child_id]).buffer(
                    -1,
                    join_style= JOIN_STYLE.mitre,
                    cap_style= CAP_STYLE.square
                ))
                holes.append(np.apply_along_axis(
                    lambda coords: np.array(coords).round().astype(np.int32),
                    0,
                    hole_poly.exterior.coords  # type: ignore
                ))
                # search for further childs
                # further childs are listed by next in hierarchy of a known child
                next_child_id: int = hierarchy[0][child_id][0]
                while next_child_id != -1:
                    hole_poly = Polygon(Polygon(contours[child_id]).buffer(
                        -1,
                        join_style= JOIN_STYLE.mitre,
                        cap_style= CAP_STYLE.square
                    ))
                    holes.append(np.apply_along_axis(
                        lambda coords: np.array(coords).round().astype(np.int32),
                        0,
                        hole_poly.exterior.coords  # type:ignore
                    ))
                    next_child_id = hierarchy[0][next_child_id][0]
                poly = Polygon(contours[current_id], holes)
                poly = Polygon(*_round_polygon(poly, export = False))

            poly = affinity.scale(poly,
                xfact = 1 * downsample_factor,
                yfact = 1 * downsample_factor,
                origin = (0,0))  # type: ignore
                                 # coords Tuple[int, int] are also valid
                                 # docu: https://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.scale
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.area > min_polygon_area:
                poly_labels.append((poly, class_id))

    return poly_labels


def _round_polygon(polygon: Polygon, export = True):
    ''' round polygon coords to discrete values

    Parameters
    ----------
        polygon:
            Polygon to round coords

    Returns
    -------
        :
            rounded exterior coords
        :
            rounded interior coords
    '''
    exteriors: NDArray[np.int32] = np.array(polygon.exterior.coords)  # type: ignore
    centroid_coords: NDArray[np.float64] = np.array(
        [polygon.centroid.x, polygon.centroid.y],  # type: ignore
        dtype=np.float64
    )

    discrete_interiors: List[NDArray[np.int32]] = []
    interior_polys: Union[InteriorRingSequence, List[Any]] = polygon.interiors
    interior_centroids: List[NDArray[np.float64]] = [
        np.array([poly.centroid.x, poly.centroid.y], dtype=np.float64)
        for poly in interior_polys
    ]
    interiors_coord: List[NDArray[np.float64]] = [
        np.array(poly.coords, dtype=np.float64)
        for poly in interior_polys
    ]

    exteriors = np.apply_along_axis(_int_coord, 1,
                                    exteriors,
                                    centroid_coords,
                                    export).astype(np.int32)
    if len(interior_polys) > 0:
        for coords, centroid_coords in zip(interiors_coord, interior_centroids):
            discrete_interiors.append(
                np.apply_along_axis(_int_coord, 1, coords, centroid_coords, export).astype(np.int32)
            )
    return exteriors, discrete_interiors


def _int_coord(coord: np.float64, centroid_coords: np.float64, export: bool) -> NDArray[np.int32]:
    ''' round coordinate,
        increase or decrease value in comparision to the centroid
        (needed to eliminate difference between QuPath and shapely)

    Parameters
    ----------
    coord:
        coordinate to be rounded
    centroid_coords:
        reference centroid coordinate
    export:
        differ between import and export: increase or decrease coordinate
        depends on relation to centroid coordinate

    Returns
    -------
    :
        [rounded and] adjusted Array
    '''
    if export:
        return np.where(
            coord > centroid_coords,
            np.round(coord) - 1,
            np.round(coord)
        )
    return np.where(
        coord > centroid_coords,
        np.round(coord) + 1,
        np.round(coord)
    )
