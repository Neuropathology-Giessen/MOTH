""" Util function supporting the QuPathTilingProject class. 
    Function also can be useful for different use cases 
"""

from typing import Any, List, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from shapely import affinity
from shapely.geometry import CAP_STYLE, JOIN_STYLE, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import InteriorRingSequence
from shapely.validation import make_valid


def label_img_to_polys(
    label_img: Union[NDArray[np.uint], NDArray[np.int_]],
    downsample_level: int = 0,
    min_polygon_area: Union[float, int] = 0,
    multichannel: bool = False,
) -> List[Tuple[Union[Polygon, BaseGeometry], int]]:
    """convert label mask to a list of Polygons

    Parameters
    ----------
        label_img:
            mask [height, width] with an annotation class for each pixel \n
            or [num_class, height, width] for multilabels \n
            background class is ignored for multilabels
        downsample_level:
            level for downsampling
        min_polygon_area:
            minimal polygon area to save polygon
        multichannel:
            True: binary image input [num_channels, height, width] \n
            False: labeled image input [height, width]
    Returns
    -------
        :
            list of detected polygons with their annotationclass number [(polygon, annot_cls), ...]
    """
    downsample_factor: int = 2**downsample_level
    poly_labels: List[Tuple[Union[Polygon, BaseGeometry], int]] = []

    # range up to the highest annotated class id
    iter_range: range = (
        range(len(label_img)) if multichannel else range(1, np.max(label_img) + 1)
    )

    class_id: int
    for class_id in iter_range:
        # get the binary mask of the current class
        it_img: NDArray[np.uint8]
        if multichannel:
            it_img = label_img.astype(np.uint8)[class_id]
            class_id += 1
        else:
            label_img_copy: Union[NDArray[np.uint], NDArray[np.int_]] = label_img.copy()
            it_img = np.where(label_img_copy == class_id, 1, 0).astype(np.uint8)

        ## find contours in binary mask
        contours: Tuple[NDArray[np.int_], ...]
        hierarchy: NDArray[np.int_]
        contours, hierarchy = cv2.findContours(
            it_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )  # type: ignore
        contours = tuple(map(np.squeeze, contours))
        if len(contours) == 0:
            continue
        ## iterate the hierarchy of detected contours
        # start with first contour
        next_id: int = 0
        while next_id != -1:
            # previous next id is now our current id
            current_id: int = next_id
            # get id of the child contour
            child_id: int = hierarchy[0][current_id][2]
            # get id of the next contour
            next_id = hierarchy[0][current_id][0]

            # skip if the creation of a linear ring is not possible
            if len(contours[current_id]) < 3:
                continue

            # final Polygon per iteration
            poly: Union[Polygon, BaseGeometry]
            # if the contour has childs, thoose contours are the holes
            # -1 -> no childs
            if child_id == -1:
                # create polygon without holes
                poly = Polygon(contours[current_id])
                poly = Polygon(*_round_polygon(poly, export=False))
            else:
                ## detect all holes
                holes: List[NDArray[np.int_]] = []
                hole_poly: Polygon = Polygon(
                    Polygon(contours[child_id]).buffer(
                        -1,
                        join_style=JOIN_STYLE.mitre,  # type: ignore
                        cap_style=CAP_STYLE.square,  # type: ignore
                    )
                )
                holes.append(
                    np.apply_along_axis(
                        lambda coords: np.array(coords).round().astype(np.int_),
                        0,
                        hole_poly.exterior.coords,  # type: ignore
                    )
                )
                ## search for further childs
                # further childs are listed by next in hierarchy of a known child
                next_child_id: int = hierarchy[0][child_id][0]
                while next_child_id != -1:
                    hole_poly = Polygon(
                        Polygon(contours[child_id]).buffer(
                            -1,
                            join_style=JOIN_STYLE.mitre,  # type: ignore
                            cap_style=CAP_STYLE.square,  # type: ignore
                        )
                    )
                    holes.append(
                        np.apply_along_axis(
                            lambda coords: np.array(coords).round().astype(np.int_),
                            0,
                            hole_poly.exterior.coords,  # type:ignore
                        )
                    )
                    next_child_id = hierarchy[0][next_child_id][0]
                # create polygon with holes
                poly = Polygon(contours[current_id], holes)
                poly = Polygon(*_round_polygon(poly, export=False))

            # scale poly to level 0 (no downsampling) size
            poly = affinity.scale(
                poly,
                xfact=1 * downsample_factor,
                yfact=1 * downsample_factor,
                origin=(0, 0),  # type: ignore
            )
            # coords Tuple[int, int] are also valid
            # docu: https://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.scale
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.area > min_polygon_area:
                poly_labels.append((poly, class_id))

    return poly_labels


def _round_polygon(
    polygon: Polygon, export: bool = True
) -> Tuple[NDArray[np.int_], List[NDArray[np.int_]]]:
    """round polygon coordinates to discrete values

    Parameters
    ----------
        polygon:
            Polygon whose coordinates are to be rounded

    Returns
    -------
        :
            rounded exterior coords
        :
            list of rounded interior coords
    """
    exteriors: NDArray[np.int_] = np.array(polygon.exterior.coords)  # type: ignore
    ## get centroid of the polygon
    centroid_coords: NDArray[np.float_] = np.array(
        [polygon.centroid.x, polygon.centroid.y], dtype=np.float_  # type: ignore
    )

    discrete_interiors: List[NDArray[np.int_]] = []
    interior_polys: Union[InteriorRingSequence, List[Any]] = polygon.interiors
    # get centroid of the polygon interiors
    interior_centroids: List[NDArray[np.float_]] = [
        np.array([poly.centroid.x, poly.centroid.y], dtype=np.float_)
        for poly in interior_polys
    ]
    # get list of the polygon coordinates
    interiors_coord: List[NDArray[np.float_]] = [
        np.array(poly.coords, dtype=np.float_) for poly in interior_polys
    ]

    ## round and adjust coordinates
    exteriors = np.apply_along_axis(
        _int_coord, 1, exteriors, centroid_coords, export
    ).astype(np.int_)
    coords: NDArray[np.float_]
    if len(interior_polys) > 0:
        for coords, centroid_coords in zip(interiors_coord, interior_centroids):
            discrete_interiors.append(
                np.apply_along_axis(
                    _int_coord, 1, coords, centroid_coords, export
                ).astype(np.int_)
            )
    return exteriors, discrete_interiors


def _int_coord(
    coord: NDArray[np.float_], centroid_coords: NDArray[np.float_], export: bool
) -> NDArray[np.int_]:
    """round coordinate \n
    increase or decrease value in comparision to the centroid
    (needed to eliminate difference between QuPath and shapely)

    Parameters
    ----------
    coord:
        coordinate to be rounded
    centroid_coords:
        reference centroid coordinate
    export:
        differ between import and export: increase or decrease coordinate \n
        depends on relation to centroid coordinate

    Returns
    -------
    :
        (rounded and) adjusted Array
    """
    if export:
        return np.where(coord > centroid_coords, np.round(coord) - 1, np.round(coord))
    return np.where(coord > centroid_coords, np.round(coord) + 1, np.round(coord))
