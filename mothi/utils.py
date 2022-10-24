import numpy as np
import cv2
import shapely

from shapely.validation import make_valid
from shapely.geometry import Polygon, CAP_STYLE, JOIN_STYLE


def label_img_to_polys(label_img, downsample_level = 0, min_polygon_area = 0, multilabel = False):
    ''' convert label mask to list of Polygons

    Parameters
    ----------
        label_img:
            mask [H, W] with values between 0 and highest label class
        downsample_level:
            level for downsampling
        min_polygon_area:
            minimal area for polygons to be saved
        multilabel:
            if True annotation mask contains boolean image for each class ([num_classes, width, height])

    Returns
    -------
        poly_labels: _ 
            list of Polygon and label tuple [(polygon, label), ...]
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
                exterior, interior = _round_polygon(poly, export = False)
                poly = Polygon(exterior, interior)
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
                exterior, interior = _round_polygon(poly, export = False)
                poly = Polygon(exterior, interior)

            poly = shapely.affinity.scale(poly, xfact = 1*downsample_factor, yfact = 1*downsample_factor, origin = (0,0))
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.area > min_polygon_area:
                poly_labels.append((poly, i))

    return poly_labels

def _round_polygon(polygon, export = True):
    ''' round polygon coords to discrete values 

        Parameters:
            polygon:    Polygon to round coords

        Returns:
            exterior:   rounded exterior coords
            interior:   rounded interior coords
    '''
    exteriors = np.array(polygon.exterior.coords)
    centroid_coords = np.array([polygon.centroid.x, polygon.centroid.y])

    discrete_interiors = []
    interior_polys = [poly for poly in polygon.interiors]
    interior_centroids = [np.array([poly.centroid.x, poly.centroid.y]) for poly in interior_polys]
    interiors_coord = [np.array(poly.coords) for poly in interior_polys]
    
    if export:
        int_coord = lambda coord, centroid_coords: np.where(
        coord > centroid_coords,
        np.round(coord) - 1,
        np.round(coord)
        )
    else:
        int_coord = lambda coord, centroid_coords: np.where(
        coord > centroid_coords,
        np.round(coord) + 1,
        np.round(coord)
        )

    exteriors = np.apply_along_axis(int_coord, 1, exteriors, centroid_coords).astype(np.int32)
    if len(interior_polys) > 0:
        for coords, centroid in zip(interiors_coord, interior_centroids):
            discrete_interiors.append(np.apply_along_axis(int_coord, 1, coords, centroid).astype(np.int32))
    return exteriors, discrete_interiors