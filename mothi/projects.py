""" This module can be used for tiling in a QuPathProjects without leaving Python. """

import pathlib
import platform
from collections.abc import Iterable
from textwrap import dedent
from typing import Any, Literal, NamedTuple, Optional, Union, cast, overload

import numpy as np
import rasterio.features
from numpy.typing import NDArray
from paquo.classes import QuPathPathClass
from paquo.hierarchy import PathObjectProxy, QuPathPathObjectHierarchy
from paquo.images import QuPathProjectImageEntry
from paquo.pathobjects import QuPathPathAnnotationObject
from paquo.projects import ProjectIOMode, QuPathProject
from PIL.Image import Image
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tiffslide import TiffSlide


class MaskParameter(NamedTuple):
    """Parameter for mask generation and saving

    Parameters
    ----------
    img_id : int
        Id of image from which the tile annotation mask will be extracted
    location : tuple[int, int]
        (x, y) coordinates for the top left pixel in the tile
        pixel location without downsampling
    downsample_level : int, optional
        Level for downsampling, by default 0
    multichannel : bool, optional
        True: create binary images [num_channels, height, width]
        False: create labeled image [height, width], by default False
    downsample_level_power_of : Optional[int], optional
        Compute custom downsample factor with this value to the power of the downsample_level,
        by default None
    """

    img_id: int
    location: tuple[int, int]
    downsample_level: int = 0
    multichannel: bool = False
    downsample_level_power_of: Optional[int] = None


class QuPathTilingProject(QuPathProject):
    """Load or create a new QuPath project

    Parameters
    ----------
    path:
        Path to `project.qpproj` file, or its parent directory
    mode:
        'r' --> readonly, error if not there \n
        'r+' --> read/write, error if not there \n
        'a' = 'a+' --> read/write, create if not there, append if there \n
        'w' = 'w+' --> read/write, create if not there, truncate if there \n
        'x' = 'x+' --> read/write, create if not there, error if there
    """

    def __init__(self, path: Union[str, pathlib.Path], mode: ProjectIOMode = "r"):
        super().__init__(path, mode)
        self._class_dict: dict[int, QuPathPathClass] = {}
        self._inverse_class_dict: dict[str, int] = {}

        for i, ann in enumerate(self.path_classes):
            self._class_dict[i] = ann
        self._inverse_class_dict = {
            value.id: key for key, value in self._class_dict.items()
        }

        ## create dictionary to hold one STRtree per image
        # {img_id: (annotationTree, {roi_id: (annotation_id, path_class)})}
        # each image has an own query tree
        # roi's in the tree are identified by their id
        # each roi_id has it's own enumerated annotation_id and path_class
        self.img_annotation_dict: dict[
            int, tuple[STRtree, dict[int, tuple[int, str]]]
        ] = {}

    @QuPathProject.path_classes.setter
    def path_classes(self, path_classes: Iterable[QuPathPathClass]) -> None:
        """Update the annotation classes of the project

        Parameters
        ----------
        path_classes : Iterable[QuPathPathClass]
            Annotation classes to set
        """
        QuPathProject.path_classes.__set__(self, path_classes)
        self._class_dict = {}
        for i, ann in enumerate(self.path_classes):
            self._class_dict[i] = ann
        self._inverse_class_dict = {
            value.id: key for key, value in self._class_dict.items()
        }

    def _update_img_annotation_dict(self, img_id: int) -> None:
        """Update annotation roi tree for faster shapely queries

        Parameters
        ----------
        img_id : int
            Id of image to generate STRtree for
        """
        slide: QuPathProjectImageEntry = self.images[img_id]
        annotations: PathObjectProxy = slide.hierarchy.annotations
        img_ann_list: list[tuple[Polygon, str]] = [
            (
                (annotation.roi, annotation.path_class.id)
                if annotation.path_class is not None
                else (annotation.roi, "Unknown")
            )
            for annotation in annotations
        ]

        # list[tuple[Polygon, str]] -> NDArray[list(roi, list(annotation_classes)]
        img_ann_transposed: NDArray = np.array(img_ann_list, dtype=object).transpose()
        ## generate dict to map roi_id to tuple(annotation_id, annotation_class)
        class_by_id: dict[int, tuple[int, str]] = dict(
            (id(ann_poly), (i, img_ann_transposed[1][i]))
            for i, ann_poly in enumerate(img_ann_transposed[0])
        )
        img_ann_tree: STRtree = STRtree(img_ann_transposed[0])
        self.img_annotation_dict[img_id] = (img_ann_tree, class_by_id)

    @overload
    def get_tile(
        self,
        img_id: int,
        location: tuple[int, int],
        size: tuple[int, int],
        downsample_level: int = 0,
    ) -> Image: ...

    @overload
    def get_tile(  # pylint: disable=too-many-arguments
        self,
        img_id: int,
        location: tuple[int, int],
        size: tuple[int, int],
        downsample_level: int = 0,
        *,
        ret_array: Literal[True],
    ) -> NDArray[np.int_]: ...

    def get_tile(  # pylint: disable=too-many-arguments
        self,
        img_id: int,
        location: tuple[int, int],
        size: tuple[int, int],
        downsample_level: int = 0,
        *,
        ret_array: bool = False,
    ) -> Union[Image, NDArray[np.int_]]:
        """Get tile starting at (x,y) (slide level 0) with given size

        Parameters
        ----------
        img_id : int
            Id of image from which a tile will be generated
        location : tuple[int, int]
            (x, y) coordinates for the top left pixel in the tile \n
            pixel location without downsampling
        size : tuple[int, int]
            (width, height) for the tile
        downsample_level : int, optional
            Level for downsampling, by default 0
        ret_array : bool, optional
            True: return tile as array
            False: return as PIL Image,
            by default False

        Returns
        -------
        Union[Image, NDArray[np.int_]]
            Requested tile as PIL Image
        """
        slide: QuPathProjectImageEntry = self.images[img_id]
        slide_url: str = self._prepare_image_url(slide)
        # get requested tile
        with TiffSlide(slide_url) as slide_data:
            # if an array is requested, return array
            if ret_array:
                return slide_data.read_region(
                    location, downsample_level, size, as_array=True
                )
            return slide_data.read_region(location, downsample_level, size)

    def get_tile_annotation(
        self,
        img_id: int,
        location: tuple[int, int],
        size: tuple[int, int],
        class_filter: Optional[list[Union[int, str]]] = None,
    ) -> list[tuple[Polygon, str]]:
        """Get tile annotations between (x,y) and (x + width, y + height)

        Parameters
        ----------
        img_id : int
            Id of image from which the tile annotations will be extracted
        location : tuple[int, int]
            (x, y) coordinates for the top left pixel in the tile \n
            pixel location without downsampling
        size : tuple[int, int]
            (width, height) for the tile
        class_filter : Optional[list[Union[int, str]]], optional
            List of annotation class names or id's to filter by

        Returns
        -------
        list[tuple[Polygon, str]]
            List of annotations (polygon, annotation_class) in tile
        """

        polygon_tile: Polygon = Polygon(
            (
                [location[0], location[1]],
                [location[0] + size[0], location[1]],
                [location[0] + size[0], location[1] + size[1]],
                [location[0], location[1] + size[1]],
            )
        )
        tile_intersections: list[tuple[Polygon, str]] = []

        if img_id not in self.img_annotation_dict:
            self._update_img_annotation_dict(img_id)

        ann_tree: STRtree
        index_and_class: dict[int, tuple[int, str]]
        ann_tree, index_and_class = self.img_annotation_dict[img_id]
        near_polys: list[BaseGeometry] = ann_tree.geometries.take(
            ann_tree.query(polygon_tile)
        )
        near_poly_classes: list[str] = [
            index_and_class[id(poly)][1] for poly in near_polys
        ]

        poly: BaseGeometry
        annotation_class: str
        ## check if detected polygons intersect with tile and save intersections
        for poly, annotation_class in zip(near_polys, near_poly_classes):
            intersection: BaseGeometry = poly.intersection(polygon_tile)
            if intersection.is_empty:
                continue

            filter_bool: bool = (
                (not class_filter)
                or (annotation_class in class_filter)
                or (self._inverse_class_dict[annotation_class] in class_filter)
            )

            # filter applies and polygon is a multipolygon
            if filter_bool and isinstance(
                intersection, (MultiPolygon, GeometryCollection)
            ):
                inter: Union[BaseGeometry, BaseMultipartGeometry]
                for inter in intersection.geoms:
                    if isinstance(inter, Polygon):
                        tile_intersections.append((inter, annotation_class))

            # filter applies and is a polygon
            elif filter_bool and isinstance(intersection, Polygon):
                tile_intersections.append((intersection, annotation_class))

        return tile_intersections

    def get_tile_annotation_mask(
        self,
        mask_params: MaskParameter,
        size: tuple[int, int],
        *,
        class_filter: Optional[list[Union[int, str]]] = None,
    ) -> NDArray[np.int32]:
        """Get tile annotations mask between (x,y) and (x + width, y + height)

        Parameters
        ----------
        mask_params : MaskParameter
            Parameter for mask generation
        size : tuple[int, int]
            (width, height) for the tile
        class_filter : Optional[list[Union[int, str]]], optional
            list of annotation class names or id's to filter by, by default None

        Returns
        -------
        NDArray[np.int32]
            mask [height, width] with an annotation class for each pixel \n
            or binary_mask[num_class, height, width] for multichannel \n
            background class is ignored for multichannel
        """

        downsample_factor: float = self.get_downsample_factor(
            mask_params.downsample_level,
            img_id=mask_params.img_id,
            base=mask_params.downsample_level_power_of,
        )

        # level_0_size needed to get all Polygons in downsample area
        level_0_size: tuple[int, int] = cast(
            tuple[int, int], tuple(x * downsample_factor for x in size)
        )
        # get all annotations in tile
        tile_intersections: list[tuple[Polygon, str]] = self.get_tile_annotation(
            mask_params.img_id, mask_params.location, level_0_size, class_filter
        )

        # generate NDArray with zeroes where annotation will be drawn
        if mask_params.multichannel:
            num_classes: int = len(self.path_classes) - 1
            annotation_mask: NDArray[np.int32] = np.zeros(
                (num_classes, size[1], size[0]), dtype=np.int32
            )

        else:
            annotation_mask: NDArray[np.int32] = np.zeros(
                (size[1], size[0]), dtype=np.int32
            )
            ## sort intersections descending by area.
            # Now we can not accidentally overwrite polys with other poly holes
            sorted_intersections: list[tuple[Polygon, str]] = sorted(
                tile_intersections,
                key=lambda tup: Polygon(tup[0].exterior).area,
                reverse=True,
            )
            tile_intersections = sorted_intersections

        ## draw annotations on empty mask (NDArray)
        inter_class: str
        intersection: Polygon
        for intersection, inter_class in tile_intersections:
            class_num: Optional[int] = self._inverse_class_dict.get(inter_class)
            if class_num is None:
                continue
            # first class should be on the lowest level for multichannel
            if not mask_params.multichannel:
                class_num += 1

            # translate Polygon to (0,0)
            trans_inter: Polygon = affinity.translate(
                intersection, mask_params.location[0] * -1, mask_params.location[1] * -1
            )
            # apply downsampling by scaling the Polygon down
            scale_inter: Polygon = affinity.scale(
                trans_inter,
                xfact=1 / downsample_factor,
                yfact=1 / downsample_factor,
                origin=(0, 0),  # type: ignore
                # coords tuple[int, int] are also valid
                # docu: https://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.scale
            )
            if mask_params.multichannel:
                rasterio.features.rasterize(
                    [(scale_inter, 1)], out=annotation_mask[class_num]
                )
            else:
                rasterio.features.rasterize(
                    [(scale_inter, class_num)], out=annotation_mask
                )

        return annotation_mask

    def save_mask_annotations(
        self,
        annotation_mask: Union[NDArray[np.uint], NDArray[np.int_]],
        mask_params: MaskParameter,
    ) -> None:
        """Saves a mask as annotations to QuPath

        Parameters
        ----------
        annotation_mask : Union[NDArray[np.uint], NDArray[np.int_]]
            Mask [height, width] with an annotation class for each pixel
            or [num_class, height, width] for multichannel.
            Background class is ignored for multichannel
        mask_params : MaskParameter
            Parameter for mask import
        """
        annotation_mask = annotation_mask.astype(rasterio.int32)

        downsample_factor: float = self.get_downsample_factor(
            mask_params.downsample_level,
            img_id=mask_params.img_id,
            base=mask_params.downsample_level_power_of,
        )
        slide: QuPathProjectImageEntry = self.images[mask_params.img_id]

        poly_annotation_iter: Iterable[tuple[dict[str, Any], Any]]
        if not mask_params.multichannel:
            poly_annotation_iter = (
                (polygon_data, class_number - 1)
                for polygon_data, class_number in rasterio.features.shapes(
                    annotation_mask, annotation_mask != 0
                )
            )
        else:
            poly_annotation_iter = []
            for class_num in range(annotation_mask.shape[0]):
                poly_annotation_iter.extend(
                    (
                        (polygon_data, class_num)
                        for polygon_data, _ in rasterio.features.shapes(
                            annotation_mask[class_num], annotation_mask[class_num] != 0
                        )
                    )
                )

        annotation_poly_data: dict[str, Any]
        annotation_class: int
        ## add detected Polygons to the QuPath project
        for annotation_poly_data, annotation_class in poly_annotation_iter:
            annotation_poly = shape(annotation_poly_data)
            # scale poly to level 0 (no downsampling) size
            annotation_poly = affinity.scale(
                annotation_poly,
                xfact=1 * downsample_factor,
                yfact=1 * downsample_factor,
                origin=(0, 0),  # type: ignore
            )
            slide.hierarchy.add_annotation(
                affinity.translate(
                    annotation_poly, mask_params.location[0], mask_params.location[1]
                ),
                self._class_dict[annotation_class],
            )

    def merge_near_annotations(self, img_id: int, max_dist: Union[float, int]) -> None:
        """Merge nearby annotations with equivalent annotation class

        Parameters
        ----------
        img_id : int
            Id of the image where annotations will be merged
        max_dist : Union[float, int]
            Maximum distance up to which the annotations are merged
        """
        hierarchy: QuPathPathObjectHierarchy = self.images[img_id].hierarchy
        annotations: PathObjectProxy[QuPathPathAnnotationObject] = hierarchy.annotations
        self._update_img_annotation_dict(img_id)
        already_merged: list[int] = []  # save merged indices
        ann_tree: STRtree
        class_by_id: dict[int, tuple[int, str]]
        # unpack img_annotation_dict
        ann_tree, class_by_id = self.img_annotation_dict[img_id]

        index: int
        annotation: QuPathPathAnnotationObject
        ## loop trough all annotations and merge fitting annotations
        for index, annotation in enumerate(annotations):
            # skip and delete annotation if the annotation is already merged
            if index in already_merged:
                annotations.discard(annotation)
                continue
            annotation_poly: BaseGeometry = annotation.roi
            annotation_poly_class: str = (
                annotation.path_class.id if annotation.path_class else "Undefined"
            )
            annotation_poly_buffered: BaseGeometry = annotation_poly.buffer(max_dist)

            # save annotation to merge (initial: current annotation of the loop)
            annotations_to_merge: list[BaseGeometry] = [annotation_poly_buffered]

            ## detect possible merges between more then two Polygons
            ## first query current Polygon and then Polygons with intersections
            # nested annotations holds all annotations to check for further neighbors
            nested_annotations: list[BaseGeometry] = [annotation_poly_buffered]
            while len(nested_annotations) > 0:
                annotation_poly_buffered = nested_annotations.pop(0)
                near_polys: list[BaseGeometry] = ann_tree.geometries.take(
                    ann_tree.query(annotation_poly_buffered)
                )
                near_poly_index_and_classes: list[tuple[int, str]] = [
                    class_by_id[id(poly)] for poly in near_polys
                ]

                # check if nearby polygons are detected
                if len(near_polys) == 0:
                    continue
                near_poly: BaseGeometry
                near_poly_index: int
                near_poly_annotation_class: str
                ## save nearby polygons if they intersect with current polygon
                for near_poly, (near_poly_index, near_poly_annotation_class) in zip(
                    near_polys, near_poly_index_and_classes
                ):
                    # detected Polygon is already merged
                    # -> no further checks needed
                    if near_poly_index in already_merged:
                        continue
                    # tree query will always return the polygon from the same annotation
                    if index == near_poly_index:
                        continue
                    # compare Polygon classes
                    if annotation_poly_class != near_poly_annotation_class:
                        continue

                    # check if nearby polygon intersects
                    near_poly_buffered: BaseGeometry = near_poly.buffer(max_dist)
                    intersects: bool = near_poly_buffered.intersects(
                        annotation_poly_buffered
                    )
                    if intersects:
                        annotations_to_merge.append(near_poly_buffered)
                        nested_annotations.append(near_poly_buffered)
                        already_merged.append(near_poly_index)

            ## merge and save annotations
            # discard merged annotations
            if len(annotations_to_merge) > 1:
                merged_annotation: BaseGeometry = unary_union(
                    annotations_to_merge
                ).buffer(-max_dist)
                hierarchy.add_annotation(
                    merged_annotation,
                    self._class_dict[self._inverse_class_dict[annotation_poly_class]],
                )
                annotations.discard(annotation)

    def _prepare_image_url(self, slide: QuPathProjectImageEntry) -> str:
        """Prepare image url for tiling

        Parameters
        ----------
        slide : QuPathProjectImageEntry
            QuPathProjectImageEntry to get the url from

        Returns
        -------
        str
            prepared image url
        """
        slide_url: str = slide.uri.removeprefix("file://")
        # remove leading '/' on windows systems '/C:/...' -> 'C:/...'
        if platform.system() == "Windows":
            slide_url = slide_url.removeprefix("/")
        return slide_url

    @overload
    def get_downsample_factor(
        self,
        downsample_level: int,
        *,
        img_id: int,
    ) -> float: ...

    @overload
    def get_downsample_factor(
        self,
        downsample_level: int,
        *,
        base: int,
    ) -> float: ...

    @overload
    def get_downsample_factor(
        self,
        downsample_level: int,
        *,
        img_id: Optional[int],
        base: Optional[int],
    ) -> float: ...

    def get_downsample_factor(
        self,
        downsample_level: int,
        *,
        img_id: Optional[int] = None,
        base: Optional[int] = None,
    ) -> float:
        """Get downsample factor for a downsample_level.
        Either for a given image
        or computed for a given base value to the power of the downsample_level

        Parameters
        ----------
        downsample_level : int
            Level for downsampling
        img_id : Optional[int], optional
            Id of the image, by default None
        base : Optional[int], optional
            Compute custom downsample factor with the given base to the power of the downsample_level,
            by default None

        Returns
        -------
        float
            Downsample factor

        Raises
        ------
        ValueError
            Either img_id or power_of is required to get downsample factor
        ValueError
            Requested downsample level is not available for the image
        """

        if base is not None and img_id is not None:
            print("Downsampling: Both img_id and power_of are given. Using power_of")
            return base**downsample_level
        if base is not None:
            return base**downsample_level
        if img_id is None:
            raise ValueError("img_id or power_of is required to get downsample factor")
        try:
            return self.images[img_id].downsample_levels[downsample_level]["downsample"]
        except IndexError as exc:
            raise ValueError(
                dedent(
                    f"""\
                    Downsample level '{downsample_level}' not available for image [{img_id}].
                    Available levels: {self.images[img_id].downsample_levels}"""
                )
            ) from exc
