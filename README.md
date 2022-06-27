- [QuPath tiling python](#qupath-tiling-python)
  - [Installation](#installation)
  - [QuPathOperations API](#qupathoperations-api)

# QuPath tiling python

## Installation
1. 

## QuPathOperations API
This proect inherits from [paquo](https://github.com/bayer-science-for-a-better-life/paquo) [(docs)](https://paquo.readthedocs.io/en/latest/).

`QuPathOperations`  
> **class to initialize object (paquo)** 

> `update_path_classes(path_classes)`  
>> **update the annotation classes and annotation dictionaries of the project**  
>> 
>> Parameters:
>> - path_classes:  
    &emsp;  annotation classes to set  

> `update_img_annot_dict(img_id)`  
>> **update annotation rois tree for faster shapely queries**
>>
>> Parameters:
>> - img_id:  
    &emsp;  id of image to operate

> `get_tile(img_dir, img_id, location, size, downsample_level = 0)`
>> **get tile starting at x|y (slide level 0) with given size**
>>
>> Parameters:
>> - img_dir:  
    &emsp;  directory containing the image
>> - img_id:  
    &emsp;  id of image to operate
>> - location:    
    &emsp;  (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
>> - size:       
    &emsp;  (width, height) tuple containing the tile size
>> - downsample_level:  
    &emsp;  level for downsampling

>> Returns:
>> - tile:  
    &emsp;  tile image

> `get_tile_annot(img_id, location, size, class_filter = None)`
>> **get tile annotations between (x|y) and (x + size| y + size)**
>>
>> Parameters:  
>> - img_id:  
    &emsp;  id of image to operate
>> - location:  
    &emsp;  (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
>> - size:  
    &emsp;  (width, height) tuple containing the tile size
>> - class_filter:  
    &emsp;  list of annotationclass names or ids to filter by  
    &emsp;  if None no filter is applied

>> Returns:
>> - tile_intersections:  
    &emsp;  list of annotations (shapely polygons) in tile
  
> `get_tile_annot_mask(img_id, location, size, downsample_level = 0, multilabel = False, class_filter = None)`
>> **get tile annotations mask between (x|y) and (x + size| y + size)**
>>
>> Parameters:
>> - img_id:  
    &emsp;  id of image to operate
>> - location:  
    &emsp;  (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
>> - size:  
    &emsp;  (width, height) tuple containing the tile size
>> - downsample_level:  
    &emsp;  level for downsampling
>> - multilabel:  
    &emsp;  if True annotation mask contains boolean image for each class ([num_classes, width, height])
>> - class_filter:  
    &emsp;  list of annotationclass names to filter by  

>> Returns:  
>> - annot_mask:  
    &emsp;  mask [height, width] with an annotation class for each pixel  
    &emsp;  or [num_class, height, width] for multilabels  
    &emsp;  background class is ignored for multilabels ([0, height, width] shows mask for the first annotation class)

> `save_mask_annotations(img_id, annot_mask, location = (0,0), downsample_level = 0, min_polygon_area = 0, multilabel = False)`
>> **saves a mask as annotations to QuPath**
>> 
>> Parameters:  
>> - img_id:  
    &emsp;  id of image to operate
>> - annot_mask:  
    &emsp;  mask with annotations
>> - location:  
    &emsp;  (x, y) tuple containing coordinates for the top left pixel in the level 0 slide
>> - downsample_level:  
    &emsp;  level for downsampling
>> - min_polygon_area:  
    &emsp;  minimal area for polygons to be saved
>> - multilabel:  
    &emsp;  if True annotation mask contains boolean image for each class ([num_classes, width, height])

> `merge_near_annotations(img_id, max_dist)`
>> **merge nearby annotations with equivalent annotation class**
>>
>> Parameters:  
>> - img_id:  
    &emsp;  id of image to operate
>> - max_dist:  
    &emsp;  maximal distance between annotations to merge

> `label_img_to_polys(label_img, downsample_level = 0, min_polygon_area = 0, multilabel = False)`
>> classmethod  
>> **convert label mask to list of Polygons**
>>
>> Parameters:  
>> - label_img:  
    &emsp;  mask [H, W] with values between 0 and highest label class
>> - downsample_level:  
    &emsp;  level for downsampling
>> - min_polygon_area:  
    &emsp;  minimal area for polygons to be saved
>> - multilabel:  
    &emsp;  if True annotation mask contains boolean image for each class ([num_classes, width, height])

>> Returns:  
>> - poly_labels:  
    &emsp;  list of Polygon and label tuple [(polygon, label), ...]

