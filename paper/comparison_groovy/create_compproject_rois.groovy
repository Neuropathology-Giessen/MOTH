import qupath.lib.roi.ROIs
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane

def plane = ImagePlane.getPlane(0, 0)
//create circle
def roi_rec2 = ROIs.createRectangleROI(3d,6d,2d,2d, plane)
def roi_rec = ROIs.createRectangleROI(7.5d,4.5d,5d,5d, plane)
def roi_circle = ROIs.createEllipseROI(15d,5d,4d,4d, plane)
def roi_circle2 = ROIs.createEllipseROI(21.25d,4.25d,5d,5d, plane)

// create class
roiClass = getPathClass('Tumor')

// create annotation
def annotations = [
    PathObjects.createAnnotationObject(roi_circle, roiClass),
    PathObjects.createAnnotationObject(roi_circle2, roiClass),
    PathObjects.createAnnotationObject(roi_rec, roiClass),
    PathObjects.createAnnotationObject(roi_rec2, roiClass),
]

//add annotation
addObjects(annotations)