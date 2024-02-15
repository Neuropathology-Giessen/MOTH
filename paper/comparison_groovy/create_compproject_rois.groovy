import qupath.lib.roi.ROIs
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane

def plane = ImagePlane.getPlane(0, 0)
//create circle
def roi_circle = ROIs.createEllipseROI(5.5d,5.5d,1d,1d, plane)
def roi_circle2 = ROIs.createEllipseROI(20.25d,5.25d,2.5d,2.5d, plane)
def roi_rec = ROIs.createRectangleROI(10.5d,5.5d,5d,5d, plane)

// create class
roiClass = getPathClass('Tumor')

// create annotation
def annotations = [
    PathObjects.createAnnotationObject(roi_circle, roiClass),
    PathObjects.createAnnotationObject(roi_circle2, roiClass),
    PathObjects.createAnnotationObject(roi_rec, roiClass),
]

//add annotation
addObjects(annotations)