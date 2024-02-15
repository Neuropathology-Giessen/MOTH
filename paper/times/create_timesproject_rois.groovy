// groovy script to add defined rois
import qupath.lib.roi.ROIs
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane

def plane = ImagePlane.getPlane(0, 0)
//create circle
def ellipse = ROIs.createEllipseROI(49d,172d,30d,40d, plane)
def circle = ROIs.createEllipseROI(62d,62d,4d,4d,plane)
def uneven_circle = ROIs.createEllipseROI(189.5d,61.5d,5d,5d,plane)
// ROIs.createPolygonROI(double[] x, double[] y, ImagePlane plane)

// create class
annotationClass = getPathClass('Tumor')

// create annotations
def annotations = [
    PathObjects.createAnnotationObject(circle, annotationClass),
    PathObjects.createAnnotationObject(ellipse, annotationClass),
    PathObjects.createAnnotationObject(uneven_circle, annotationClass),
]
//add annotation
addObjects(annotations)