import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath('groovy_export', name)
mkdirs(pathOutput)


def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK)
    .addLabel('Tumor',1)
    .build()
    
new TileExporter(imageData)
    .imageExtension('.tif')
    .tileSize(32)
    .labeledServer(labelServer)
    .annotatedTilesOnly(false)
    .writeTiles(pathOutput) 
    
    
print 'Done!'