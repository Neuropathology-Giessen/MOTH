import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath('groovy_export', name)


def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK)
    .addLabel('Mitose',1)
    .build()


for(var x = 0; x<7; x++) {
    x_position = 60375 + x * 375 
    for(var y = 0; y<7; y++) {
        y_position = 100375 + y * 375 
        region = RegionRequest.createInstance(
            labelServer.getPath(), 1, 
            x_position, 
            y_position,
            375,
            375
        )
        writeImageRegion(labelServer, region, pathOutput + "x=" + x_position + "y=" + y_position)
    }
}