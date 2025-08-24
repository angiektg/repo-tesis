from osgeo import gdal,ogr, os, osr
import numpy as np
import time
from skimage import morphology as morph
from skimage.transform import rescale, resize, downscale_local_mean
from subprocess import call

cont =0
fecha= '0203'

def array2raster(newRasterfn,array):
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn,214,214, 1, gdal.GDT_Float32)#el numero 1 es la cantidad de bandas de la imagen
    outRaster.SetGeoTransform(palm.GetGeoTransform())
    for num,banda in enumerate (array): 
        outband =outRaster.GetRasterBand(num+1)
        outband.WriteArray(banda)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(3116)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
capa = iface.mapCanvas()
lyr = capa.currentLayer()
cont =0
for i in list(lyr.getFeatures()):
    box = i.geometry().boundingBox()
    caja = str((box.xMinimum(), box.yMaximum(), box.xMaximum(), box.yMinimum())).replace("(","").replace(")","").replace(",","")
    atributo=str(i.attributes()[13])#ID de la palama en la tabla de atributos del Buffer
    comando =  'gdal_translate -projwin ' + caja + ' -ot Float32 -of GTiff  E:/Angie/Raster/Termicas_norm/SCG_20200203_TERM_NORM.tif E:/Angie/Experimento3/Sanas/Sanas_Sin_Rotar/' +'P'+atributo+'F'+fecha+'E3'+'.tif'
    print (comando)
    call(comando)
    time.sleep(1)
    newRasterfn1='P'+atributo+'F'+fecha+'E3'+'.tif'
    os.chdir('E:/Angie/Experimento3/Sanas/Sanas_Sin_Rotar/')
    #tamaño=np.shape(palma_dem)

    palm=gdal.Open('E:/Angie/Experimento3/Sanas/Sanas_Sin_Rotar/' +'P'+atributo+'F'+fecha+'E3'+'.tif')
    
    palma=[0]#Cantidad de bandas de la imagen palma[0,0,0] es 3 bandas

    for j in range(palm.RasterCount):
        palma[j]=palm.GetRasterBand(j+1).ReadAsArray()
        palma[j]= resize(palma[j],(214,214),anti_aliasing=True)
    
   
    array2raster(newRasterfn1, palma)
    cont = cont + 1

