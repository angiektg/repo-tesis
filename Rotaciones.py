from osgeo import gdal, osr,os
import numpy as np
from matplotlib import pyplot as plt



def array2raster(newRasterfn,array):
    
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn,214,214, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform(palm.GetGeoTransform())
    for num,banda in enumerate (array): 
        outband =outRaster.GetRasterBand(num+1)
        outband.WriteArray(banda)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(3116)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
def rotaciones (palma,nombre):
     for  j in range(3):
        palma = [np.rot90(banda) for banda in palma]
        n_archivo=nombre[:-4]+'R'+str(90*(j+1))+'.tif'
        array2raster(n_archivo,palma)

    
    
os.chdir('E:\Angie\Experimento3\Sanas\Sanas_general')
ruta = os.getcwd()  # obtiene ruta de cada imagen
contenido = os.listdir(ruta)  # obtiene lista con archivos/dir 
for nombre in contenido:
    
    palm=gdal.Open(nombre)

    palma=[0]

    for i in range(palm.RasterCount):
        palma[i]=palm.GetRasterBand(i+1).ReadAsArray()
    rotaciones(palma, nombre)
    palma = [np.flipud(banda) for banda in palma]
    n_archivo=nombre[:-4]+'M'+'.tif'
    array2raster(n_archivo,palma)
    rotaciones(palma, n_archivo)






