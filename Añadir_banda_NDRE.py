import gdal, ogr, os, osr
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology as morph
from skimage.transform import rescale, resize, downscale_local_mean


def array2raster(newRasterfn,array):
    
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn,214,214, 6, gdal.GDT_Float32)
    outRaster.SetGeoTransform(palm.GetGeoTransform())
    for num,banda in enumerate (array): 
        outband =outRaster.GetRasterBand(num+1)
        outband.WriteArray(banda)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(3116)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

os.chdir('C:\prueba1')
ruta = os.getcwd()  # obtiene ruta de cada imagen
contenido = os.listdir(ruta)  # obtiene lista con archivos/dir 
for nombre in contenido:
    
    palm=gdal.Open(nombre)
    palma=[0,0,0,0,0]

    for i in range(palm.RasterCount):
        palma[i]=palm.GetRasterBand(i+1).ReadAsArray()
    ndre=palma[3]-palma[4]/palma[3]+palma[4]
    palma.append(ndre) 
    n_archivo='C:\prueba1\\'+ nombre.replace('E1','E2')
    array2raster(n_archivo,palma)