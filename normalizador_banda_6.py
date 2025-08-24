# -*- coding: utf-8 -*-
"""
Script para verificar y normalizar imágenes raster multibanda
en una estructura de carpetas anidada (Clase -> Individuo -> Imagen).
"""

# --- 1. Importación de bibliotecas ---
import os
import numpy as np
from osgeo import gdal

# --- 2. Configuración de Rutas ---

# IMPORTANTE: Modifica estas dos rutas según tus necesidades
# Ruta de la carpeta principal que contiene 'Sanas' y 'Enfermas'
ruta_entrada_principal = 'E:/Angie/Experimento_2'

# Ruta de la carpeta donde se guardará la nueva estructura con imágenes normalizadas
ruta_salida_principal = 'E:/Angie/Experimento_2_Normalizadas'

# --- 3. Función para Guardar Raster (sin cambios) ---

def guardar_raster(ruta_archivo_salida, dataset_original, arrays_bandas):
    """
    Guarda un nuevo archivo GeoTIFF a partir de una lista de arrays de NumPy.
    """
    driver = gdal.GetDriverByName('GTiff')
    num_bandas = len(arrays_bandas)
    altura, anchura = arrays_bandas[0].shape
    
    nuevo_dataset = driver.Create(ruta_archivo_salida, anchura, altura, num_bandas, gdal.GDT_Float32)
    
    nuevo_dataset.SetGeoTransform(dataset_original.GetGeoTransform())
    nuevo_dataset.SetProjection(dataset_original.GetProjection())
    
    for i, banda_array in enumerate(arrays_bandas):
        nuevo_dataset.GetRasterBand(i + 1).WriteArray(banda_array)
        
    nuevo_dataset.FlushCache()
    nuevo_dataset = None

# --- 4. Proceso Principal de Verificación y Normalización ---

print("\nIniciando proceso de verificación y normalización en carpetas anidadas...")

# Recorrer las carpetas de clase ('Sanas', 'Enfermas')
for nombre_clase in os.listdir(ruta_entrada_principal):
    ruta_clase_entrada = os.path.join(ruta_entrada_principal, nombre_clase)
    
    if os.path.isdir(ruta_clase_entrada):
        # Recorrer cada carpeta de individuo de palma
        for nombre_individuo in os.listdir(ruta_clase_entrada):
            ruta_individuo_entrada = os.path.join(ruta_clase_entrada, nombre_individuo)
            
            if os.path.isdir(ruta_individuo_entrada):
                # Crear la estructura de directorios correspondiente en la carpeta de salida
                ruta_individuo_salida = os.path.join(ruta_salida_principal, nombre_clase, nombre_individuo)
                if not os.path.exists(ruta_individuo_salida):
                    os.makedirs(ruta_individuo_salida)
                
                # Recorrer cada archivo de imagen dentro de la carpeta del individuo
                for nombre_archivo in os.listdir(ruta_individuo_entrada):
                    if nombre_archivo.endswith(('.tif', '.tiff')):
                        ruta_completa_entrada = os.path.join(ruta_individuo_entrada, nombre_archivo)
                        ruta_completa_salida = os.path.join(ruta_individuo_salida, nombre_archivo)
                        
                        dataset = gdal.Open(ruta_completa_entrada)
                        if dataset is None:
                            print(f"No se pudo abrir el archivo: {ruta_completa_entrada}")
                            continue
                        
                        necesita_normalizacion = False
                        bandas_leidas = []
                        
                        # Leer y verificar todas las bandas de la imagen
                        for i in range(1, dataset.RasterCount + 1):
                            banda = dataset.GetRasterBand(i)
                            array = banda.ReadAsArray().astype(float)
                            bandas_leidas.append(array)
                            if np.min(array) < 0 or np.max(array) > 1:
                                necesita_normalizacion = True
                        
                        # Si al menos una banda no está normalizada, se procesa la imagen completa
                        if necesita_normalizacion:
                            print(f"Normalizando archivo: {nombre_archivo} en {nombre_individuo}/{nombre_clase}")
                            bandas_normalizadas = []
                            for array_banda in bandas_leidas:
                                min_val, max_val = np.min(array_banda), np.max(array_banda)
                                if max_val > min_val:
                                    array_norm = (array_banda - min_val) / (max_val - min_val)
                                else:
                                    array_norm = np.zeros(array_banda.shape)
                                bandas_normalizadas.append(array_norm)
                            
                            guardar_raster(ruta_completa_salida, dataset, bandas_normalizadas)
                        else:
                            # Si ya está normalizado, simplemente se copia
                            print(f"Copiando archivo ya normalizado: {nombre_archivo}")
                            driver = gdal.GetDriverByName('GTiff')
                            driver.CreateCopy(ruta_completa_salida, dataset)

                        dataset = None

print("\nProceso completado.")