# -*- coding: utf-8 -*-
"""
Script para validar un modelo CNN ya entrenado con un
conjunto de datos de validación completamente nuevo.
"""

# --- 1. Importación de bibliotecas ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
# Se importa solo la función necesaria para cargar el modelo
from tensorflow.keras.models import load_model
from osgeo import gdal
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- 2. CONFIGURACIÓN DEL EXPERIMENTO ---

# ** IMPORTANTE: MODIFICA ESTAS VARIABLES **
# Ruta a la carpeta que contiene tu conjunto de datos de VALIDACIÓN
data_dir_validacion = 'E:/Angie/Experimento1_Validacion'
# Ruta donde guardaste tu modelo completo (archivo .keras)
# Asegúrate de especificar qué pliegue quieres validar, ej. el pliegue 1
ruta_modelo_keras = 'E:/Angie/codigos/Modelos_Guardados/modelo_completo_pliegue_2.keras'
# Número de bandas del experimento que estás validando
NUM_BANDAS = 5

# --- 3. Preparación de los Datos de Validación ---
clases = ['Sanas', 'Enfermas']
rutas_archivos, etiquetas = [], []

def escanear_directorio_clase(ruta_clase, etiqueta_clase):
    for carpeta_palma in os.listdir(ruta_clase):
        ruta_palma_individual = os.path.join(ruta_clase, carpeta_palma)
        if os.path.isdir(ruta_palma_individual):
            for archivo in os.listdir(ruta_palma_individual):
                if archivo.endswith(('.tif', '.tiff')):
                    rutas_archivos.append(os.path.join(ruta_palma_individual, archivo))
                    etiquetas.append(etiqueta_clase)

for i, clase in enumerate(clases):
    ruta_clase_actual = os.path.join(data_dir_validacion, clase)
    if os.path.isdir(ruta_clase_actual):
        escanear_directorio_clase(ruta_clase_actual, i)

df_val = pd.DataFrame({'filepath': rutas_archivos, 'label': etiquetas})
print(f"Total de imágenes de validación: {len(df_val)}")

# --- 4. Generador de Datos ---
def generador_multibanda(dataframe, batch_size, num_bandas, shuffle=False, target_size=(214, 214)):
    num_muestras = len(dataframe)
    while True:
        # No es necesario mezclar para la validación final
        for offset in range(0, num_muestras, batch_size):
            batch_df = dataframe.iloc[offset:offset+batch_size]
            actual_batch_size = len(batch_df)
            X_lote, y_lote = np.zeros((actual_batch_size, target_size[0], target_size[1], num_bandas)), np.zeros((actual_batch_size,))
            for batch_idx, (original_idx, row) in enumerate(batch_df.iterrows()):
                dataset = gdal.Open(row['filepath'])
                if dataset is not None and dataset.RasterCount == num_bandas:
                    imagen_array = np.zeros((target_size[0], target_size[1], num_bandas))
                    for b in range(num_bandas):
                        banda = dataset.GetRasterBand(b + 1).ReadAsArray()
                        min_val, max_val = np.min(banda), np.max(banda)
                        if max_val > min_val:
                            banda_norm = (banda - min_val) / (max_val - min_val)
                        else:
                            banda_norm = np.zeros_like(banda)
                        if num_bandas == 1:
                            imagen_array = np.expand_dims(banda_norm, axis=-1)
                        else:
                            imagen_array[:, :, b] = banda_norm
                    X_lote[batch_idx], y_lote[batch_idx] = imagen_array, row['label']
            yield X_lote, y_lote

# --- 5. Carga del Modelo Entrenado ---
print("\nCargando modelo entrenado...")
# Se utiliza load_model para cargar el archivo .keras guardado
modelo = load_model(ruta_modelo_keras)
print("¡Modelo cargado exitosamente!")

# --- 6. Predicción y Evaluación ---
batch_size = 16
# Se asegura de que shuffle=False para que las predicciones coincidan con las etiquetas
generador_validacion = generador_multibanda(df_val, batch_size, NUM_BANDAS, shuffle=False)
num_steps = -(-len(df_val) // batch_size)

print("\nRealizando predicciones sobre el conjunto de validación...")
predicciones_probs = modelo.predict(generador_validacion, steps=num_steps, verbose=1).flatten()
y_true = df_val['label'].iloc[:len(predicciones_probs)].values
y_pred = (predicciones_probs > 0.5).astype("int32")

# --- 7. Resultados Finales de la Validación ---
print("\n--- Reporte de Clasificación (Validación Final) ---")
print(classification_report(y_true, y_pred, target_names=clases))

auc_final = roc_auc_score(y_true, predicciones_probs)
print(f"\nAUC Final: {auc_final:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
plt.title('Matriz de Confusión (Validación Final)')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()