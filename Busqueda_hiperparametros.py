# -*- coding: utf-8 -*-
"""
Script 1 (Completo): Búsqueda de Hiperparámetros para la CNN,
optimizado para reducir el uso de memoria y con la indentación correcta.
"""

# --- 1. Importación de bibliotecas ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers, backend as K
import numpy as np
import os
import pandas as pd
from osgeo import gdal
from sklearn.model_selection import train_test_split
# Asegúrate de que Keras Tuner esté importado correctamente
import kerastuner as kt

# --- 2. Preparación de los Datos ---
K.clear_session()
# IMPORTANTE: Reemplaza esta ruta con la ruta a tu carpeta principal
data_dir = 'E:\Angie\Experimento_1'
clases = ['Sanas', 'Enfermas']
rutas_archivos, etiquetas, grupos = [], [], []

def escanear_directorio_clase(ruta_clase, etiqueta_clase):
    for carpeta_palma in os.listdir(ruta_clase):
        ruta_palma_individual = os.path.join(ruta_clase, carpeta_palma)
        if os.path.isdir(ruta_palma_individual):
            id_individuo = carpeta_palma
            for archivo in os.listdir(ruta_palma_individual):
                if archivo.endswith(('.tif', '.tiff')):
                    rutas_archivos.append(os.path.join(ruta_palma_individual, archivo))
                    etiquetas.append(etiqueta_clase)
                    grupos.append(id_individuo)

for i, clase in enumerate(clases):
    ruta_clase_actual = os.path.join(data_dir, clase)
    if os.path.isdir(ruta_clase_actual):
        escanear_directorio_clase(ruta_clase_actual, i)

df = pd.DataFrame({'filepath': rutas_archivos, 'label': etiquetas, 'group': grupos})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Dividir los datos una sola vez para la sintonización
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# --- 3. Generador de Datos Personalizado ---
def generador_multibanda(dataframe, batch_size, target_size=(214, 214)):
    num_muestras = len(dataframe)
    while True:
        dataframe = dataframe.sample(frac=1)
        for offset in range(0, num_muestras, batch_size):
            batch_df = dataframe.iloc[offset:offset+batch_size]
            actual_batch_size = len(batch_df)
            X_lote, y_lote = np.zeros((actual_batch_size, target_size[0], target_size[1], 5)), np.zeros((actual_batch_size,))
            for batch_idx, (original_idx, row) in enumerate(batch_df.iterrows()):
                dataset = gdal.Open(row['filepath'])
                if dataset is not None:
                    imagen_array = np.zeros((target_size[0], target_size[1], 5))
                    for b in range(5):
                        banda = dataset.GetRasterBand(b + 1).ReadAsArray()
                        min_val, max_val = np.min(banda), np.max(banda)
                        if max_val > min_val:
                            banda_norm = (banda - min_val) / (max_val - min_val)
                        else:
                            banda_norm = banda - min_val
                        imagen_array[:, :, b] = banda_norm
                    X_lote[batch_idx], y_lote[batch_idx] = imagen_array, row['label']
            yield X_lote, y_lote

# --- 4. Creación de la Función del Hipermodelo ---
def construir_hipermodelo(hp):
    """
    Función que define el modelo y el espacio de búsqueda de hiperparámetros.
    """
    modelo = Sequential()
    
    # Espacio de Búsqueda para Capas Convolucionales
    modelo.add(Convolution2D(
        filters=hp.Int('conv_1_filtros', min_value=16, max_value=64, step=16),
        kernel_size=(3,3), padding='same', input_shape=(214, 214, 5), activation='relu'
    ))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    modelo.add(Convolution2D(
        filters=hp.Int('conv_2_filtros', min_value=32, max_value=128, step=32),
        kernel_size=(3,3), padding='same', activation='relu'
    ))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    
    modelo.add(Flatten())
    
    # Espacio de Búsqueda para Capa Densa
    modelo.add(Dense(
        units=hp.Int('unidades_densas', min_value=128, max_value=256, step=64),
        activation='relu'
    ))
    
    # Espacio de Búsqueda para Dropout
    modelo.add(Dropout(rate=hp.Float('tasa_dropout', min_value=0.4, max_value=0.5, step=0.1)))
    
    modelo.add(Dense(1, activation='sigmoid'))

    # Espacio de Búsqueda para la Tasa de Aprendizaje
    hp_learning_rate = hp.Choice('tasa_aprendizaje', values=[1e-3, 1e-4])

    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return modelo

# --- 5. Instanciación y Ejecución del Sintonizador ---
tuner = kt.RandomSearch(
    construir_hipermodelo,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='palma_tuning',
    overwrite=True
)

batch_size = 8

train_generator = generador_multibanda(train_df, batch_size)
validation_generator = generador_multibanda(val_df, batch_size)

print("\n--- Iniciando la Búsqueda de Hiperparámetros ---")
tuner.search(
    train_generator,
    epochs=10,
    steps_per_epoch=max(1, len(train_df) // batch_size),
    validation_data=validation_generator,
    validation_steps=max(1, len(val_df) // batch_size)
)

# --- 6. Obtención y Resumen de los Mejores Resultados ---
mejores_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
--- Búsqueda Completa ---
Los mejores hiperparámetros encontrados son:
- Filtros Capa Conv 1: {mejores_hps.get('conv_1_filtros')}
- Filtros Capa Conv 2: {mejores_hps.get('conv_2_filtros')}
- Neuronas Capa Densa: {mejores_hps.get('unidades_densas')}
- Tasa de Dropout: {mejores_hps.get('tasa_dropout'):.2f}
- Tasa de Aprendizaje: {mejores_hps.get('tasa_aprendizaje')}
""")