# -*- coding: utf-8 -*-
"""
Script final y unificado para entrenar, evaluar y GUARDAR una CNN.
"""

# --- 1. CONFIGURACIÓN E IMPORTACIÓN DE BIBLIOTECAS ---
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(f"TF_GPU_ALLOCATOR establecido en: {os.environ.get('TF_GPU_ALLOCATOR')}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers, backend as K
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from osgeo import gdal
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0
from tensorflow.keras.models import Model

# --- Verificación de GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detectada: {gpus[0]}")
else:
    print("Advertencia: No se detectó GPU.")

# --- 2. PREPARACIÓN DE LOS DATOS ---
K.clear_session()
data_dir = 'E:/Angie/Experimento_1'
NUM_BANDAS =5
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
print(f"Total de imágenes: {len(df)}, Total de individuos: {df['group'].nunique()}")

pesos = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
pesos_clase = {0: pesos[0], 1: pesos[1]}
print(f"Pesos de clase calculados: {pesos_clase}")

# --- 3. GENERADOR DE DATOS PERSONALIZADO ---
def generador_multibanda(dataframe, batch_size, num_bandas, shuffle=True, target_size=(214, 214)):
 
    num_muestras = len(dataframe)
    while True:
        if shuffle:
            dataframe = dataframe.sample(frac=1)
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
                    
                    if np.isnan(imagen_array).any():
                        print(f"Advertencia: Se encontraron valores NaN en {row['filepath']}. Se omitirá esta muestra.")
                        continue

                    X_lote[batch_idx], y_lote[batch_idx] = imagen_array, row['label']
            yield X_lote, y_lote


# --- 4. DEFINICIÓN DEL MEJOR MODELO CNN ---


MEJORES_FILTROS_1, MEJORES_FILTROS_2, MEJORES_UNIDADES = 32, 64, 192
MEJOR_DROPOUT, MEJOR_LR = 0.40, 1e-6
epocas = 15
altura, longitud = 214, 214
batch_size = 8
#..................................CNN ORIGINAL..........................................
# def crear_mejor_modelo_cnn(num_bandas):
#     cnn = Sequential()
#     cnn.add(Input(shape=(altura, longitud, num_bandas)))
#     cnn.add(Convolution2D(MEJORES_FILTROS_1, (3, 3), padding='same', activation='relu'))
#     cnn.add(MaxPooling2D(pool_size=(2, 2)))
#     cnn.add(Dropout(0.25))
#     cnn.add(Convolution2D(MEJORES_FILTROS_2, (3, 3), padding='same', activation='relu'))
#     cnn.add(MaxPooling2D(pool_size=(2, 2)))
#     cnn.add(Dropout(0.25))
#     cnn.add(Flatten())
#     cnn.add(Dense(MEJORES_UNIDADES, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
#     cnn.add(Dropout(0.5))
#     cnn.add(Dense(1, activation='sigmoid'))
    
#     optimizador = optimizers.Adam(learning_rate=MEJOR_LR, clipvalue=1.0)
#     cnn.compile(optimizer=optimizador, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
#     return cnn


# .............................ResNet50V2..................................................

# def crear_mejor_modelo_cnn(num_bandas):
#     """
#     Crea un modelo basado en la arquitectura ResNet50V2 pre-entrenada
#     utilizando transferencia de aprendizaje.
#     """
#     # 1. Cargar el modelo base ResNet50V2 pre-entrenado en ImageNet.
#     # No incluimos la parte superior (las capas de clasificación originales).
#     # 'weights="imagenet"' descarga los pesos pre-entrenados.
#     base_model = ResNet50V2(include_top=False, 
#                             weights="imagenet", 
#                             input_shape=(altura, longitud, 3)) # ResNet espera 3 canales

#     # 2. Congelar las capas del modelo base para que no se entrenen inicialmente.
#     base_model.trainable = False
    
#     # 3. Construir tus propias capas de clasificación encima del modelo base.
#     # La entrada a tus capas será la salida del modelo base.
#     inputs = Input(shape=(altura, longitud, num_bandas))

#     # --- Adaptador de Canales ---
#     # ResNet fue entrenado con imágenes RGB (3 canales). Tus imágenes tienen 'num_bandas'.
#     # Usamos una capa convolucional 1x1 para adaptar tus 5 o 6 bandas a las 3 que ResNet espera.
#     if num_bandas != 3:
#         x = Convolution2D(3, (1, 1), padding='same')(inputs) # Capa adaptadora
#     else:
#         x = inputs

#     # Pasar los datos adaptados a través del modelo base
#     x = base_model(x, training=False)
    
#     # --- Añadir tus propias capas de clasificación ---
#     x = Flatten()(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     outputs = Dense(1, activation='sigmoid')(x)
    
#     # 4. Crear el modelo final
#     modelo = Model(inputs, outputs)

#     # 5. Compilar el modelo
#     optimizador = optimizers.Adam(learning_rate=1e-4) # Una tasa de aprendizaje más alta para empezar
#     modelo.compile(optimizer=optimizador, 
#                  loss='binary_crossentropy', 
#                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
#     return modelo


# .............................. EfficientNetB0 ..................................

def crear_mejor_modelo_cnn(num_bandas):
    # """
    # Crea un modelo basado en la arquitectura EfficientNetB0 pre-entanada.
    # """
    # 1. Cargar el modelo base EfficientNetB0
    base_model = EfficientNetB0(include_top=False, 
                                weights="imagenet", 
                                input_shape=(altura, longitud, 3))

    # 2. Congelar las capas del modelo base
    base_model.trainable = False
    
    # 3. Construir tus propias capas encima (igual que con ResNet)
    inputs = Input(shape=(altura, longitud, num_bandas))

    # Capa adaptadora si es necesario
    if num_bandas != 3:
        x = Convolution2D(3, (1, 1), padding='same')(inputs)
    else:
        x = inputs
    
    x = base_model(x, training=False)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    # 4. Crear y compilar el modelo final
    modelo = Model(inputs, outputs)
    
    optimizador = optimizers.Adam(learning_rate=1e-4)
    modelo.compile(optimizer=optimizador, 
                 loss='binary_crossentropy', 
                 metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    return modelo

# --- 5. IMPLEMENTACIÓN DE LA VALIDACIÓN CRUZADA Y GUARDADO DE MODELOS ---

# *** NUEVO: Crear directorio para guardar los modelos ***
directorio_modelos = './Modelos_Guardados'
if not os.path.exists(directorio_modelos):
    os.makedirs(directorio_modelos)
    print(f"Directorio creado para guardar modelos: {directorio_modelos}")

n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
historias, reportes_df_list = [], []
y_true_total, y_pred_probs_total = [], []

for i, (train_index, val_index) in enumerate(sgkf.split(df['filepath'], df['label'], df['group'])):
    print(f"\n--- Iniciando Pliegue {i+1}/{n_splits} ---")
    K.clear_session()
    modelo = crear_mejor_modelo_cnn(NUM_BANDAS)
    train_df, val_df = df.iloc[train_index], df.iloc[val_index]
    
    train_generator = generador_multibanda(train_df, batch_size, NUM_BANDAS, shuffle=True)
    validation_generator = generador_multibanda(val_df, batch_size, NUM_BANDAS, shuffle=True) # Shuffle para validación durante fit
        
    historia = modelo.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_df) // batch_size),
        epochs=epocas,
        validation_data=validation_generator,
        validation_steps=max(1, len(val_df) // batch_size),
        class_weight=pesos_clase,
        callbacks=[TerminateOnNaN()],
        verbose=1
    )
    historias.append(historia.history)
    
    
    # *** NUEVO: Guardar el modelo y los pesos de este pliegue ***
    print(f"Guardando modelo del pliegue {i+1}...")
    # Guardar la arquitectura en un archivo JSON
    modelo_json = modelo.to_json()
    
    ruta_json = os.path.join(directorio_modelos, f"modelo_pliegue_{i+1}.json")
    with open(ruta_json, "w") as json_file:
        json_file.write(modelo_json)
    # Guardar los pesos en un archivo H5
    # ruta_h5 = os.path.join(directorio_modelos, f"pesos_pliegue_{i+1}.h5")
    # modelo.save_weights(ruta_h5)
    # print(f"Modelo guardado en: {ruta_json} y {ruta_h5}")
    
 
   
    ruta_modelo_completo = os.path.join(directorio_modelos, f"modelo_completo_pliegue_{i+1}.keras")
    modelo.save(ruta_modelo_completo)
    print(f"Modelo guardado en: {ruta_modelo_completo}")
   
    # Generador de evaluación SIN MEZCLAR para mantener el orden
    val_gen_eval = generador_multibanda(val_df, batch_size, NUM_BANDAS, shuffle=False)
    num_val_steps = -(-len(val_df) // batch_size) # División de techo para incluir todas las muestras

    if num_val_steps > 0:
        predicciones_probs = modelo.predict(val_gen_eval, steps=num_val_steps, verbose=0).flatten()
        etiquetas_verdaderas = val_df['label'].iloc[:len(predicciones_probs)].values
        
        y_true_total.extend(etiquetas_verdaderas)
        y_pred_probs_total.extend(predicciones_probs)
        
        try:
            predicciones_clases = (predicciones_probs > 0.5).astype("int32")
            reporte = classification_report(etiquetas_verdaderas, predicciones_clases, target_names=clases, output_dict=True, zero_division=0)
            df_reporte = pd.DataFrame(reporte).transpose()
            df_reporte['pliegue'] = i + 1
            reportes_df_list.append(df_reporte)
        except ValueError as e:
            print(f"Error al generar el reporte para el pliegue {i+1}: {e}")


# --- 6. ANÁLISIS DEL UMBRAL Y RESULTADOS FINALES ---
from sklearn.metrics import precision_recall_curve

# Usar las etiquetas verdaderas y las probabilidades predichas de todos los pliegues
y_true = np.array(y_true_total)
y_pred_probs = np.array(y_pred_probs_total)

# Calcular la curva de precisión-recall
precisiones, recalls, umbrales = precision_recall_curve(y_true, y_pred_probs)

# Calcular el F1-score para cada umbral
# Se añade un pequeño epsilon (1e-7) para evitar la división por cero
f1_scores = (2 * precisiones * recalls) / (precisiones + recalls + 1e-7)

# Encontrar el índice del mejor F1-score
ix = np.argmax(f1_scores)

# Obtener el umbral que corresponde al mejor F1-score
umbral_optimo = umbrales[ix]
print(f'\n--- Optimización del Umbral ---')
print(f'Mejor F1-Score={f1_scores[ix]:.4f}, Umbral Óptimo={umbral_optimo:.4f}')

# --- 7. RESULTADOS Y GRÁFICAS FINALES CON UMBRAL ÓPTIMO ---

# Matriz de confusión global usando el UMBRAL ÓPTIMO
y_pred_final_optimizado = (y_pred_probs > umbral_optimo).astype("int32")
cm = confusion_matrix(y_true_total, y_pred_final_optimizado)

# Crear las gráficas
plt.figure(figsize=(20, 5))


#  Curva de Precisión-Recall
plt.subplot(1, 4, 3)
plt.plot(recalls, precisiones, marker='.', label='Curva PR')
plt.scatter(recalls[ix], precisiones[ix], marker='o', color='red', label=f'Mejor Umbral ({umbral_optimo:.2f})')
plt.title('Curva Precisión-Recall')
plt.xlabel('Recall (Sensibilidad)')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

# Matriz de confusión global con umbral optimizado
plt.subplot(1, 4, 4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases, annot_kws={"size": 14})
plt.title('Matriz de Confusión (Umbral Óptimo)')
plt.ylabel('Etiqueta Verdadera'); plt.xlabel('Etiqueta Predicha')

plt.suptitle('Resultados del Entrenamiento y Evaluación del Modelo CNN', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# Imprimir el reporte de clasificación final con el umbral optimizado
print("\n--- Reporte de Clasificación Final (con Umbral Óptimo) ---")
print(classification_report(y_true_total, y_pred_final_optimizado, target_names=clases))
# --- 6. RESULTADOS Y GRÁFICAS FINALES ---
tabla_comparativa = pd.concat(reportes_df_list)
tabla_comparativa = tabla_comparativa.reset_index().rename(columns={'index': 'clase'})
tabla_comparativa = tabla_comparativa[['pliegue', 'clase', 'precision', 'recall', 'f1-score', 'support']]
print("\n--- Tabla Comparativa de Métricas por Pliegue ---")
print(tabla_comparativa.to_string())

# Calcular métricas promedio
avg_acc = np.mean([h['accuracy'] for h in historias], axis=0)
avg_val_acc = np.mean([h['val_accuracy'] for h in historias], axis=0)
avg_loss = np.mean([h['loss'] for h in historias], axis=0)
avg_val_loss = np.mean([h['val_loss'] for h in historias], axis=0)
avg_auc = np.mean([h['auc'] for h in historias], axis=0)
avg_val_auc = np.mean([h['val_auc'] for h in historias], axis=0)
epochs_range = range(1, len(avg_acc) + 1)

# Crear las gráficas
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.plot(epochs_range, avg_acc, 'bo-', label='Entrenamiento')
plt.plot(epochs_range, avg_val_acc, 'ro-', label='Validación')
plt.title('Exactitud Promedio'); plt.xlabel('Época'); plt.ylabel('Exactitud'); plt.legend(); plt.grid(True)

plt.subplot(1, 4, 2)
plt.plot(epochs_range, avg_loss, 'bo-', label='Entrenamiento')
plt.plot(epochs_range, avg_val_loss, 'ro-', label='Validación')
plt.title('Pérdida Promedio'); plt.xlabel('Época'); plt.ylabel('Pérdida'); plt.legend(); plt.grid(True)

plt.subplot(1, 4, 3)
plt.plot(epochs_range, avg_auc, 'bo-', label='Entrenamiento')
plt.plot(epochs_range, avg_val_auc, 'ro-', label='Validación')
plt.title('AUC Promedio'); plt.xlabel('Época'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)

# Matriz de confusión global
y_pred_final = (np.array(y_pred_probs_total) > 0.5).astype("int32")
cm = confusion_matrix(y_true_total, y_pred_final)
plt.subplot(1, 4, 4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases, annot_kws={"size": 14})
plt.title('Matriz de Confusión Global'); plt.ylabel('Etiqueta Verdadera'); plt.xlabel('Etiqueta Predicha')

plt.suptitle('Resultados del Entrenamiento y Evaluación del Modelo CNN', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

auc_global = roc_auc_score(y_true_total, y_pred_probs_total)
print(f"\nAUC Global (sobre todas las predicciones de validación): {auc_global:.4f}")