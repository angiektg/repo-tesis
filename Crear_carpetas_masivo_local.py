import os
import shutil

# -------------------------------
# Configura tus rutas locales aquí
# -------------------------------
# Carpeta con los archivos desde los cuales se extraerá el nombre base (para crear carpetas)
origen_nombres = r'E:\Angie\Experimento3\Sanas\Sanas_Sin_Rotar'

# Carpeta donde se crearán las carpetas
destino_base = r'E:\Angie\Experimento3\Sanas'

# Carpeta con los archivos que se copiarán a cada carpeta correspondiente
origen_archivos = r'E:\Angie\Experimento3\Sanas\Sanas_general'

# -------------------------------
# Crear carpetas masivamente
# -------------------------------
contenido = os.listdir(origen_nombres)

for nombre in contenido:
    if 'F' in nombre:  # Asegura que el formato esperado exista
        nombre_base = nombre.split('F')[0]
        carpeta_destino = os.path.join(destino_base, nombre_base)
        try:
            os.mkdir(carpeta_destino)
            print(f'Carpeta creada: {carpeta_destino}')
        except FileExistsError:
            print(f'La carpeta ya existe: {carpeta_destino}')

# -------------------------------
# Copiar archivos a las carpetas correspondientes
# -------------------------------
contenido_archivos = os.listdir(origen_archivos)
comandos = []

for nombre in contenido_archivos:
    if 'F' in nombre:
        fuente = os.path.join(origen_archivos, nombre)
        nombre_base = nombre.split('F')[0]
        destino = os.path.join(destino_base, nombre_base, nombre)

        try:
            shutil.copy(fuente, destino)
            comandos.append(f"Copiado: {fuente} -> {destino}")
        except Exception as e:
            print(f"Error al copiar {fuente}: {e}")

# Verifica una muestra del resultado
if comandos:
    print('\nEjemplo de copia realizada:\n', comandos[0])
