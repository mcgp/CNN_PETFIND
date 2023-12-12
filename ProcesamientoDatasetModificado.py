
# Importamos las bibliotecas necesarias
import os
import shutil
import numpy as np
from PIL import Image
from icrawler.builtin import BingImageCrawler
from bing_image_downloader import downloader
import glob


# Función para redimensionar y renombrar las imágenes
def process_image(file_path, label, output_dir, img_id):
    img = Image.open(file_path)
    img = img.resize((128, 128))
    new_filename = f"{output_dir}/{label}/{label}_{img_id}.jpg"
    os.makedirs(os.path.dirname(new_filename), exist_ok=True)
    img.save(new_filename)
    os.remove(file_path)
def descargar_imagenes(keyword, output_directory_bing, output_directory_icrawler):
    # Descargamos las imágenes con bing-image-downloader
    downloader.download(keyword, limit=30,  output_dir=output_directory_bing, adult_filter_off=False, force_replace=False, timeout=100)

    # Creamos los crawlers de icrawler
    bing_crawler = BingImageCrawler(storage={"root_dir": output_directory_icrawler})

    # Descargamos las imágenes con icrawler
    bing_crawler.crawl(keyword=keyword, max_num=50)


def obtener_rutas_imagenes(output_directory_bing, output_directory_icrawler):
    # Obtenemos las rutas de las imágenes descargadas
    all_files = [os.path.join(output_directory_bing, f) for f in os.listdir(output_directory_bing) if os.path.isfile(os.path.join(output_directory_bing, f))]
    all_files += [os.path.join(output_directory_icrawler, f) for f in os.listdir(output_directory_icrawler) if os.path.isfile(os.path.join(output_directory_icrawler, f))]
    return all_files

def organizar_imagenes(all_files, output_directory):
    # Organizamos las imágenes
    for img_id, img_path in enumerate(all_files):
        split_path = img_path.split(os.sep)
        if len(split_path) < 2:
            print(f"Ruta de imagen inválida: {img_path}")
            continue
        label = split_path[-2]
        process_image(img_path, label, output_directory, img_id)

def dividir_datos_prueba(image_paths, test_pct):
    # Mezclamos las rutas de las imágenes
    np.random.shuffle(image_paths)

    # Obtenemos el número de imágenes para la carpeta de prueba
    num_images = int(test_pct * len(image_paths))

    # Obtenemos las rutas de las imágenes para la carpeta de prueba
    test_image_paths = image_paths[:num_images]

    return test_image_paths

def mover_imagenes_prueba(test_image_paths, output_directory):
    # Movemos las imágenes a la carpeta de prueba
    for img_path in test_image_paths:
        label = img_path.split(os.sep)[-2]
        output_subdirectory = os.path.join(output_directory, label)
        
        # Creamos el subdirectorio si no existe
        os.makedirs(output_subdirectory, exist_ok=True)
        
        # Imprimimos la ruta de origen y destino
        print(f"Moving {img_path} to {os.path.join(output_subdirectory, os.path.basename(img_path))}")
        
        shutil.move(img_path, os.path.join(output_subdirectory, os.path.basename(img_path)))


# Definimos los directorios de salida
output_directory_perro_bing = 'datos/entrenamiento/dog'
output_directory_gato_bing = 'datos/entrenamiento/cat'
output_directory_perro_icrawler = 'datos/entrenamiento/dog/dog'
output_directory_gato_icrawler = 'datos/entrenamiento/cat/cat'
"""
# Descargamos las imágenes
descargar_imagenes("dog", output_directory_perro_bing, output_directory_perro_icrawler)
descargar_imagenes("cat", output_directory_gato_bing, output_directory_gato_icrawler)
"""
# Obtenemos las rutas de las imágenes descargadas
all_files_perro = obtener_rutas_imagenes(output_directory_perro_bing, output_directory_perro_icrawler)
all_files_gato = obtener_rutas_imagenes(output_directory_gato_bing, output_directory_gato_icrawler)

# Organizamos las imágenes
#organizar_imagenes(all_files_perro, "datos/entrenamiento")
#organizar_imagenes(all_files_gato, "datos/entrenamiento")

# Obtenemos las rutas de todas las imágenes descargadas
image_paths = []
for root, dirs, files in os.walk("datos/entrenamiento"):
    for filename in files:
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            image_paths.append(os.path.join(root, filename))

# Imprimimos las rutas de las imágenes
print(f"Image paths: {image_paths}")

# Dividimos los datos en conjuntos de entrenamiento y prueba
test_image_paths = dividir_datos_prueba(image_paths, 0.2)

# Imprimimos las rutas de las imágenes de prueba
print(f"Test image paths: {test_image_paths}")

# Movemos las imágenes a la carpeta de prueba
mover_imagenes_prueba(test_image_paths, "datos/prueba")
