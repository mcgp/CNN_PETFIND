"""
Este código es para entrenar un modelo de clasificación de imágenes para distinguir entre imágenes de perros y gatos. 
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2
import os

"""
Esta función carga las imágenes de la ruta dada, las redimensiona a las dimensiones dadas y las normaliza.
También crea una etiqueta de categoría para cada imagen (1 para 'dog' y 0 para 'cat'). Finalmente, devuelve las imágenes y las etiquetas como arrays de numpy.
"""
def cargarDatos(rutaOrigen, numeroCategorias, ancho, alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in os.listdir(rutaOrigen):
        for imagen in os.listdir(os.path.join(rutaOrigen, categoria)):
            ruta = os.path.join(rutaOrigen, categoria, imagen)
            imagen = cv2.imread(ruta)
            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen / 255.0
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[int(categoria == 'dog')] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados

ancho=128
alto=128
numeroCanales=3  # Cambiado a 3 para usar VGG16
formaImagen=(ancho,alto,numeroCanales)
numeroCategorias=2

imagenes, probabilidades=cargarDatos("datos/entrenamiento", numeroCategorias, ancho, alto)

# Crear un generador de imágenes con aumento de datos
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Esto es para dividir tus datos en entrenamiento y validación
)

# Ajustar los datos al generador
datagen.fit(imagenes)

# Usar VGG16 como base del modelo
base_model = VGG16(weights='imagenet', include_top=False, input_shape=formaImagen)
base_model.trainable = False  # No entrenar las capas de VGG16

model=Sequential()
model.add(base_model)  # Añadir VGG16 como primera capa

model.add(Flatten())
model.add(Dense(256, activation='relu'))  # Aumentar el número de nodos
model.add(Dropout(0.5))  # Mantener la capa de Dropout

model.add(Dense(numeroCategorias, activation='softmax'))

# Ajustar la tasa de aprendizaje
optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenar el modelo utilizando el generador de imágenes
model.fit(datagen.flow(imagenes, probabilidades, batch_size=60, subset='training'),
          validation_data=datagen.flow(imagenes, probabilidades, batch_size=60, subset='validation'),
          epochs=30)

imagenesPrueba, probabilidadesPrueba=cargarDatos("datos/prueba", numeroCategorias, ancho, alto)
resultados=model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
print("Accuracy=",resultados[1])

# Guardar el modelo en formato HDF5
ruta_hdf5 = "modelos/modeloA.h5"
model.save(ruta_hdf5)

# Guardar el modelo en el formato nativo de Keras
ruta_keras = "modelos/modeloA.keras"
model.save(ruta_keras)