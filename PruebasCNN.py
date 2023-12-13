import cv2
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np

clases=["dog", "cat"]

ancho=128  # Asegúrate de que estas dimensiones coincidan con las de tus imágenes de entrenamiento
alto=128

ruta_modelo = "modelos/modeloA.h5"  # Reemplaza esto con la ruta a tu modelo
miModeloCNN = load_model(ruta_modelo, custom_objects={'Custom>Adam': Adam})

imagen=cv2.imread("datos/prueba/cat/cat_53.jpg")  # Reemplaza esto con la ruta a tu imagen de prueba

# Redimensionar la imagen a (ancho, alto, 3)
imagen_redimensionada = cv2.resize(imagen, (ancho, alto))

# Normalizar los valores de los píxeles a [0, 1]
imagen_normalizada = imagen_redimensionada / 255.0

# Asegurarse de que la imagen tiene la forma correcta
imagen_normalizada = np.expand_dims(imagen_normalizada, axis=0)

claseResultado = miModeloCNN.predict(imagen_normalizada)
claseResultado = np.argmax(claseResultado)  # Obtener el índice de la clase con la mayor probabilidad

print("La imagen cargada es un",clases[claseResultado])

while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()