import tensorflow as tf
import numpy as np
import cv2
import os

# Cargar modelo
model_path = "../models/model_arandano_vgg16.h5"
model = tf.keras.models.load_model(model_path)

# Clases (mismo orden que tus carpetas)
classes = [
    "Blueberry_AnthracnoseLeaf",
    "Blueberry_ExobasidiumLeafSpot",
    "Blueberry_Healthy",
    "Blueberry_SeptoriaLeafSpot"
]

def predecir_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0

    pred = model.predict(img)
    clase = classes[np.argmax(pred)]
    confianza = np.max(pred) * 100

    print(f"Predicción: {clase} ({confianza:.2f}% de confianza)")

# Ejemplo: cambiar por una imagen real
ruta = "../dataset/valid/Blueberry_SeptoriaLeafSpot/Blueberry-Septoria-Leaf-Spot60_jpg.rf.518e23e8f3db6d71faf79193c09e5026.jpg"
if os.path.exists(ruta):
    predecir_imagen(ruta)
else:
    print("⚠️ Cambia la ruta a una imagen real para probar.")