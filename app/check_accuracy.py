import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# === Cargar modelo ===
model = tf.keras.models.load_model("../models/model_arandano_vgg16.h5")

# === MISMO PREPROCESADO QUE EN ENTRENAMIENTO ===
def crop_leaf_preprocess(img):
    img_uint8 = img.astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    lower_green = np.array([13, 64, 30])
    upper_green = np.array([108, 240, 207])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 20:
            img_uint8 = img_uint8[y:y + h, x:x + w]

    img_uint8 = cv2.resize(img_uint8, (224, 224))
    return img_uint8.astype(np.float32) / 255.0


# === Generador VALIDACIÓN ===
val_dir = "../dataset/valid"

datagen = ImageDataGenerator(
    preprocessing_function=crop_leaf_preprocess
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# === EVALUAR ===
loss, acc = model.evaluate(val_gen)

print("\n==============================")
print(f"🔥 ACCURACY REAL DEL MODELO: {acc*100:.2f}%")
print(f"📉 LOSS: {loss:.4f}")
print("==============================")
