import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import os

# === RUTAS DE DATOS ===
train_dir = "../dataset/train"
val_dir = "../dataset/valid"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# === GENERADORES DE DATOS ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# === CARGAR BASE PREENTRENADA (VGG16) ===
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# === DESCONGELAR SOLO LAS CAPAS FINALES ===
for layer in base_model.layers:
    if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
        layer.trainable = True
    else:
        layer.trainable = False

# === CONSTRUCCIÓN DEL MODELO ===
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# === COMPILAR CON TASA DE APRENDIZAJE BAJA ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === ENTRENAMIENTO ===
history = model.fit(
    train_gen,
    epochs=8,               # Fine-tuning: pocas épocas para ajustar
    validation_data=val_gen
)

# === GUARDAR MODELO ===
os.makedirs("../models", exist_ok=True)
model.save("../models/model_arandano_vgg16.h5")

print("✅ Modelo VGG16 ajustado y guardado en /models/model_arandano_vgg16.h5")
