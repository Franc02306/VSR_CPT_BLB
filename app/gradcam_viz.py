import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# === CARGAR MODELO ===
model_path = "../models/model_arandano_vgg16.h5"
model = tf.keras.models.load_model(model_path)

# === CLASES ===
classes = [
    "Blueberry_AnthracnoseLeaf",
    "Blueberry_ExobasidiumLeafSpot",
    "Blueberry_Healthy",
    "Blueberry_SeptoriaLeafSpot"
]

# === FUNCIÓN PARA GENERAR HEATMAP (Grad-CAM) ===
def make_gradcam_heatmap(img_array, model):
    # Submodelo base (VGG16)
    vgg_model = model.get_layer("vgg16")
    last_conv_layer_name = "block5_conv3"

    print(f"🔍 Usando submodelo: {vgg_model.name}, capa: {last_conv_layer_name}")

    # Modelo que va desde la entrada hasta la última capa convolucional
    grad_model = tf.keras.models.Model(
        inputs=vgg_model.input,
        outputs=[vgg_model.get_layer(last_conv_layer_name).output, vgg_model.output]
    )

    # Predicción de la clase
    preds = model.predict(img_array)
    class_idx = tf.argmax(preds[0])

    # Calcular gradientes de la clase respecto a la última capa conv
    with tf.GradientTape() as tape:
        conv_outputs, vgg_features = grad_model(img_array)
        tape.watch(conv_outputs)
        # Pasamos por las capas densas manualmente (sin errores de forma)
        x = tf.reshape(vgg_features, (vgg_features.shape[0], -1))
        for layer in model.layers[2:]:  # saltar vgg16 y Flatten()
            x = layer(x)
        output = x[:, class_idx]

    grads = tape.gradient(output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# === FUNCIÓN PARA MOSTRAR RESULTADO ===
def mostrar_gradcam(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    preds = model.predict(img_array)
    clase_pred = classes[np.argmax(preds)]
    conf = np.max(preds) * 100

    # Obtener heatmap
    heatmap = make_gradcam_heatmap(img_array, model)

    # Redimensionar heatmap a tamaño de imagen original
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Aplicar colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superponer con transparencia
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

    # Mostrar resultados
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original\n{clase_pred} ({conf:.2f}%)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Mapa de calor (Grad-CAM)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# === PRUEBA ===
ruta = "../dataset/valid/Blueberry_AnthracnoseLeaf/Blueberry-Anthracnose-Leaf26_jpg.rf.f75182aa3599e8790a7e1cbf4f030a3d.jpg"
if os.path.exists(ruta):
    mostrar_gradcam(ruta)
else:
    print("⚠️ Cambia la ruta a una imagen real para probar.")
