from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image

app = Flask(__name__)

# Carga del modelo
model = tf.keras.models.load_model("../models/model_arandano_vgg16.h5")
_ = model(tf.zeros((1, 224, 224, 3)), training=False)

classes = [
    "Blueberry_AnthracnoseLeaf",
    "Blueberry_ExobasidiumLeafSpot",
    "Blueberry_Healthy",
    "Blueberry_SeptoriaLeafSpot"
]

def make_gradcam_heatmap(img_array, model):
    # 1) localizar base
    try:
        base = model.get_layer("vgg16")
    except ValueError:
        base = model.layers[0]

    # 2) capa objetivo
    target = None
    for lyr in reversed(base.layers):
        if isinstance(lyr, tf.keras.layers.Conv2D):
            target = lyr
            break
    if target is None:
        raise RuntimeError("No se encontró capa Conv2D.")

    # 3) crear pipeline explícito: base -> resto del modelo
    base_out = base.output
    x = base_out
    for lyr in model.layers[1:]:  # saltamos la VGG16 base
        x = lyr(x)
    full_out = x

    grad_model = tf.keras.Model(
        inputs=base.input,
        outputs=[target.output, full_out]
    )

    # 4) gradientes
    x = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise RuntimeError("Gradientes nulos: asegúrate de que el modelo está conectado correctamente.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_mean(conv_out * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files["file"]

    # 1) Leer en RGB y preprocesar como en el entrenamiento
    img_rgb = np.array(Image.open(file.stream).convert("RGB"))
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = (np.expand_dims(img_resized, axis=0).astype("float32")) / 255.0

    # 2) Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model)
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))

    # 3) (Opcional) Enmascarar el fondo para forzar foco en la hoja
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (25, 20, 20), (95, 255, 255))           # verde aprox
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    heatmap = heatmap * (mask.astype(np.float32) / 255.0)

    # 4) Colormap y superposición en RGB
    cmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_TURBO)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img_rgb, 0.65, cmap, 0.45, 0)

    # 5) Respuesta
    _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
