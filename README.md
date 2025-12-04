## 📌 **Descripción general**

Este repositorio contiene el módulo principal del **Visor Computacional**, un sistema desarrollado en Python que permite identificar condiciones fitosanitarias en hojas de arándano mediante técnicas de visión computacional y aprendizaje profundo.

El sistema procesa imágenes, ejecuta un modelo convolucional previamente entrenado y genera mapas de calor explicativos mediante **Grad-CAM**.

Este repositorio corresponde estrictamente al **módulo operativo** del visor, que incluye inferencia, preprocesamiento y servidor Flask.

---

## ⚙️ **Componentes incluidos**

Dentro del directorio `app/` se encuentran los scripts esenciales:

- **`app.py`** – Servidor Flask que recibe imágenes, ejecuta la predicción y retorna resultados en formato JSON.
- **`predict.py`** – Módulo encargado de la inferencia utilizando el modelo entrenado.
- **`gradcam_viz.py`** – Generación de mapas de calor Grad-CAM para interpretabilidad.
- **`calibrar_hsv.py`** – Utilidad para calibrar el rango de segmentación HSV aplicado en el preprocesamiento.
- **`visor_app.py`** – Script para pruebas locales del visor sin necesidad de API.
- **`train_model.py`** – Archivo que contiene la lógica de entrenamiento del modelo (incluido a modo referencial).

> El repositorio no contiene el dataset ni el modelo para evitar un peso excesivo.
> 
> 
> El modelo `.h5` debe ubicarse en `app/models/` para ejecutar correctamente las predicciones.
> 

---

## 🔧 **Requisitos**

Instalar dependencias principales:

```bash
pip install tensorflow opencv-python numpy flask pillow matplotlib

```

---

## ▶️ **Ejecución**

### 1. Iniciar el servidor Flask:

```bash
cd app
python app.py

```

El servicio quedará disponible en:

```
http://127.0.0.1:5000/predict

```

### 2. Enviar una imagen (ejemplo con cURL):

```bash
curl -X POST -F "image=@hoja.jpg" http://127.0.0.1:5000/predict

```

---

## 🧠 **Tecnologías empleadas**

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow
- Flask

---

## 🎯 **Objetivo del módulo**

Brindar una herramienta computacional capaz de:

- Analizar imágenes de hojas.
- Clasificar su condición fitosanitaria.
- Mostrar visualmente las regiones relevantes mediante Grad-CAM.

Este visor está diseñado para integrarse con aplicaciones móviles desarrolladas en Flutter, facilitando diagnósticos inmediatos en campo.
