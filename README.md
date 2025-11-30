📌 Descripción del proyecto

El presente proyecto implementa un visor computacional basado en técnicas de visión por computadora y aprendizaje profundo, diseñado para identificar enfermedades presentes en hojas de arándano (Vaccinium corymbosum).
El sistema emplea un modelo convolucional entrenado con TensorFlow y utiliza Grad-CAM para generar mapas de calor que expliquen visualmente las regiones relevantes para la predicción.

Este visor funciona como un módulo independiente que puede integrarse fácilmente con aplicaciones móviles u otros sistemas que requieran un análisis automatizado de imágenes.

🧠 Características principales

Clasificación de hojas de arándano en múltiples categorías (saludable y plagas específicas).

Preprocesamiento avanzado:

Normalización

Escalado automático

Segmentación aproximada por color (HSV calibrado)

Explicabilidad mediante Grad-CAM.

Exportación del modelo en formato .h5.

API basada en Flask para recibir imágenes y responder con el análisis.

Estructura modular que separa:

Entrenamiento

Predicción

Procesamiento visual

Servidor backend

📂 Estructura del proyecto
VISOR-COMPUTACIONAL/
│── app/
│   ├── app.py                 # Servidor Flask
│   ├── train_model.py         # Script de entrenamiento del modelo
│   ├── predict.py             # Realiza la inferencia
│   ├── gradcam_viz.py         # Generación de Grad-CAM
│   ├── calibrar_hsv.py        # Calibración de segmentación HSV
│   └── visor_app.py           # Ejecutor principal para pruebas locales
│
│── dataset/
│   ├── train/                 # Imágenes de entrenamiento
│   └── valid/                 # Imágenes de validación
│
│── models/
│   └── model_arandano_vgg16.h5 # Modelo entrenado
│
│── venv/                      # Entorno virtual (opcional)
│── .gitignore
