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
