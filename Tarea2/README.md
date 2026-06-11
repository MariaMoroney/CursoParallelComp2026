# Tarea 2: Aprendizaje Distribuido para Computer Vision

Clasificación de imágenes de Maravillas del Mundo usando **JAX + Flax NNX** sobre GPU/TPU en Google Colab.

## Requisitos

- Cuenta de Google (para Colab)
- No se requiere API key de Kaggle: el dataset se descarga con `kagglehub`

## Instrucciones de ejecución

1. Abrir `Tarea2_JAX_FlaxNNX.ipynb` en Google Colab (Archivo → Subir notebook).
2. Activar el acelerador: `Runtime → Change runtime type → GPU (T4)` o `TPU`.
3. Ejecutar todas las celdas en orden (`Runtime → Run all`).
4. La primera celda instala las dependencias (`flax`, `optax`, `kagglehub`).
5. La celda de verificación imprime los dispositivos detectados — tomar captura de pantalla para el informe.

## Estructura del notebook

1. Configuración del entorno y verificación del acelerador
2. Descarga y preprocesamiento del dataset (64×64, normalización [0,1], split 70/15/15)
3. Modelo CNN en Flax NNX (Conv → BN → ReLU → MaxPool ×2 → Dense → Output)
4. Entrenamiento con `nnx.jit`, Adam y cross-entropy
5. Experimentos: tamaño de lote, learning rate, tamaño de red
6. Grid search de hiperparámetros
7. Análisis de precisión numérica (float32 / float16 / bfloat16)
8. Análisis, discusión y conclusiones

## Notas

- Semilla fija (`SEED = 42`) para reproducibilidad.
- El throughput se mide en imágenes/segundo; la memoria pico se reporta en GPU mediante `memory_stats()`.
- Tiempo estimado de ejecución completa en T4: ~30–60 min (depende de la grilla de hiperparámetros).