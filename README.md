# NNI

Clasificación de Letras con HOG y Perceptrón Multicapa

Este proyecto implementa un sistema de clasificación de letras (A-Z) utilizando características HOG (Histogram of
Oriented Gradients) y un Perceptrón Multicapa (MLP) entrenado desde cero. El objetivo es reconocer letras en imágenes,
útil para aplicaciones como OCR básico o reconocimiento de caracteres.

Características

Extracción de características: Usa HOG con parámetros configurables (pixels_per_cell=(8,8), cells_per_block=(2,2)) para
imágenes de 64x64 píxeles.
Modelo: Perceptrón Multicapa con ReLU, Softmax, regularización L2, y mini-batches. Incluye guardado/carga del modelo
para predicciones futuras.
Evaluación: Reportes detallados con precisión, F1-score, y matriz de confusión.
Rendimiento: Alcanza ~55-70% de precisión en test (dependiendo de ajustes), con potencial para mejorar con data
augmentation o más datos.
Instalación

Requisitos del sistema:

Python 3.8+
Bibliotecas: numpy, scikit-image, scikit-learn, pillow (opencv opcional).
Instalar dependencias:

bash

Copy code
1pip install numpy scikit-image scikit-learn pillow
Clona o descarga el proyecto:

Coloca los scripts en un directorio, ej. nn_project/.
Asegúrate de tener el dataset de imágenes en ./dataset/ con subcarpetas A-Z.
Estructura del Proyecto

Copy code
1nn_project/
2├── dataset/ # Dataset de imágenes (A-Z subcarpetas)
3│ ├── A/
4│ ├── B/
5│ └── ...
6├── features/ # Archivos generados
7│ ├── caracteristicas_letras.npz # Características HOG extraídas
8│ └── modelo_entrenado.pkl # Modelo guardado
9├── main.py # Script principal (entrenamiento y evaluación)
10├── extract_hog.py # Script para extraer HOG (opcional, si separado)
11└── README.md # Este archivo
Uso

1. Preparar el Dataset

Organiza imágenes en ./dataset/A/, ./dataset/B/, etc. (formatos: PNG, JPG, etc.).
Cada imagen debe ser procesable (escala de grises, 64x64 recomendado).

2. Extraer Características HOG

Ejecuta el script para generar caracteristicas_letras.npz:

bash

Copy code
1python extract_hog.py # O integra en main.py
Parámetros: Imágenes a 64x64, HOG con 1764 features (sin PCA).

3. Entrenar el Modelo

Ejecuta main.py para cargar datos, entrenar y evaluar:

bash

Copy code
1python main.py
Configuraciones clave:
PCA: Reduce a 150 componentes (ajustable).
Modelo: 128 nodos ocultos, LR=0.01, regularización L2.
Entrenamiento: 3000 épocas max, batch_size=16, early stopping en error < 0.0001.
Salida: Precisión en train/test, reporte de clasificación, y guarda el modelo en ./features/modelo_entrenado.pkl.

4. Predecir una Imagen

Usa la función cargar_y_predecir en main.py:

python
7 lines
Copy code
Download code
Click to expand
from main import cargar_y_predecir
import numpy as np
...
El modelo se carga automáticamente, aplica preprocesamiento (scaler, PCA) y predice.
Resultados Esperados

Precisión: ~55-70% en test (mejorable con más datos o ajustes).
Ejemplo de salida:

Copy code
1Precisión General: 0.6500 (65.00%)
2Matriz de Confusión: [detalles por letra]
Clases difíciles: K, C, D (confusión con formas similares).
Dependencias Detalladas

numpy: Cálculos matriciales.
scikit-image: Extracción HOG.
scikit-learn: PCA, StandardScaler, métricas.
pillow: Carga de imágenes.
Mejoras y Contribuciones

Data Augmentation: Agrega rotaciones/escalas para más datos.
Ajustes: Prueba más capas, dropout, o optimizadores.
Contribuye: Abre issues o PRs en GitHub. Asegura compatibilidad con Python 3.8+.
Licencia

Este proyecto es de código abierto. Usa bajo MIT License. Atribuye si reutilizas.
