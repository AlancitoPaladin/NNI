# predict.py
import os
import pickle
import sys

import numpy as np
from PIL import Image
from skimage.feature import hog


def predecir_imagen(ruta_imagen):
    """
    Predice la letra de una imagen usando el modelo entrenado.

    Args:
        ruta_imagen: ruta a la imagen (ej: './dataset/A/img1.png')

    Returns:
        letra_predicha: letra predicha (A-Z)
        confianza: probabilidad de la predicción (0-1)
    """

    print("Cargando modelo entrenado...")
    with open('./models/trained_model.pkl', 'rb') as f:
        modelo_data = pickle.load(f)

    perceptron = modelo_data['perceptron']
    scaler = modelo_data['scaler']
    pca = modelo_data['pca']
    int_to_label = modelo_data['int_to_label']

    print(f"Procesando imagen: {ruta_imagen}")

    if not os.path.exists(ruta_imagen):
        print(f" Error: La imagen '{ruta_imagen}' no existe")
        return None, None

    try:
        # Cargar imagen
        img = Image.open(ruta_imagen).convert('L')

        # Redimensionar a 64x64
        img = img.resize((64, 64))

        # Convertir a array
        img_array = np.array(img)

        # Extraer características HOG
        hog_features = hog(img_array, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

        # Normalizar
        hog_features = hog_features.reshape(1, -1)  # Hacer 2D
        hog_features = scaler.transform(hog_features)

        # Aplicar PCA
        if pca is not None:
            hog_features = pca.transform(hog_features)

        output = perceptron.forward(hog_features)
        prediccion_idx = np.argmax(output, axis=1)
        confianza = np.max(output, axis=1)

        letra_predicha = int_to_label[prediccion_idx[0]]
        confianza_valor = confianza[0]

        return letra_predicha, confianza_valor

    except Exception as e:
        print(f" Error procesando imagen: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Ejemplo de uso
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        # Prueba con una imagen de ejemplo
        ruta = './dataset/A/img277_1.1_cell.png'

    letra, confianza = predecir_imagen(ruta)

    if letra:
        print("\n" + "=" * 80)
        print(f" Predicción: {letra}")
        print(f"   Confianza: {confianza * 100:.2f}%")
        print("=" * 80)
    else:
        print(" No se pudo realizar la predicción")
