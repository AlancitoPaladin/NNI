import os
import string
import numpy as np
from PIL import Image
from skimage.feature import hog

features_list = []
labels_list = []

base_dir = '../dataset'

# Iterar sobre las letras A-Z
for letter in string.ascii_uppercase:
    letter_dir = os.path.join(base_dir, letter)

    if os.path.exists(letter_dir):
        for filename in os.listdir(letter_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(letter_dir, filename)

                try:
                    # Cargar la imagen con Pillow y convertir a escala de grises
                    img = Image.open(img_path).convert('L')

                    # Reescalar a 64x64 píxeles
                    img = img.resize((64, 64))

                    # Convertir a array de NumPy
                    img_array = np.array(img)

                    # Extraer características HOG
                    hog_features = hog(img_array, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

                    # Agregar a las listas
                    features_list.append(hog_features)
                    labels_list.append(letter)

                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")
                    continue

# Convertir las listas a arrays de NumPy
X = np.array(features_list)
y = np.array(labels_list)

# Guardar en un archivo .npz
np.savez('proof.npz', X=X, y=y)

print("Procesamiento completado. Archivo 'caracteristicas_letras.npz' guardado.")
