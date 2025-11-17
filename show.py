import os
import cv2
from tqdm import tqdm

# Ruta donde guardaste las letras descargadas
input_root = "./dataset3/letters_byclass_uppercase"
output_root = "./dataset3/letters_corrected"

os.makedirs(output_root, exist_ok=True)

# Recorrer carpetas A–Z
for letter in sorted(os.listdir(input_root)):
    input_dir = os.path.join(input_root, letter)
    output_dir = os.path.join(output_root, letter)
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc=f"Corrigiendo {letter}"):
        filepath = os.path.join(input_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Rotar 90° en sentido antihorario y reflejar horizontalmente
        corrected = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        corrected = cv2.flip(corrected, 1)

        cv2.imwrite(os.path.join(output_dir, filename), corrected)

print("✅ Rotación y corrección completadas.")
print("📁 Imágenes corregidas en:", output_root)
