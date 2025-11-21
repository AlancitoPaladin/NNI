"""
Perceptrón Multicapa para predicción basado en características HOG.
Versión optimizada con mini-batches, regularización L2.
"""

import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


def cargar_datos_hog(file_path, reduce_dim=True, n_components=200):
    """Carga y preprocesa datos HOG."""
    data = np.load(file_path)
    X = data['X'].astype(np.float32)
    y = data['y']

    print(f"Datos cargados: X shape {X.shape}, y shape {y.shape}")

    # Convertir etiquetas a enteros
    unique_labels = np.unique(y)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    y_int = np.array([label_to_int[label] for label in y], dtype=np.int32)

    print(f"Número de clases: {len(unique_labels)}")

    # Normalizar
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # PCA
    pca = None
    if reduce_dim and X.shape[1] > n_components:
        print(f"Aplicando PCA: {X.shape[1]} -> {n_components}")
        pca = PCA(n_components=n_components, random_state=42)
        X = pca.fit_transform(X).astype(np.float32)

    # One-hot encoding
    y_one_hot = np.zeros((len(y_int), len(unique_labels)), dtype=np.float32)
    y_one_hot[np.arange(len(y_int)), y_int] = 1

    return X, y_one_hot, y_int, label_to_int, int_to_label, scaler, pca


class PerceptronMulticapa:
    """Perceptrón multicapa con regularización L2."""

    def __init__(self, nodos_ent, nodos_ocu, nodos_sal, learning_rate=0.01, lambda_reg=0.0005):
        self.nodos_ent = nodos_ent
        self.nodos_ocu = nodos_ocu
        self.nodos_sal = nodos_sal
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

        self.W1 = np.random.randn(nodos_ent, nodos_ocu) * np.sqrt(2.0 / nodos_ent)
        self.b1 = np.zeros((1, nodos_ocu))
        self.W2 = np.random.randn(nodos_ocu, nodos_sal) * np.sqrt(2.0 / nodos_ocu)
        self.b2 = np.zeros((1, nodos_sal))

    def relu(self, x):
        return np.maximum(0, np.clip(x, -500, 500))

    def drelu(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]

        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m + self.lambda_reg * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.W2.T)
        dz1 = self.drelu(self.z1) * da1
        dW1 = np.dot(X.T, dz1) / m + self.lambda_reg * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def entrenar_batch(self, X, y, epochs=100, batch_size=32, verbose=True):
        n_samples = X.shape[0]
        errores = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            error_epoch = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

                error_epoch += np.mean((y_batch - output) ** 2)
                n_batches += 1

            error_promedio = error_epoch / n_batches
            errores.append(error_promedio)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Época {epoch + 1}/{epochs}, Error: {error_promedio:.10f}")


            if error_promedio < 0.0000001:
                print(f"Convergencia en época {epoch + 1}")
                break

        return errores

    def predecir(self, X):
        return np.argmax(self.forward(X), axis=1)

    def evaluar(self, X, y):
        predicciones = self.predecir(X)
        etiquetas_reales = np.argmax(y, axis=1)
        return np.mean(predicciones == etiquetas_reales)


def generar_reporte_clasificacion(perceptron, X_test, y_test, int_to_label):
    """Genera reporte completo de clasificación."""
    y_pred = perceptron.predecir(X_test)
    y_true = np.argmax(y_test, axis=1)

    y_true_labels = [int_to_label[idx] for idx in y_true]
    y_pred_labels = [int_to_label[idx] for idx in y_pred]

    accuracy = accuracy_score(y_true, y_pred)

    print("Reporte de clasificación")
    print("=" * 80)
    print(f"\nPrecisión General: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

    print(classification_report(y_true_labels, y_pred_labels))

    print("\nMatriz de Confusión:")
    print("-" * 80)
    cm = confusion_matrix(y_true, y_pred)
    labels = [int_to_label[idx] for idx in range(len(int_to_label))]

    print("Real ↓ / Pred →", end="")
    for label in labels:
        print(f"{label:>6}", end="")
    print()

    for i, label in enumerate(labels):
        print(f"{label:<15}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>6}", end="")
        print()

    return accuracy, cm


def cargar_y_predecir(vector_features, modelo_path='./features/modelo_entrenado.pkl'):
    """
    Carga el modelo guardado y predice la clase para un vector de características HOG.

    Args:
    - vector_features: np.array de shape (n_features,) - características de una imagen.
    - modelo_path: str - ruta al archivo .pkl del modelo.

    Returns:
    - str: Letra predicha (ej. 'A').
    """
    with open(modelo_path, 'rb') as f:
        modelo_data = pickle.load(f)

    perceptron = modelo_data['perceptron']
    scaler = modelo_data['scaler']
    pca = modelo_data['pca']
    int_to_label = modelo_data['int_to_label']

    # Preprocesar el vector
    vector_features = scaler.transform(vector_features.reshape(1, -1))
    if pca is not None:
        vector_features = pca.transform(vector_features)

    # Predicción
    output = perceptron.forward(vector_features)
    clase_predicha_idx = np.argmax(output, axis=1)[0]
    letra_predicha = int_to_label[clase_predicha_idx]

    return letra_predicha


if __name__ == "__main__":
    # Cargar datos
    print("Cargando datos")
    X, y, y_int, label_to_int, int_to_label, scaler, pca = cargar_datos_hog(
        './features/proof.npz',
        reduce_dim=True,
        n_components=150
    )

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print(f"\nEntrenamiento: {X_train.shape} | Prueba: {X_test.shape}\n")

    # Entrenar
    print("Entrenando modelo")

    perceptron = PerceptronMulticapa(
        nodos_ent=X_train.shape[1],
        nodos_ocu=128,
        nodos_sal=y_train.shape[1],
        learning_rate=0.01
    )

    perceptron.entrenar_batch(X_train, y_train, epochs=3000, batch_size=16)

    # Evaluar
    print("Evaluación")
    print("=" * 80)

    accuracy_train = perceptron.evaluar(X_train, y_train)
    accuracy_test = perceptron.evaluar(X_test, y_test)

    generar_reporte_clasificacion(perceptron, X_test, y_test, int_to_label)

    print("\n" + "=" * 80)
    print("Guardando modelo entrenado...")

    modelo_data = {
        'perceptron': perceptron,
        'scaler': scaler,
        'pca': pca,
        'int_to_label': int_to_label,
        'label_to_int': label_to_int,
        'n_components': 150
    }

    with open('models/trained_model.pkl', 'wb') as f:
        pickle.dump(modelo_data, f)

    print("Modelo guardado en './models/trained_model.pkl'")

    # Supón que tienes un vector HOG de una imagen (extraído previamente)
    # ejemplo_vector = np.random.rand(1764)  # Reemplaza con tu vector real
    # resultado = cargar_y_predecir(ejemplo_vector)
    # print(f"Letra predicha: {resultado}")