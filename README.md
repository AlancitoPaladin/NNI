# NNI — Letter Classification with HOG + MLP

A compact, from-scratch machine learning project that recognizes uppercase letters (A–Z) from images. It uses HOG (Histogram of Oriented Gradients) features and a custom multilayer perceptron (MLP) for classification, making it a clean, interpretable pipeline for classic OCR-style tasks.

## Highlights
- Feature extraction with HOG on 64×64 grayscale images
- Custom MLP with ReLU + Softmax, L2 regularization, and mini-batch training
- PCA support to reduce feature dimensionality
- Evaluation with accuracy, classification report, and confusion matrix
- Saved model for easy reuse in predictions

## Tech Stack
- Python 3.8+
- NumPy, scikit-image, scikit-learn, Pillow
- Optional: OpenCV (for dataset cleanup scripts)

## Project Structure
```
.
├── features/
│   └── extract.py              # HOG feature extraction -> proof.npz
├── models/
│   └── (model output)
├── main.py                     # Train + evaluate MLP
├── predict.py                  # Load model and predict a single image
├── show.py                     # Optional image rotation/correction helper
└── README.md
```

## Quick Start
### 1) Install dependencies
```bash
pip install numpy scikit-image scikit-learn pillow
```

### 2) Prepare the dataset
Organize images in `./dataset/` with one folder per letter:
```
./dataset/
├── A/
├── B/
└── ...
```

Images should be grayscale or convertible to grayscale. The pipeline resizes them to 64×64.

### 3) Extract HOG features
```bash
python features/extract.py
```
This creates `proof.npz` (features + labels). Move it into `./features/` so training can find it, or update the path in `main.py`.

### 4) Train the model
```bash
python main.py
```
The script reports accuracy and saves the trained model to `./models/modelo_entrenado.pkl`.

### 5) Predict a single image
```bash
python predict.py ./dataset/H/img277_6.8_tight.png
```
Output includes the predicted letter and confidence score.

## Notes
- Performance depends on dataset quality and class balance.
- You can improve accuracy with augmentation or a deeper network.
- PCA is optional and configurable in `main.py`.

## Why this project matters (CV-friendly)
This project demonstrates a full classical ML workflow: data preparation, feature engineering, model training from scratch, evaluation, and deployment-ready inference. It’s a clear, lightweight alternative to deep learning for OCR-style problems and shows hands-on understanding of the math and pipeline design.

## License
MIT License.
