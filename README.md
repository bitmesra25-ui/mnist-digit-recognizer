# Handwritten Digit Recognizer

A CNN-based machine learning project that classifies handwritten digits using the MNIST dataset, with a live interactive web app that can recognize both single digits and full multi-digit numbers.

## Live Demo

Try it here: [dhriti-digit-recognizer.streamlit.app](https://dhriti-digit-recognizer.streamlit.app/)

## Overview

This project implements and compares multiple approaches to handwritten digit recognition:
- A baseline Artificial Neural Network (ANN)
- A Convolutional Neural Network (CNN) with Dropout regularization
- Data augmentation experiments to improve real-world generalization
- A live drawing canvas web app for real-time digit prediction
- OpenCV-based digit segmentation to recognize full multi-digit numbers (e.g. "18"), not just single digits

## Results

| Model | Test Accuracy |
|-------|---------------|
| Baseline ANN | 97.7% |
| CNN + Dropout | 99.2% |
| CNN + Data Augmentation | 99.3% |

## Key Findings

- **CNN vs ANN**: The CNN outperformed the baseline ANN by ~1.5%, since convolutional layers capture spatial patterns (edges, curves) that a flat dense network cannot.
- **Overfitting analysis**: The ANN showed a 1.69% train-test accuracy gap, while adding Dropout to the CNN reduced this gap to 0.43%, indicating better generalization.
- **Confusion Matrix**: The most common misclassification was digit 2 being predicted as 7, likely due to visual similarity when '2' is written quickly without a clear curve.
- **Real-world testing**: Testing the model on my own handwriting revealed a "domain gap" — thin pen strokes were classified with much lower confidence (~22%) compared to bold/thick strokes (~77%), since MNIST training images use thick strokes.
- **Data augmentation**: Applying rotation, shift, and zoom augmentation improved general test accuracy slightly, but did not fully resolve the thin-stroke issue, highlighting that augmentation strategy must match the specific problem (stroke thickness vs. position/rotation).

## Tech Stack

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit (web app)
- streamlit-drawable-canvas (interactive canvas)
- OpenCV (digit segmentation for multi-digit recognition)

## Project Structure

```
digit-recognizer/
├── app.py                  # Streamlit web app
├── digit_model.h5          # Trained CNN model
├── requirements.txt        # Python dependencies
├── mnist_digit_recognizer.ipynb   # Full training notebook
└── README.md
```

## How to Run Locally

1. Clone this repository
```
git clone https://github.com/bitmesra25-ui/mnist-digit-recognizer.git
cd mnist-digit-recognizer
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Run the app
```
streamlit run app.py
```

4. Open the local URL shown in the terminal (usually `http://localhost:8501`)

## Model Architecture

```
Conv2D(32, 3x3, relu) → MaxPooling2D
Conv2D(64, 3x3, relu) → MaxPooling2D
Flatten
Dense(128, relu) → Dropout(0.3)
Dense(10, softmax)
```

## Dataset

MNIST dataset (60,000 training images, 10,000 test images), sourced in CSV format, each image being 28x28 grayscale pixels.

## Limitations

- Digit segmentation assumes digits are written with clear gaps between them; touching or overlapping digits may be detected as a single region.
- Works best with bold, thick strokes, consistent with MNIST's training style; very thin strokes reduce prediction confidence.

## Future Improvements

- Train with stroke-thickness-specific augmentation to close the thin-stroke accuracy gap
- Deploy with a persistent, custom domain
- Extend to recognize digits from a live camera feed using OpenCV

## Author

Rohit Ranjan — EEE undergraduate, Birla Institute of Technology, Mesra
