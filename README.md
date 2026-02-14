# 🍎 Real-Time Fruit Classification with MobileNetV2
Transfer Learning + Real-Time Webcam Inference (Google Colab)

# 🚀 Project Overview
This project implements a real-time multi-class fruit classification system using transfer learning with MobileNetV2 and live webcam inference.

The pipeline moves from:
Data → Augmentation → Transfer Learning → Evaluation → Real-Time Deployment
It demonstrates practical ML engineering beyond notebook experimentation.

# 📂 Dataset

Source: Kaggle – Fruit Classification (10 Classes)
https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class

10 Classes

# 📊 Model Performance

Test Set Results (460 images, 10 classes):

Accuracy: 76%

Macro F1-Score: 0.76

Weighted F1-Score: 0.76

Perfectly Distributed Class Distribution: 46 images per class

Per-Class Highlights

🍍 Pineapple: F1 = 0.87

🍓 Strawberries: F1 = 0.86

🍒 Cherry: F1 = 0.83

🍌 Banana: Recall = 0.87

🥭 Mango: Hardest class (F1 = 0.58)

The balanced macro average confirms consistent performance across categories rather than bias toward dominant classes.

# 🧠 Model Architecture

## Pre Trained Model: 
MobileNetV2 (ImageNet pretrained)
Trainig set to frozen

## Architecture Design

Input: 128×128 grayscale

Conv2D layer converts 1-channel → 3-channel

MobileNetV2 (feature extraction)

GlobalAveragePooling2D

Dense(128) + Dropout(0.5)

Softmax output (10 classes)

# 📈 Evaluation Components

Training vs Validation Accuracy Line Graphs

Training vs Validation Loss Line Graphs

Full Classification Report

Confusion Matrix 

# 📸 Real-Time Deployment (Colab Webcam)

This project integrates a live webcam pipeline using:

JavaScript (browser-side capture)

eval_js bridge to Colab

OpenCV preprocessing

TensorFlow inference

Inference Pipeline

Capture frame from webcam

Convert to grayscale

Resize to 128×128

Normalize

Add batch + channel dimensions

Predict

Overlay label + confidence

Example:

Prediction: Pineapple (0.91)

This demonstrates:

Practical model deployment logic

Cross-language integration (JS ↔ Python)

Real-time inference handling

Robust frame validation checks

# 🛠 Tech Stack

Python

TensorFlow / Keras

MobileNetV2 (ImageNet pretrained)

OpenCV

NumPy

Scikit-learn

Matplotlib

JavaScript

Google Colab


# 🔬 Notes and Future Imrpovements/Plans

Mango and Orange classes show lower recall → potential feature similarity issues

Using RGB instead of grayscale could improve separability

Fine-tuning top MobileNetV2 layers may increase performance

Could be extended to object detection (YOLO) instead of full-frame classification 

TensorFlow Lite conversion for edge deployment
