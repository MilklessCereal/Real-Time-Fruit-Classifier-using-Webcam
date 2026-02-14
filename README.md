# 🍎 Real-Time Fruit Classification with MobileNetV2
Transfer Learning + Real-Time Webcam Inference (Google Colab)

# 🚀 Project Overview
This project implements a real-time multi-class fruit classification system using transfer learning with MobileNetV2 and live webcam inference.

The pipeline moves from:
Data → Augmentation → Transfer Learning → Evaluation → Real-Time Deployment

# 📂 Dataset

Source: Fruit Classification (10 Classes) 
https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class

10 Classes

Perfectly Distributed Class Distribution: 

184 images per class (Training)

46 images per class (Validation)

# 📊 Model Performance

Test Set Results (460 images, 10 classes):

Decently Overfit

Accuracy: 90%

Macro F1-Score: 0.90

Weighted F1-Score: 0.90

# 🧠 Model Architecture

## Pre Trained Model: 
MobileNetV2 (ImageNet pretrained)

The MobileNetV2 backbone was initialized with ImageNet weights and used as a fixed feature extractor. During training, only the custom classification layers was optimized while backbone parameters remained unchanged.

## Architecture Design

Input: 128×128 RGB

MobileNetV2 (feature extraction)

GlobalAveragePooling2D

Dense(128) + Dropout(0.7)

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

Resize to 128×128 + Normalize + Add batch + channel dimensions

Predict -> Overlay label + confidence

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

Could be extended to object detection (YOLO) instead of full-frame classification 

TensorFlow Lite conversion for edge deployment
