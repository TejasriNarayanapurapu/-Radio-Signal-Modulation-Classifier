📡 Radio Signal Modulation Classifier

A Deep Learning project that classifies wireless signal modulation types directly from raw IQ signal data using a 1D Convolutional Neural Network (CNN) and Flask deployment.

Overview

This system automatically predicts modulation types such as BPSK, QPSK, QAM, FM, etc., from radio signal samples.

Input: .npy IQ signal file (1024 × 2)

Output: Modulation type + confidence score

Model

1D CNN for time-series signal classification

Batch Normalization for stable training

Global Average Pooling to reduce overfitting

Softmax output (24 modulation classes)

Tech Stack

Python

TensorFlow / Keras

NumPy & HDF5

Flask

Signal Processing

Run the Project
python3 app.py


Open:

http://127.0.0.1:5000


Upload a .npy signal file to get prediction.

📂 Input Format
Shape: (1024, 2)
Format: .npy

👩‍💻 Author

Narayanapurapu Tejasri
B.Tech CSE (AI & ML) — 2026
GitHub: https://github.com/TejasriNarayanapurapu
