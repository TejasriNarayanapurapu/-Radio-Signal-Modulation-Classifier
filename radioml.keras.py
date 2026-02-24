# -------------------------------
# RadioML Modulation Classification - CNN
# Works on 50k samples safely
# -------------------------------

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Dataset (ONLY 50k)
# -------------------------------

file_path = "/Users/narayanapuraputejasri/PycharmProjects/PythonProject/GOLD_XYZ_OSC.0001_1024.hdf5"

N = 200000   # IMPORTANT: limit samples

print("Loading dataset...")

with h5py.File(file_path, "r") as f:
    X = f["X"][:N]
    Y = f["Y"][:N]

print("Dataset loaded")
print("X shape:", X.shape)

# -------------------------------
# 2. Convert labels
# -------------------------------

labels = np.argmax(Y, axis=1)
indices = np.random.permutation(len(X))
X = X[indices]
labels = labels[indices]


# -------------------------------
# 3. Normalize IQ signals
# -------------------------------

X = X / np.max(np.abs(X), axis=(1,2), keepdims=True)

# -------------------------------
# 4. Train/Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -------------------------------
# Improved CNN Model
# -------------------------------

model = Sequential([
    Input(shape=(1024, 2)),

    Conv1D(64, 5, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(256, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    GlobalAveragePooling1D(),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(24, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# Callbacks
# -------------------------------

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=6,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

# -------------------------------
# Train (ONLY 30 epochs)
# -------------------------------

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=256,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_reduce]
)

# -------------------------------
# Evaluate
# -------------------------------

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")


model.save("radioml.keras")

