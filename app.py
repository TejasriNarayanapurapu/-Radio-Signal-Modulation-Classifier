# ----------------------------------------
# Radio Signal Modulation Classifier (Flask)
# ----------------------------------------

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# ----------------------------------------
# Load Trained Model
# ----------------------------------------
print("Loading model...")

model = tf.keras.models.load_model("radioml.keras")

print("✅ Model loaded successfully!")

# ----------------------------------------
# RadioML Classes
# (keep same order as training labels)
# ----------------------------------------
CLASSES = [
    "BPSK","QPSK","8PSK","QAM16","QAM64",
    "AM-DSB","AM-SSB","WBFM","CPFSK","GFSK",
    "PAM4","QAM256","OOK","APSK16","APSK32",
    "FM","GMSK","DSB-SC","SSB-SC","OFDM",
    "FSK","ASK","PSK","Unknown"
]

# ----------------------------------------
# Home Page
# ----------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------------------
# Prediction Route
# ----------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return render_template("index.html",
                               prediction="No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html",
                               prediction="No file selected")

    # Save uploaded file
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    try:
        # -----------------------------
        # Load .npy signal
        # -----------------------------
        signal = np.load(filepath)

        # -----------------------------
        # Validate input shape
        # Expected shape = (1024, 2)
        # -----------------------------
        if signal.shape != (1024, 2):
            return render_template(
                "index.html",
                prediction=f"Invalid input shape {signal.shape}. Expected (1024,2)"
            )

        # -----------------------------
        # SAME normalization as training
        # (VERY IMPORTANT)
        # -----------------------------
        signal = signal / np.max(np.abs(signal))

        # Add batch dimension
        signal = np.expand_dims(signal, axis=0)

        # -----------------------------
        # Prediction
        # -----------------------------
        prediction = model.predict(signal, verbose=0)

        predicted_index = int(np.argmax(prediction))
        predicted_class = CLASSES[predicted_index]
        confidence = float(np.max(prediction) * 100)

        print("Prediction vector:", prediction)
        print("Predicted:", predicted_class)

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=round(confidence, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


# ----------------------------------------
# Run Flask App
# ----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
