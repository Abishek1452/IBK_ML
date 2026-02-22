import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image


# Initialize Flask app
app = Flask(__name__)

# Load model once when server starts
MODEL_PATH = "Mobile_Net_Fine_Tuned_v3.keras"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Home Route
@app.route("/")
def home():
    return render_template("IBK_Website.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        # Process image
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))  
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]

        infected = bool(prediction > 0.5)
        confidence = float(
            prediction * 100 if infected else (1 - prediction) * 100
        )

        return jsonify({
            "infected": infected,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# IMPORTANT for Render/Gunicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)