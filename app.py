import os
import numpy as np
from flask import Flask, request, render_template
import keras
from PIL import Image

app = Flask(__name__)

# Load model once when server starts
model = keras.models.load_model("Mobile_Net_Fine_Tuned_v3.keras")

@app.route('/')
def home():
    return render_template("IBK_Website.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        result = "Infection"
    else:
        result = "Healthy"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)