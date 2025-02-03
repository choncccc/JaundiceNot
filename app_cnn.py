from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load the trained CNN model
model = tf.keras.models.load_model("Jaundice_Classifier_CNN.h5")

def preprocess_image(image, target_size=(200, 200)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0][0]
    predicted_label = 1 if prediction >= 0.50 else 0

    print(f"Predicted Label: {predicted_label}, value: {prediction}")

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(host='192.168.100.7', port=5000, debug=True)