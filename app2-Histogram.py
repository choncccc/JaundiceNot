from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("Jaundice_ClassifierV1.keras")  

def extract_color_features(image, bins=10):
    image = np.array(image.convert("RGB"))  # Ensure RGB format
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    hue, sat, val = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    
    hue_hist = np.histogram(hue / 180.0, bins=bins, range=(0, 1), density=True)[0]
    sat_hist = np.histogram(sat / 255.0, bins=bins, range=(0, 1), density=True)[0]
    val_hist = np.histogram(val / 255.0, bins=bins, range=(0, 1), density=True)[0]
    
    feature_vector = np.concatenate([hue_hist, sat_hist, val_hist])  
    return feature_vector

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    features = extract_color_features(image, bins=10).reshape(1, -1)

    if features.shape[1] != 30:
        return jsonify({'error': f'Invalid input shape. Expected 30, got {features.shape[1]}'}), 400

    prediction = model.predict(features)[0][0]
    predicted_label = 1 if prediction >= 0.50 else 0  

    print(f"Predicted Label: {predicted_label}, value: {prediction}")  

    return jsonify({'prediction': int(predicted_label)})

if __name__ == '__main__':
    app.run(host='192.168.100.7', port=5000, debug=True)

