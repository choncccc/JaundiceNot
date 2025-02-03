from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("Jaundice_Classifier_kmeans1.keras")

def extract_kmeans_features(image, num_clusters=6):
    """Extracts dominant Hue values from the sclera using K-Means clustering."""
    image = np.array(image.convert("RGB"))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    hue_values = hsv_image[:, :, 0].flatten()
    hue_values = hue_values[hue_values > 0]

    if len(hue_values) > 0:
        hue_values = hue_values.reshape(-1, 1)

        kmeans = KMeans(n_clusters=min(num_clusters, len(hue_values)), random_state=0, n_init=10)
        kmeans.fit(hue_values)

        dominant_hues = kmeans.cluster_centers_.flatten() / 180.0 
        return np.pad(dominant_hues, (0, num_clusters - len(dominant_hues)), 'constant')
    else:
        return np.zeros(num_clusters)  # Handle empty cases


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    # Extract features using K-Means
    features = extract_kmeans_features(image, num_clusters=6).reshape(1, -1)
    if features.shape[1] != 6:
        return jsonify({'error': f'Invalid input shape. Expected 6, got {features.shape[1]}'}), 400

    # Model prediction
    prediction = model.predict(features)[0][0]
    predicted_label = 1 if prediction >= 0.55 else 0  # Adjust threshold if needed

    # Bias handling: Identify potential bias
    bias_flag = False
    if predicted_label == 1 and prediction >= 0.8: 
        bias_flag = True

    # Print logs for debugging
    print(f"Predicted Label: {predicted_label}, Value: {prediction}, Bias Detected: {bias_flag}")

    response = {
        'prediction': int(predicted_label),
        'confidence': float(prediction),
        'bias_warning': bias_flag
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='192.168.100.7', port=5000, debug=True)
