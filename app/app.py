from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import io

app = Flask(__name__)

pipe = pipeline("image-classification", model="itsTomLie/Jaundice_Classifier")

def compute_way_kmeans_lab(image, k=3):
    image = np.array(image.convert("RGB"))
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    pixels = lab_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    normalized_B_values = ((cluster_centers[:, 2] - 128) / 127.0) * 50 + 20
    
    cluster_sizes = np.bincount(cluster_labels, minlength=k)
    total_pixels = np.sum(cluster_sizes)
    cluster_proportions = cluster_sizes / total_pixels
    
    WAY = np.dot(cluster_proportions, normalized_B_values)
    return WAY


def classify_jaundice(WAY):
    if 20 <= WAY <= 25:
        return "Onset/Mild Jaundice"
    elif 25 < WAY < 35:
        return "Moderate Jaundice"
    elif WAY > 35:
        return "Severe Jaundice"
    else:
        return "No Jaundice"

# endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    test_image_dir = "C:/Users/kenji/OneDrive/Desktop/JaundiceNot/app/test_image/image_24.jpg"
    test_image = Image.open(test_image_dir)

    #result = pipe(test_image)
    result = pipe(image)
    predicted_label = result[0]['label']
    
    response_data = {'prediction': predicted_label}

    if predicted_label == "Jaundiced Eyes": 
        WAY = compute_way_kmeans_lab(image)
        #WAY = compute_way_kmeans_lab(test_image)
        severity = classify_jaundice(WAY)
        response_data["WAY"] = WAY
        response_data["severity"] = severity
        print(f"WAY: {WAY}, Severity: {severity}")

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='192.168.100.7', port=5000, debug=True)
