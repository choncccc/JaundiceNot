from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("Jaundice_Classifier.keras")  

def preprocess_image(image):
    image = image.resize((200, 200)) 
    image = np.array(image) / 255.0  
    image = image.flatten() 
    image = np.expand_dims(image, axis=0) 
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    image = preprocess_image(image)

    prediction = model.predict(image)[0][0]
    predicted_label = 1 if prediction >= 0.30 else 0  

    print(f"Predicted Label: {prediction}")  # Print the predicted label

    return jsonify({'prediction': str(predicted_label)})

if __name__ == '__main__':
    app.run(host='192.168.100.7', port=5000, debug=True)
