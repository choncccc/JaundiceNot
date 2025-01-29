from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)

pipe = pipeline("image-classification", model="itsTomLie/Jaundice_Classifier")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    result = pipe(image)
    label = result[0]['label']
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='192.168.100.7', port=5000, debug=True)
