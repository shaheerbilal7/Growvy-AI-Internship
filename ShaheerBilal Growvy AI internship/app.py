from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('optimized_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    
    return jsonify({'class_id': int(class_id)})

if __name__ == '__main__':
    app.run(debug=True) # Runs a local server for testing

@app.route('/')
def home():
    return "The AI Model API is running! Use the /predict endpoint to send images."