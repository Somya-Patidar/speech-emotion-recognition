import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from utils.feature_utils import extract_features # Ensure this function is ready

app = Flask(__name__)
MODEL_PATH = 'models/gru_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = "static/temp_audio.wav"
    file.save(file_path)

    # 1. Extract features (adjust based on your GRU input shape)
    features = extract_features(file_path) 
    features = np.expand_dims(features, axis=0) # Add batch dimension

    # 2. Predict
    prediction = model.predict(features)
    emotion_idx = np.argmax(prediction)
    result = EMOTIONS[emotion_idx]

    return jsonify({'emotion': result})

if __name__ == '__main__':
    app.run(debug=True)