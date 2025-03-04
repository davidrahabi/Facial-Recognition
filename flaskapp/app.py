from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from layers import L1Dist
import numpy as np
import cv2

import logging
from functools import wraps

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Decorators
def require_consent(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.headers.get('user-consent'):
            return jsonify({'error': 'User consent required'}), 403
        return f(*args, **kwargs)
    return decorated_function

def verify_age(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.headers.get('age-verified'):
            return jsonify({'error': 'Age verification required'}), 403
        return f(*args, **kwargs)
    return decorated_function


# Load the model globally
from model import make_embedding, make_siamese_model

# Load and compile model
embedding = make_embedding()
siamese_model = make_siamese_model()
siamese_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.losses.BinaryCrossentropy()
)
siamese_model.load_weights('../siamese_model_final.h5')

def preprocess(image_data):
    # Convert to tensor and preprocess
    img = tf.image.decode_jpeg(image_data)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    # Get both images from the request
    image1 = request.files['image1']
    image2 = request.files['image2']
    
    # Read and preprocess both images
    img1_data = image1.read()
    img2_data = image2.read()
    
    input_img = preprocess(img1_data)
    validation_img = preprocess(img2_data)
    
    # Make prediction
    result = siamese_model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
    
    # Use detection threshold
    verified = bool(result > 0.60)
    
    return jsonify({
        'verified': verified,
        'similarity_score': float(result[0][0])
    })

if __name__ == '__main__':
    app.run(debug=True)