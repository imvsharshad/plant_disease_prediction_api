import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration ---
# Allow requests from your Next.js frontend (update the origin if your frontend URL is different)
CORS(app, resources={r"/api/*": {"origins": "*"}}) 

# Define the path to the model. Assumes the model is in the same directory as this script.
MODEL_PATH = "plant_disease_model.h5"
CLASS_NAMES = ['Disease', 'Healthy']  # Must be in alphabetical order
IMAGE_SIZE = (224, 224)

# --- Model Loading ---
try:
    print("Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

def preprocess_and_predict(image_bytes):
    """
    Loads image from bytes, preprocesses it, and returns the prediction.
    """
    if model is None:
        raise RuntimeError("Model is not loaded. Cannot make predictions.")

    # Load the image from bytes using PIL and convert to RGB
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    
    # Convert the image to a NumPy array
    img_array = tf.keras.utils.img_to_array(img)
    
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image for the ResNetV2 model
    img_preprocessed = tf.keras.applications.resnet_v2.preprocess_input(img_array)

    # Make a prediction
    prediction = model.predict(img_preprocessed)
    
    # The output of a sigmoid is a single value between 0 and 1
    score = float(prediction[0][0]) # Ensure it's a standard Python float
    
    # Interpret the result
    if score < 0.5:
        predicted_class = CLASS_NAMES[0] # Disease
        confidence = 1 - score
    else:
        predicted_class = CLASS_NAMES[1] # Healthy
        confidence = score
        
    return predicted_class, confidence

# --- API Endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint that receives an image file and returns the prediction.
    """
    # Check if a file was sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    # Check if the file is an allowed type (optional but good practice)
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid image type'}), 400

    try:
        # Read image bytes from the file
        image_bytes = file.read()
        
        # Get the prediction
        predicted_class, confidence = preprocess_and_predict(image_bytes)
        
        # Return the result as JSON
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2%}"
        })

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# Health check route
@app.route('/')
def index():
    return "Plant Disease Prediction API is running!"

# This is used for local development
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your local network
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
