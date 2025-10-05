import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# Import TensorFlow, which includes the TFLite interpreter
import tensorflow as tf

# Initialize the Flask application
app = Flask(__name__)
# Configure Cross-Origin Resource Sharing (CORS) to allow your frontend to call the API
CORS(app, resources={r"/api/*": {"origins": "*"}}) 

# --- Configuration ---
TFLITE_MODEL_PATH = "plant_disease_model.tflite"
CLASS_NAMES = ['Disease', 'Healthy'] # Ensure this is in alphabetical order
IMAGE_SIZE = (224, 224)

# --- TFLite Model Loading ---
# Load the TFLite model once when the application starts.
interpreter = None
try:
    print("Loading TFLite model...")
    # Initialize the TFLite interpreter from the main TensorFlow package
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    # Allocate memory for the model's tensors
    interpreter.allocate_tensors()

    # Get details about the model's input and output layers
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite Model loaded successfully!")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    # interpreter remains None if loading fails

def preprocess_and_predict(image_bytes):
    """
    Preprocesses an image from bytes and runs inference using the loaded TFLite model.
    """
    if interpreter is None:
        raise RuntimeError("Model is not loaded. Cannot make predictions.")

    # 1. Load image from bytes using Pillow, ensuring it's in RGB format
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # 2. Resize the image to the size the model expects
    img = img.resize(IMAGE_SIZE)
    # 3. Convert the image to a NumPy array with a float32 data type
    img_array = np.array(img, dtype=np.float32)
    # 4. Add a batch dimension to match the model's input shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Preprocess the image: Scale pixel values to the [0, 1] range.
    img_array = img_array / 255.0

    # --- Run Inference ---
    # 6. Set the input tensor of the model with the preprocessed image data
    interpreter.set_tensor(input_details[0]['index'], img_array)
    # 7. Execute the prediction
    interpreter.invoke()
    # 8. Get the prediction result from the output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])
    score = float(prediction[0][0])
    
    # 9. Interpret the raw score
    if score < 0.5:
        predicted_class = CLASS_NAMES[0] # Corresponds to 'Disease'
        confidence = 1 - score
    else:
        predicted_class = CLASS_NAMES[1] # Corresponds to 'Healthy'
        confidence = score
        
    return predicted_class, confidence

# --- API Endpoints ---
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        image_bytes = file.read()
        predicted_class, confidence = preprocess_and_predict(image_bytes)
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2%}"
        })
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

@app.route('/')
def index():
    return "Plant Disease Prediction API (TFLite) is running!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

