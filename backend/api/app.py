from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__, 
    template_folder='../../frontend/templates',
    static_folder='../../frontend/static'
)
CORS(app)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'digit_model.h5')
model = tf.keras.models.load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:        # Get the image data from the request
        data = request.json.get('image')
        
        if not data:
            return jsonify({'error': 'No image data provided'}), 400

        # Convert to numpy array and ensure proper shape
        image = np.array(data, dtype=np.float32).reshape(1, 28, 28, 1)
        
        # Ensure values are in [0, 1] range
        image = np.clip(image, 0, 1)
        
        # Make prediction
        prediction = model.predict(image)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100
        
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()  
# setting for vercel below line should be use in place on empty bracket when running offline on local machine     
# host='0.0.0.0', port=5000, debug=True
