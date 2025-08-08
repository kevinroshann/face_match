# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
# Assuming your original script is named 'your_face_recognition_script.py'
from index2 import find_similar_celebs, convert_np 

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes to allow requests from the React frontend
CORS(app)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file uploads, processes the image for celebrity matching,
    and returns the results as a JSON response.
    """
    # Check if a file was uploaded in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Generate a unique filename to avoid conflicts
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded file temporarily
        file.save(filepath)

        try:
            # Process the image with your face recognition function
            results = find_similar_celebs(filepath)
            
            # Convert NumPy types to standard Python types for JSON serialization
            clean_results = convert_np(results)
            
            return jsonify(clean_results)
        
        except Exception as e:
            # Handle any errors during processing
            return jsonify({"error": str(e)}), 500
        
        finally:
            # Clean up the temporary file after processing
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    # Run the Flask development server
    # The host='0.0.0.0' makes the server externally visible
    # if you need to access it from another machine on the network.
    app.run(debug=True, host='0.0.0.0', port=5000)