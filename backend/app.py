# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from index2 import find_similar_celebs, convert_np 

# Get the absolute path to the frontend build directory
# This assumes your directory structure is:
# project_root/
# ├── backend/
# │   └── app.py
# └── frontend/
#     └── build/
react_build_dir = os.path.join(os.path.dirname(__file__), '../frontend/build')

# Initialize Flask app
app = Flask(__name__, static_folder=react_build_dir, static_url_path='')

# Enable CORS for the API routes
CORS(app, resources={r"/upload/*": {"origins": "*"}})

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Route for serving the React application
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """
    Serve the React frontend's static files.
    """
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file uploads, processes the image for celebrity matching,
    and returns the results as a JSON response.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)

        try:
            results = find_similar_celebs(filepath)
            clean_results = convert_np(results)
            return jsonify(clean_results)
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)