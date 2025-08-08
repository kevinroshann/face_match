import pickle
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
import cv2
import json
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS # Required for cross-origin requests from React

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Folder to temporarily save uploaded images
EMB_FILE = "celebs_embeddings.pkl"
MODEL_NAME = "Facenet512"
TOP_K = 5
CELEB_FOLDER = "images"  # your celeb images folder (e.g., "images/Jennifer_Aniston.jpg")

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Your original functions ---
def load_embeddings():
    """Loads pre-computed celebrity embeddings."""
    try:
        with open(EMB_FILE, "rb") as f:
            data = pickle.load(f)
        return data["names"], np.array(data["embeddings"], dtype=np.float32)
    except FileNotFoundError:
        print(f"Error: {EMB_FILE} not found. Please ensure it exists.")
        return [], np.array([], dtype=np.float32)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return [], np.array([], dtype=np.float32)

def compute_embedding(img_path):
    """Computes the facial embedding for a given image."""
    try:
        emb = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            enforce_detection=False
        )[0]["embedding"]
        return np.array(emb, dtype=np.float32)
    except Exception as e:
        print(f"Error computing embedding for {img_path}: {e}")
        raise

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0: # Handle zero vectors to avoid division by zero
        return 0.0
    return dot_product / (norm_a * norm_b)

def get_attributes(img_path):
    """Analyzes facial attributes (age, gender, emotion, race) from an image."""
    try:
        analysis = DeepFace.analyze(
            img_path=img_path,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False
        )
        if isinstance(analysis, list) and analysis:
            analysis = analysis[0] # Take the first detected face's analysis
        elif not analysis:
            return {} # No face detected or analysis failed

        # Map 'Woman'/'Man' to 'Female'/'Male' for consistency if desired
        gender = analysis.get("gender")
        if isinstance(gender, dict): # DeepFace 0.0.12 returns dict like {'Woman': 99.99, 'Man': 0.01}
            gender = "Woman" if gender.get("Woman", 0) > gender.get("Man", 0) else "Man"
        else: # Older versions might return 'Woman' or 'Man' directly
            gender = gender

        return {
            "age": analysis.get("age"),
            "gender": gender,
            "dominant_emotion": analysis.get("dominant_emotion"),
            "dominant_race": analysis.get("dominant_race")
        }
    except Exception as e:
        print(f"Error getting attributes for {img_path}: {e}")
        return {}

def get_landmarks(img_path):
    """Detects facial landmarks using MTCNN."""
    detector = MTCNN()
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image at {img_path}")
            return {}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)
        if faces:
            # MTCNN returns a list of detected faces. We take the keypoints of the first face.
            return faces[0]["keypoints"]
    except Exception as e:
        print(f"Error getting landmarks for {img_path}: {e}")
    return {}

def convert_np(obj):
    """Converts numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def find_similar_celebs(user_img_path):
    """Finds celebrity matches for a user image and gathers detailed analysis."""
    names, celeb_embs = load_embeddings()
    if not names: # Handle case where embeddings could not be loaded
        return {
            "error": "Celebrity embeddings not loaded. Please check EMB_FILE path and content."
        }

    try:
        user_emb = compute_embedding(user_img_path)
    except Exception as e:
        return {"error": f"Failed to compute user image embedding: {e}"}


    sims = [cosine_similarity(user_emb, e) for e in celeb_embs]
    ranked_idx = np.argsort(sims)[::-1][:TOP_K]

    results = []
    for idx in ranked_idx:
        results.append({
            "name": names[idx],
            "similarity_percent": round(sims[idx] * 100, 2)
        })

    top_match_name = results[0]["name"] if results else "Unknown"

    # DeepFace models require image paths, so we need to construct them
    # Ensure celeb image names match your file system (e.g., "Jennifer_Aniston.jpg")
    celeb_img_path = os.path.join(CELEB_FOLDER, f"{top_match_name}.jpg") # Assuming .jpg extension

    user_attr = get_attributes(user_img_path)
    celeb_attr = get_attributes(celeb_img_path)
    user_landmarks = get_landmarks(user_img_path)
    celeb_landmarks = get_landmarks(celeb_img_path)

    output = {
        "top_match": results[0] if results else None,
        "top_k_matches": results,
        "user_attributes": user_attr,
        "celeb_attributes": celeb_attr,
        "user_landmarks": user_landmarks,
        "celeb_landmarks": celeb_landmarks
    }

    return output

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

@app.route('/api/find_celebs', methods=['POST'])
def find_celebs_api():
    """API endpoint to receive an image and return celebrity match results."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath) # Save the uploaded file temporarily

        try:
            results = find_similar_celebs(filepath)
            # Ensure all numpy types are converted for JSON serialization
            clean_results = convert_np(results)
            return jsonify(clean_results)
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the uploaded file after processing
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    # You can specify a different port if needed, e.g., port=5001
    app.run(debug=True, port=5000)
