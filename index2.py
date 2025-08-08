import pickle
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
import cv2
import json

EMB_FILE = "celebs_embeddings.pkl"
MODEL_NAME = "Facenet512"
TOP_K = 5
CELEB_FOLDER = "images"  # your celeb images folder


def load_embeddings():
    with open(EMB_FILE, "rb") as f:
        data = pickle.load(f)
    return data["names"], np.array(data["embeddings"], dtype=np.float32)


def compute_embedding(img_path):
    emb = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        enforce_detection=False
    )[0]["embedding"]
    return np.array(emb, dtype=np.float32)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_attributes(img_path):
    analysis = DeepFace.analyze(
        img_path=img_path,
        actions=['age', 'gender', 'emotion', 'race'],
        enforce_detection=False
    )
    if isinstance(analysis, list):
        analysis = analysis[0]
    return {
        "age": analysis.get("age"),
        "gender": analysis.get("gender"),
        "dominant_emotion": analysis.get("dominant_emotion"),
        "dominant_race": analysis.get("dominant_race")
    }


def get_landmarks(img_path):
    detector = MTCNN()
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    if faces:
        return faces[0]["keypoints"]  # first face only
    return {}


def convert_np(obj):
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


def find_similar_celebs(user_img):
    names, celeb_embs = load_embeddings()
    user_emb = compute_embedding(user_img)

    sims = [cosine_similarity(user_emb, e) for e in celeb_embs]
    ranked_idx = np.argsort(sims)[::-1][:TOP_K]

    results = []
    for idx in ranked_idx:
        results.append({
            "name": names[idx],
            "similarity_percent": round(sims[idx] * 100, 2)
        })

    top_match_name = results[0]["name"]

    # Use your correct folder here for celeb images
    celeb_img_path = f"{CELEB_FOLDER}/{top_match_name}.jpg"

    try:
        celeb_attr = get_attributes(celeb_img_path)
    except Exception as e:
        print(f"Failed to analyze celeb attributes: {e}")
        celeb_attr = {}

    try:
        user_attr = get_attributes(user_img)
    except Exception as e:
        print(f"Failed to analyze user attributes: {e}")
        user_attr = {}

    try:
        user_landmarks = get_landmarks(user_img)
    except Exception as e:
        print(f"Failed to get user landmarks: {e}")
        user_landmarks = {}

    try:
        celeb_landmarks = get_landmarks(celeb_img_path)
    except Exception as e:
        print(f"Failed to get celeb landmarks: {e}")
        celeb_landmarks = {}

    output = {
        "top_match": results[0],
        "top_k_matches": results,
        "user_attributes": user_attr,
        "celeb_attributes": celeb_attr,
        "user_landmarks": user_landmarks,
        "celeb_landmarks": celeb_landmarks
    }

    return output


if __name__ == "__main__":
    query_img = "user_photo.jpg"  # your input image here
    output = find_similar_celebs(query_img)

    clean_output = convert_np(output)
    print(json.dumps(clean_output, indent=2))
