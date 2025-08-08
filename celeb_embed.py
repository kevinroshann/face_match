# celeb_embed.py
import os
import numpy as np
import pickle
from deepface import DeepFace
from tqdm import tqdm

DATASET_DIR = "/home/kevin/Desktop/facematch/images"
EMB_FILE = "celebs_embeddings.pkl"
MODEL_NAME = "Facenet512"  # high accuracy

def create_embeddings():
    embeddings = []
    names = []

    for file in tqdm(os.listdir(DATASET_DIR), desc="Processing Celebs"):
        path = os.path.join(DATASET_DIR, file)
        if not os.path.isfile(path):
            continue

        try:
            emb = DeepFace.represent(
                img_path=path,
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]  # Get embedding vector

            embeddings.append(emb)
            names.append(os.path.splitext(file)[0])  # remove .jpg/.png
        except Exception as e:
            print(f"Failed for {file}: {e}")

    embeddings = np.array(embeddings, dtype=np.float32)
    with open(EMB_FILE, "wb") as f:
        pickle.dump({"names": names, "embeddings": embeddings}, f)

    print(f"Saved {len(names)} celebrity embeddings to {EMB_FILE}")

if __name__ == "__main__":
    create_embeddings()
