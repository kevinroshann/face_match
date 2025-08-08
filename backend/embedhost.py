


import os
import pickle
from pinecone import Pinecone, ServerlessSpec

# Load your Pinecone API key from environment or set directly here
#api_key = os.getenv("pcsk_5DuaBJ_KJjaNxsEc1zByyyuzd54K7SwVo9DupKNUhWjMSfxvvKww2acqM1Ew6Y54F2gyyH") or "YOUR_API_KEY"

api_key = "pcsk_5DuaBJ_KJjaNxsEc1zByyyuzd54K7SwVo9DupKNUhWjMSfxvvKww2acqM1Ew6Y54F2gyyH"
#pc = Pinecone(api_key=api_key)


# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

index_name = "my-embedding-index"
embedding_dim = 512  # change to your embedding dimension

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,
        metric='cosine',  # or 'euclidean', 'dotproduct'
        spec=ServerlessSpec(
            cloud='aws',    # adjust based on your Pinecone environment
            region='us-west-2'
        )
    )

# Connect to the index
index = pc.index(index_name)

# Load your embeddings from file
with open("backend/celebs_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)  # e.g. {'id1': [0.1, 0.2, ...], 'id2': [0.3, 0.4, ...]}

# Upsert embeddings in batches (recommended for large datasets)
batch_size = 100
items = list(embeddings.items())

for i in range(0, len(items), batch_size):
    batch = items[i:i+batch_size]
    vectors = [(id_, vector) for id_, vector in batch]  # metadata optional, so omitted here
    index.upsert(vectors)

print("All embeddings upserted!")
