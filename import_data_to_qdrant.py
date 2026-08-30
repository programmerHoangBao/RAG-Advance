import os
import uuid
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)
from sentence_transformers import SentenceTransformer

load_dotenv("./.env")

def create_collection(
    qdrant_client: QdrantClient,
    collection_name: str = "stackoverflow_questions",
    vector_size: int = 768
):

    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]

    # If collection exists -> delete old collection
    if collection_name in collection_names:

        qdrant_client.delete_collection(
            collection_name=collection_name
        )

        print(f"Deleted old collection: {collection_name}")

    # Create new collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    print(f"Created collection: {collection_name}")

    # Create index for tags
    qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="tags",
        field_schema="keyword"
    )

    print("Created payload index for tags")

# Load model
def load_model(model_path: str = "BAAI/bge-base-en-v1.5"):
    model = SentenceTransformer(model_path)
    print(f"Loaded model: {model_path} successfully")
    return model
    
# Load parquet data
def load_parquet(file_path: str):
    df = pd.read_parquet(file_path)
    # Convert tags: np.array -> list
    df["tags"] = df["tags"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
    print(f"Number of row: {len(df)}")
    return df

# Convert embedding
def embedding_texts(model: SentenceTransformer, texts: list, batch_size: int = 64):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings

# Insert data into Qdrant
def insert_data(
    qdrant_client: QdrantClient,
    model: SentenceTransformer, 
    data_path: str,
    collection_name: str = "stackoverflow_questions",
    batch_size=128
):

    df = load_parquet(data_path)
    questions = (df['title'] + '. ' + df['question']).tolist()

    # Embedding
    embeddings = embedding_texts(model, questions)

    points = []

    for idx, row in df.iterrows():
        question = str(row['title']) + '. ' + str(row["question"])
        answer = str(row["answer"])
        tags = row["tags"]
        vector = embeddings[idx].tolist()
        payload = {
            "question": question,
            "answer": answer,
            "tags": tags
        }

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload
        )

        points.append(point)

        # Insert by batch
        if len(points) >= batch_size:

            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )

            print(f"Inserted {len(points)} points")
            points = []

    # Insert remaining points
    if len(points) > 0:

        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

        print(f"Inserted remaining {len(points)} points")

    print("Done insert data")

def main():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("API_KEY")
    collection_name = "stackoverflow_questions"
    model_path = "./bge-base-en-v1-5"
    data_path = "./data/data.parquet"
    vector_size = 768
    
    qdrant_client = QdrantClient(
        url=url,
        api_key=api_key,
    )
    
    # 1. Create collection
    create_collection(
        qdrant_client=qdrant_client, 
        collection_name=collection_name, 
        vector_size=vector_size
    )
    
    # 2. Load model
    model = load_model(model_path=model_path)
    
    # 3. Insert parquet data
    insert_data(
        qdrant_client=qdrant_client, 
        model=model, 
        data_path=data_path, 
        collection_name=collection_name, 
        batch_size=1024
    )

if __name__ == "__main__":
    main()