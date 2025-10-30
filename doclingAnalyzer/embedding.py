from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter , FieldCondition,MatchValue
from transformers import AutoTokenizer
from doclingAnalyzer.chunking import extract_and_chunk
from dotenv import load_dotenv
import os
from openai import OpenAI
import uuid  # <-- pour générer des UUID

load_dotenv()

MAX_TOKENS = 500
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# OpenAI client
openai_client = OpenAI()

# Qdrant config
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "MindTrace-documents"
VECTOR_SIZE = 1536

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120,
    https=True
)

def get_embedding(text: str) -> List[float]:
    """Create embedding via OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def process_document_to_qdrant(path_or_url: str, project_id: str):
    data = extract_and_chunk(path_or_url)
    chunks = data["chunks"]
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"Collection '{COLLECTION_NAME}' created in Qdrant.")
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="project_id",
            field_schema="keyword",
        )
        print("Index on 'project_id' created.")

    # 3️⃣ Préparer les points
    points: List[PointStruct] = []
    for chunk in chunks:
        text = chunk.text
        vector = get_embedding(text)
        payload = {
            "project_id": project_id,
            "text": text,
            "filename": getattr(chunk.meta.origin, "filename", None),
            "page_numbers": [
                page_no
                for item in getattr(chunk.meta, "doc_items", [])
                for prov in getattr(item, "prov", [])
                for page_no in [prov.page_no]
            ] or None,
            "title": chunk.meta.headings[0] if getattr(chunk.meta, "headings", []) else None,
        }
        # ID valide: UUID
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

    # 4️⃣ Upsert dans Qdrant
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"{len(points)} chunks inserted for document {project_id}.")
    return points


def create_project_id_index(collection_name: str):
    """
    Creates an index on the 'project_id' field in the given collection.
    :param collection_name: Name of the Qdrant collection
    """
    qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="project_id",
        field_schema="keyword",
    )

def delete_by_project_id(collection_name: str, project_id: str):
    """
    Deletes all vectors of a specific document (via its project_id)
    in a Qdrant collection.
    """
    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="project_id",
                    match=MatchValue(value=project_id)
                )
            ]
        )
    )
    print(f"✅ All vectors with project_id='{project_id}' have been removed from the collection '{collection_name}'.")
    return True




def main1():
    # Exemple : chemin vers ton document local ou lien URL
    path_or_url = "C:/Users/dell/Downloads/ilyasM.pdf"  # 🔁 change selon ton cas
    project_id = "rapport_OCP_2025"  # identifiant unique du document

    try:
        print(f"📄 Traitement du document : {path_or_url}")
        points = process_document_to_qdrant(path_or_url, project_id)
        print(f"✅ Insertion réussie : {len(points)} chunks insérés dans Qdrant.")
    except Exception as e:
        print(f"❌ Erreur lors du traitement : {e}")

def main2():
    delete_by_project_id("MindTrace-documents", "rapport_OCP_2025")
if __name__ == "__main__":
    main1()


