from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# --- Config ---
COLLECTION_NAME = "MindTrace-documents"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    https=True
)

openai_client = OpenAI()

def get_query_embedding(query: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

def search_qdrant(query: str, project_id: str, limit: int = 3):
    vector = get_query_embedding(query)

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=Filter(
            must=[FieldCondition(key="project_id", match=MatchValue(value=project_id))]
        ),
        limit=limit,
        with_payload=True
    )


    df = pd.DataFrame([{
        "project_id": r.payload.get("project_id"),
        "text": r.payload.get("text"),
        "distance": r.score
    } for r in results])

    return df


def main():
    # Exemple de paramètres
    query = "c'est quoi l'age d'ilyas"
    project_id = "rapport_OCP_2025"
    limit = 3

    try:
        print(f"🔍 Recherche de similarités pour : '{query}' dans le projet '{project_id}'...")
        df = search_qdrant(query, project_id, limit)
        print("✅ Résultats trouvés :")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"❌ Erreur pendant la recherche : {e}")

if __name__ == "__main__":
    main()