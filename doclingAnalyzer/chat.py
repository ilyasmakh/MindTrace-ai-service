from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI()

# -----------------------------
# Qdrant config
# -----------------------------
COLLECTION_NAME = "MindTrace-documents"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    https=True
)


def get_context(query: str, project_id: str, num_results: int = 5) -> list[dict]:
    """Search Qdrant collection filtered by project_id and return top chunks as context."""

    query_vector = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding


    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=num_results,
        with_payload=True,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="project_id",
                    match=MatchValue(value=project_id)
                )
            ]
        )
    )

    contexts = []
    for point in results:
        payload = point.payload
        filename = payload.get("filename")
        page_numbers = payload.get("page_numbers")
        title = payload.get("title")

        source_parts = []
        if filename:
            source_parts.append(filename)
        if page_numbers:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        contexts.append({
            "text": payload.get("text"),
            "source": " - ".join(source_parts) if source_parts else None,
            "title": title
        })

    return contexts


def ask_question(question: str, project_id: str, num_results: int = 5) -> dict:
    contexts = get_context(question, project_id, num_results)

    context_text = "\n\n".join([
        f"{c['text']}\n(Source: {c['source']}, Title: {c['title']})"
        for c in contexts
    ])
    system_prompt = f"""
    You are a precise and collaborative AI assistant specialized in software development projects.

    Answer questions as accurately as possible using the provided project context.

    If the context does not contain enough information to give a confident answer,
    do not say that the information is missing. 
    Instead, respond naturally by asking thoughtful and specific follow-up questions 
    that help clarify the user's intent or gather more details about the topic.

    When answering:
    - If the context provides enough detail → give a clear, concise, and accurate answer.
    - If the context is incomplete → ask a relevant clarifying question related to the user's query (without mentioning the lack of context).
    - Always maintain a professional, conversational, and helpful tone.

    Context:
    {context_text}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )

    assistant_answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": assistant_answer,
        "contexts": contexts
    }



def main():
    # Exemple de test
    project_id = "rapport_OCP_2025"  # 🔁 identifiant du projet indexé dans Qdrant
    question = "C'est quoi l'age d'ilyas"  # 💬 ta question

    print(f"🔍 Question : {question}")
    print(f"📁 Projet : {project_id}\n")

    try:
        result = ask_question(question, project_id, num_results=3)

        print("🧠 Réponse de l'assistant :")
        print(result["answer"])
        print("\n📚 Contextes utilisés :")
        for i, c in enumerate(result["contexts"], start=1):
            print(f"\n--- Contexte {i} ---")
            print(f"Source : {c['source']}")
            print(f"Titre  : {c['title']}")
            print(f"Texte  : {c['text'][:300]}...")  # affiche seulement 300 caractères
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")


if __name__ == "__main__":
    main()
