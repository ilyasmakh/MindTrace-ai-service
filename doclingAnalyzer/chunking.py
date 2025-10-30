from docling.chunking import HybridChunker
from doclingAnalyzer.extraction import extract_document
from transformers import AutoTokenizer


MAX_TOKENS = 500
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def extract_and_chunk(path_url: str) -> dict:
    """
    Extracts a PDF and splits it into chunks.

    Args:
    path_url (str): Local path or URL of the PDF file

    Returns:
    dict: {
    "markdown": Markdown content,
    "json": Dictionary representation,
    "chunks": List of chunks
    }
    """
    pdf_data = extract_document(path_url)
    document = pdf_data["document"]

    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=False,
    )
    chunks = list(chunker.chunk(dl_doc=document))

    return {
        "markdown": pdf_data["markdown"],
        "json": pdf_data["json"],
        "chunks": chunks
    }



if __name__ == "__main__":
    # Chemin vers ton fichier PDF local (à modifier si besoin)
    test_pdf_path = "C:/Users/dell/Downloads/ilyasM.pdf"

    import os

    if os.path.exists(test_pdf_path):
        # Appel de la fonction
        result = extract_and_chunk(test_pdf_path)

        # Affichage du contenu Markdown (500 premiers caractères)
        print("=== Markdown ===")
        print(result["markdown"][:500])

        # Affichage des clés principales du JSON
        print("\n=== JSON keys ===")
        print(result["json"].keys())

        # Vérification du type et du nombre de chunks
        print("\n=== Chunks info ===")
        print(f"Nombre de chunks : {len(result['chunks'])}")
