from docling.document_converter import DocumentConverter
import os

# Disable HF Hub symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Initialize the converter
converter = DocumentConverter()


def extract_document(url_or_path: str) -> dict:
    """
    Extract content from a PDF file or URL.

    Args:
        url_or_path: URL or local path to the PDF

    Returns:
        dict: {
            "markdown": markdown content,
            "json": dictionary representation,
            "document": objet Document pour chunking
        }
    """
    result = converter.convert(url_or_path)
    document = result.document
    return {
        "markdown": document.export_to_markdown(),
        "json": document.export_to_dict(),
        "document": document
    }



def extract_sitemap(sitemap_urls: list) -> list:
    """
    Extract content from multiple URLs (sitemap).

    Args:
        sitemap_urls: list of URLs

    Returns:
        list: list of documents as dicts {
            "markdown": markdown content,
            "json": dictionary representation
        }
    """
    conv_results_iter = converter.convert_all(sitemap_urls)
    docs = []
    for result in conv_results_iter:
        if result.document:
            document = result.document
            docs.append({
                "markdown": document.export_to_markdown(),
                "json": document.export_to_dict()
            })
    return docs



if __name__ == "__main__":
    # Exemple de test avec un fichier PDF local
    test_pdf_path = "C:/Users/dell/Downloads/recalcul du trajet.pdf"  # Remplace par le chemin vers ton PDF

    if os.path.exists(test_pdf_path):
        result = extract_document(test_pdf_path)
        print("=== Markdown ===")
        print(result["markdown"][:500])  # Affiche juste les 500 premiers caractères
        print("\n=== JSON keys ===")
        print(result["json"].keys())  # Affiche les clés principales du dictionnaire
        print("\n=== Document type ===")
        print(type(result["document"]))  # Vérifie que c'est un objet Document
    else:
        print(f"Le fichier {test_pdf_path} n'existe pas, merci de le créer ou changer le chemin.")
