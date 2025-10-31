from fastapi import FastAPI, Query, Body, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

from TraceSpecAdjustment.traceSpecAdjustment import analyze_requirement_changes
from doclingAnalyzer.chat import ask_question
from doclingAnalyzer.extraction import extract_document
from doclingAnalyzer.chunking import extract_and_chunk
from doclingAnalyzer.embedding import process_document_to_qdrant , delete_by_project_id
from doclingAnalyzer.search import search_qdrant


app = FastAPI(
    title="MindTrace AI Service",
    description="AI-driven service that retrieves, analyzes, and answers questions about project documents using contextual search and embeddings",
    version="1.0.0"
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 🔹 Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "MindTrace AI Service"}



@app.post("/api/ai-analyze/extract-document")
async def extract_pdf_endpoint(file: UploadFile = File(...)):
    try:
        # Sauvegarder temporairement le fichier uploadé
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # Extraction (contient aussi document)
        result = extract_document(temp_file)

        # On supprime document uniquement du retour JSON
        safe_result = {
            "markdown": result["markdown"],
            "json": result["json"]
        }

        return JSONResponse(content=safe_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PDFRequest(BaseModel):
    url_or_path: str
@app.post("/api/ai-analyze/extract-and-chunk")
async def extract_and_chunk_endpoint(request: PDFRequest):
    try:
        # Appel de ta fonction
        result = extract_and_chunk(request.url_or_path)

        # ⚠️ chunks non sérialisables → transformer en dict minimal
        safe_chunks = [
            {
                "id": i,
                "text": chunk.text,   # attribut du chunk
                "tokens": chunk.tokens if hasattr(chunk, "tokens") else None
            }
            for i, chunk in enumerate(result["chunks"])
        ]

        safe_result = {
            "markdown": result["markdown"],
            "json": result["json"],
            "chunks": safe_chunks
        }

        return JSONResponse(content=safe_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ProcessPDFRequest(BaseModel):
    url_or_path: str
    project_id: str


@app.post("/api/ai-analyze/process-document")
async def process_pdf_endpoint(request: ProcessPDFRequest):
    try:
        points = process_document_to_qdrant(request.url_or_path, request.project_id)

        safe_points = [
            {
                "id": str(p.id),
                "vector": list(p.vector),
                "payload": p.payload
            }
            for p in points
        ]

        return JSONResponse(content={
            "project_id": request.project_id,
            "num_chunks": len(safe_points),
            "points": safe_points
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SearchRequest(BaseModel):
    query: str
    project_id: str
    limit: int = 3
@app.post("/api/ai-analyze/search")
async def search_endpoint(request: SearchRequest):
    try:
        results = search_qdrant(request.query, request.project_id, request.limit)
        return {
            "query": request.query,
            "project_id": request.project_id,
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    query: str
    project_id: str
    num_results: Optional[int] = 5

@app.post("/api/ai-analyze/ask")
def ask(req: QueryRequest):
    try:
        result = ask_question(
            question=req.query,
            project_id=req.project_id,
            num_results=req.num_results
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/ai-analyze/delete-project")
async def delete_document(collection: str, project_id: str):
    """
    Supprime tous les vecteurs liés à un doc_id dans une collection Qdrant
    """
    try:
        delete_by_project_id(collection, project_id)
        return {"status": "ok", "message": f"Vecteurs avec project_id={project_id} supprimés de '{collection}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur suppression: {str(e)}")

class RequirementChangeRequest(BaseModel):
    old_desc: str
    new_desc: str

@app.post("/api/ai-analyze/analyze-spec-changes")
async def analyze_spec_changes(request: RequirementChangeRequest):
    """
    Analyze requirement changes between two Jira ticket descriptions.
    """
    changes = analyze_requirement_changes(
        old_description=request.old_desc,
        new_description=request.new_desc
    )
    return changes
