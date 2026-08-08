"""Search endpoints for the XRFM API (forensic-audit fix, F-41).

The original `api/main.py` imported `api.routes.search_routes`, which never
existed, making the whole API fail to import. This module provides the
missing search routes backed by the in-repo `xrfm.search` module.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import SearchRequest, SearchResponse, SearchResult
from xrfm.search.indexer import SearchIndexer

router = APIRouter()

_indexer: SearchIndexer | None = None


def _get_indexer() -> SearchIndexer:
    global _indexer
    if _indexer is None:
        _indexer = SearchIndexer()
    return _indexer


def set_indexer(indexer: SearchIndexer) -> None:
    """Allow the app lifespan to inject a pre-populated indexer."""
    global _indexer
    _indexer = indexer


@router.post("/v1/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """BM25 search over documents added to the local index."""
    indexer = _get_indexer()
    if not indexer.chunks:
        return SearchResponse(query=req.query, results=[], total=0)

    results = indexer.search(req.query, top_k=req.top_k)
    items = [
        SearchResult(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            text=chunk.text,
            score=round(score, 4),
            source_url=chunk.source_url,
        )
        for chunk, score in results
    ]
    return SearchResponse(query=req.query, results=items, total=len(items))


@router.post("/v1/search/index")
async def add_document(req: dict):
    """Index a document: {"doc_id": str, "text": str, "title": str = ""}."""
    doc_id = req.get("doc_id")
    text = req.get("text")
    if not doc_id or not text:
        raise HTTPException(400, "doc_id and text are required")
    n = _get_indexer().add_document(doc_id, text, title=req.get("title", ""))
    return {"indexed_chunks": n}
