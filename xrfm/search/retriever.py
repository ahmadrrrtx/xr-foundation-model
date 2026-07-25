"""Hybrid Search Retriever for XRFM Local LLM Search Engine."""

from typing import Any

from xrfm.search.indexer import SearchIndexer


class SearchRetriever:
    """Hybrid search retriever and context pack builder."""

    def __init__(self, indexer: SearchIndexer | None = None) -> None:
        self.indexer = indexer or SearchIndexer()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        results = self.indexer.search(query, top_k=top_k)

        retrieved_items = []
        for chunk, score in results:
            retrieved_items.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "score": round(score, 4),
                    "source_url": chunk.source_url,
                }
            )
        return retrieved_items

    def format_context_prompt(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return f"Question: {query}\nAnswer:"

        context_str = "Context Information:\n"
        for idx, item in enumerate(results, start=1):
            source_info = f" [Source: {item['title']}]" if item["title"] else ""
            context_str += f"[{idx}]{source_info}: {item['text']}\n"

        prompt = (
            f"You are a helpful local AI search assistant. Answer the question accurately using ONLY "
            f"the provided context below. Cite sources where applicable.\n\n"
            f"{context_str}\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        return prompt
