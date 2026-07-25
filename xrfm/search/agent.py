"""Search Agent for XRFM Local LLM Search Engine."""

from typing import Any

import torch

from inference.engine import GenerationEngine
from tokenizer.bpe import BytePairEncoder
from xrfm.search.indexer import SearchIndexer
from xrfm.search.retriever import SearchRetriever


class LocalSearchAgent:
    """Grounded Local LLM Search Agent."""

    def __init__(
        self,
        engine: GenerationEngine,
        tokenizer: BytePairEncoder,
        indexer: SearchIndexer | None = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.indexer = indexer or SearchIndexer()
        self.retriever = SearchRetriever(self.indexer)

    def add_document(
        self, doc_id: str, text: str, title: str = "", source_url: str | None = None
    ) -> int:
        return self.indexer.add_document(doc_id, text, title=title, source_url=source_url)

    def search_and_generate(
        self,
        query: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k_retrieval: int = 3,
    ) -> dict[str, Any]:
        retrieved_sources = self.retriever.retrieve(query, top_k=top_k_retrieval)
        prompt = self.retriever.format_context_prompt(query, top_k=top_k_retrieval)
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)

        output_ids = self.engine.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        prompt_len = input_ids.shape[1]
        new_ids = output_ids[prompt_len:]
        answer_text = self.tokenizer.decode(new_ids.tolist())

        return {
            "query": query,
            "answer": answer_text,
            "sources": retrieved_sources,
            "prompt_used": prompt,
        }
