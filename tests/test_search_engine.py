"""
Unit tests for XRFM Search Engine & RAG module.
"""

import os
import tempfile

from xrfm.search.agent import LocalSearchAgent
from xrfm.search.indexer import SearchIndexer
from xrfm.search.retriever import SearchRetriever

from inference.engine import GenerationEngine
from model.gpt import GPTModel
from tokenizer.bpe import BytePairEncoder


class TestSearchIndexer:
    def test_chunking_and_bm25_search(self):
        indexer = SearchIndexer(chunk_size=50, chunk_overlap=10)
        doc1 = "PyTorch is an open source machine learning framework based on the Torch library."
        doc2 = "FastAPI is a modern, fast web framework for building APIs with Python."

        indexer.add_document("doc1", doc1, title="PyTorch Info")
        indexer.add_document("doc2", doc2, title="FastAPI Info")

        results = indexer.search("machine learning", top_k=2)
        assert len(results) > 0
        assert results[0][0].doc_id == "doc1"

    def test_save_and_load_index(self):
        indexer = SearchIndexer()
        indexer.add_document(
            "doc1", "Artificial intelligence and local language models.", title="AI Doc"
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            save_path = f.name

        try:
            indexer.save(save_path)
            loaded_indexer = SearchIndexer()
            loaded_indexer.load(save_path)
            assert len(loaded_indexer.chunks) == len(indexer.chunks)
            assert loaded_indexer.total_docs == indexer.total_docs
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestSearchRetrieverAndAgent:
    def test_retriever_prompt_formatting(self):
        indexer = SearchIndexer()
        indexer.add_document(
            "doc1", "DeepSeek and Llama 3 are open-source LLM families.", title="LLM Models"
        )
        retriever = SearchRetriever(indexer)

        prompt = retriever.format_context_prompt("What are DeepSeek and Llama 3?")
        assert "DeepSeek" in prompt
        assert "Llama 3" in prompt

    def test_search_agent_generation(self):
        model = GPTModel()
        engine = GenerationEngine(model)
        tokenizer = BytePairEncoder()

        agent = LocalSearchAgent(engine=engine, tokenizer=tokenizer)
        agent.add_document(
            "doc1",
            "XRFM is a high performance foundation model built in PyTorch.",
            title="XRFM Overview",
        )

        res = agent.search_and_generate("What is XRFM?", max_new_tokens=10)
        assert "query" in res
        assert "answer" in res
        assert "sources" in res
        assert len(res["sources"]) > 0
