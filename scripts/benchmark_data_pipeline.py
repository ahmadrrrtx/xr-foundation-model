"""
Benchmark data pipeline throughput for Phase 1.

Measures:
- documents/sec
- MB/sec
- tokens/sec
- peak RAM
- output size
- for 1,2,4 workers where practical

This establishes baseline for later optimization.
Does NOT make performance claims without actual measurements.
"""

import os
import time
import psutil
import tempfile
from pathlib import Path

from xrfm.data.schema import Document
from xrfm.data.pipeline import PipelineConfig, DataPipeline
from xrfm.data.sharding import ShardingConfig
from xrfm.tokenization import BPETokenizer


def generate_synthetic_docs(num_docs: int, avg_chars: int = 1000) -> list[Document]:
    import random

    rng = random.Random(42)
    docs = []
    for i in range(num_docs):
        # Generate random-ish text
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "machine", "learning", "data", "model", "training"]
        text = " ".join(rng.choice(words) for _ in range(avg_chars // 5))
        docs.append(Document(text=text, source="synthetic", language="en", document_id=f"synth-{i}"))
    return docs


def benchmark(num_docs: int = 100, sequence_length: int = 128):
    tokenizer = BPETokenizer.pretrained()

    docs = generate_synthetic_docs(num_docs)
    total_chars = sum(len(d.text) for d in docs)
    total_mb = total_chars / (1024 * 1024)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            dataset_name=f"bench-{num_docs}",
            dataset_version="0.1.0",
            output_dir=tmpdir,
            sequence_length=sequence_length,
        )
        config.sharding = ShardingConfig(num_shards=4, output_dir=os.path.join(tmpdir, "tokens"))
        config.packing.sequence_length = sequence_length

        pipeline = DataPipeline(config=config, tokenizer=tokenizer)

        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        start = time.time()
        result = pipeline.run(docs)
        elapsed = time.time() - start

        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_mem = mem_after  # approximate

        total_tokens = result["report"].token_count
        num_sequences = sum(si.num_sequences for si in result["shard_infos"])
        storage_bytes = sum(os.path.getsize(si.path) for si in result["shard_infos"] if os.path.exists(si.path))

        docs_per_sec = num_docs / elapsed if elapsed > 0 else 0
        mb_per_sec = total_mb / elapsed if elapsed > 0 else 0
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

        print(f"\n=== Benchmark: {num_docs} docs, seq_len={sequence_length} ===")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Docs/sec: {docs_per_sec:.2f}")
        print(f"MB/sec: {mb_per_sec:.2f}")
        print(f"Tokens/sec: {tokens_per_sec:.2f}")
        print(f"Peak RAM: {peak_mem:.1f} MB (before {mem_before:.1f} MB)")
        print(f"Total tokens: {total_tokens}")
        print(f"Sequences: {num_sequences}")
        print(f"Output size: {storage_bytes} bytes ({storage_bytes/1e6:.2f} MB)")
        print(f"Shards: {len(result['shard_infos'])}")

        return {
            "num_docs": num_docs,
            "elapsed": elapsed,
            "docs_per_sec": docs_per_sec,
            "mb_per_sec": mb_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "peak_ram_mb": peak_mem,
            "total_tokens": total_tokens,
            "num_sequences": num_sequences,
            "storage_bytes": storage_bytes,
        }


if __name__ == "__main__":
    print("XRFM Data Pipeline Performance Baseline")
    print("========================================")

    results = []
    for n in [50, 100, 200]:
        res = benchmark(num_docs=n, sequence_length=128)
        results.append(res)

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['num_docs']} docs: {r['docs_per_sec']:.1f} docs/sec, {r['tokens_per_sec']:.1f} tokens/sec, {r['peak_ram_mb']:.1f} MB RAM")
