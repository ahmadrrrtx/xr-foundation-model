"""
Validation checklist for Phase 1.

Executes real commands and verifies:
- dataset manifest exists
- checksums exist
- statistics generated
- outputs deterministic
- training step succeeds
- dataset identity same on repeat, changes when config changes
"""

import os
import sys
import tempfile
import shutil

# Ensure src in path
sys.path.insert(0, "src")

from xrfm.data.schema import Document
from xrfm.data.pipeline import PipelineConfig, DataPipeline
from xrfm.data.sharding import ShardingConfig
from xrfm.tokenization import BPETokenizer
from xrfm.data.tokenized_dataset import ShardedTokenDatasetMap
from xrfm import XRFMModel, load_config
import torch
from torch.utils.data import DataLoader


def load_golden():
    docs = []
    with open("tests/data/golden/documents.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            docs.append(Document.from_jsonl(line))
    return docs


def run_pipeline(output_dir, sequence_length=128, num_shards=2):
    docs = load_golden()
    tokenizer = BPETokenizer.pretrained()
    config = PipelineConfig(
        dataset_name="validation",
        dataset_version="0.1.0",
        output_dir=output_dir,
        sequence_length=sequence_length,
    )
    config.sharding = ShardingConfig(num_shards=num_shards, output_dir=os.path.join(output_dir, "tokens"))
    config.packing.sequence_length = sequence_length

    pipeline = DataPipeline(config=config, tokenizer=tokenizer)
    result = pipeline.run(docs)
    return result


def main():
    print("=== Phase 1 Validation Checklist ===\n")

    # 1. Golden pipeline from start to finish
    print("1. Running golden data pipeline...")
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pipeline(tmpdir, sequence_length=128, num_shards=2)

        manifest_path = result["manifest_path"]
        report = result["report"]
        shard_infos = result["shard_infos"]

        # 2. Verify manifest exists
        assert os.path.isfile(manifest_path), "Manifest does not exist"
        print(f"   ✓ Manifest exists: {manifest_path}")

        # 3. Checksums exist
        assert result["manifest"].checksums, "Checksums missing"
        for si in shard_infos:
            assert si.sha256, f"Shard {si.shard_id} missing checksum"
            assert os.path.isfile(si.path), f"Shard file missing: {si.path}"
            # Verify checksum
            from xrfm.data.checksums import sha256_file

            actual = sha256_file(si.path)
            assert actual == si.sha256, f"Checksum mismatch for shard {si.shard_id}"
        print(f"   ✓ Checksums exist and verified for {len(shard_infos)} shards")

        # 4. Statistics generated
        assert report.documents_ingested > 0
        assert report.token_count > 0
        assert report.train_size + report.val_size + report.test_size > 0
        print(f"   ✓ Statistics generated: {report.documents_ingested} ingested, {report.token_count} tokens")

        # 5. Load shards through training data interface
        print("\n2. Loading shards through XRFM training interface...")
        ds = ShardedTokenDatasetMap(manifest_path=manifest_path)
        assert len(ds) > 0
        input_ids, targets = ds[0]
        assert input_ids.shape[0] == 128
        print(f"   ✓ Training dataset loads: len={len(ds)}, sample shape={input_ids.shape}")

        # 6. One real training step
        print("\n3. Running one real training step...")
        cfg = load_config("config/tiny.yaml")
        cfg.model.vocab_size = BPETokenizer.pretrained().vocab_size()
        model = XRFMModel(cfg.model)
        loader = DataLoader(ds, batch_size=2)
        batch = next(iter(loader))
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        input_batch, target_batch = batch
        logits = model(input_batch)[0] if isinstance(model(input_batch), tuple) else model(input_batch)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_batch.view(-1))
        loss.backward()
        optimizer.step()
        print(f"   ✓ Training step succeeded, loss={loss.item():.4f}")

        # 7. Deterministic outputs
        print("\n4. Verifying deterministic outputs...")
        with tempfile.TemporaryDirectory() as tmpdir2:
            result2 = run_pipeline(tmpdir2, sequence_length=128, num_shards=2)
            id1 = result["manifest"].dataset_id
            id2 = result2["manifest"].dataset_id
            assert id1 == id2, f"Dataset IDs differ for same config: {id1} vs {id2}"
            print(f"   ✓ Deterministic: same config → same dataset_id={id1}")

        # 8. Identity changes when config changes
        print("\n5. Verifying dataset identity changes with config...")
        with tempfile.TemporaryDirectory() as tmpdir3:
            result3 = run_pipeline(tmpdir3, sequence_length=256, num_shards=2)
            id3 = result3["manifest"].dataset_id
            assert id1 != id3, f"Dataset ID should change when seq_len changes"
            print(f"   ✓ Identity changes: seq_len 128 → {id1[:16]}, seq_len 256 → {id3[:16]}")

        # 9. Worker no duplicates
        print("\n6. Verifying worker sharding no duplicates...")
        from xrfm.data.sharding import verify_worker_no_duplicates

        for nw in [1, 2, 4, 8]:
            assert verify_worker_no_duplicates(len(ds), nw), f"Worker dup check failed for {nw} workers"
        print(f"   ✓ Worker sharding no duplicates for 1,2,4,8 workers")

        # 10. Split no overlap
        print("\n7. Verifying split no overlap...")
        from xrfm.data.document_split import verify_no_overlap

        splits = result["splits"]
        assert verify_no_overlap(splits), "Train/val/test overlap detected"
        print(f"   ✓ No overlap: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    print("\n=== All validation checks passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
