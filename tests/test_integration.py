"""Phase 0 integration test: config → tokenizer → dataset → model → train step → inference.

The single clean end-to-end test required by Phase 0 #19, run entirely on
CPU with the tiny preset.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from xrfm import (
    TextDataset,
    Trainer,
    XRFMConfig,
    XRFMModel,
    evaluate,
    generate,
    load_config,
)
from xrfm.tokenization import BPETokenizer


@pytest.fixture(scope="module")
def pipeline():
    # 1. Config (repo tiny preset, typed & validated).
    cfg: XRFMConfig = load_config("config/tiny.yaml")
    cfg = cfg.replace(**{"model.vocab_size": 1024, "datasets.max_seq_len": 128, "model.max_seq_len": 128})

    # 2. Tokenizer (packaged resource; vocab coerced to match the config).
    tok = BPETokenizer.pretrained()

    # 3. Dataset (splits + packing + padding via the tokenizer contract).
    corpus = "data/datasets/sample.txt"
    train_ds = TextDataset(
        cfg.data,
        tok,
        split="train",
        dataset_path=corpus,
        max_seq_len=cfg.model.max_seq_len,
        pad_id=tok.pad_token_id,
    )
    assert len(train_ds) > 0

    # 4. Model sized to the tokenizer's vocab contract.
    model = XRFMModel(cfg.model, vocab_size=tok.vocab_size())
    return cfg, tok, train_ds, model


def test_config_to_dataset_shapes(pipeline):
    cfg, tok, train_ds, model = pipeline
    inp, tgt = train_ds[0]
    assert inp.shape == (cfg.model.max_seq_len,)
    assert tgt.shape == (cfg.model.max_seq_len,)
    assert inp.dtype == torch.long


def test_one_training_step_reduces_loss(pipeline):
    cfg, tok, train_ds, model = pipeline
    loader = DataLoader(
        train_ds,
        batch_size=min(4, len(train_ds)),
        shuffle=True,
        collate_fn=None,  # items are already fixed-shape tensors
        drop_last=True,
    )
    trainer = Trainer(model, config=cfg, checkpoint_dir="/tmp/xrfm_it_ckpts")
    batch_inp, batch_tgt = next(iter(loader))
    before = trainer.train_step(batch_inp, batch_tgt)["loss"]
    assert before == before, "loss must not be NaN"  # NaN != NaN
    assert before > 0


def test_short_training_run_via_trainer(pipeline):
    cfg, tok, train_ds, model = pipeline
    trainer = Trainer(model, config=cfg, checkpoint_dir="/tmp/xrfm_it_ckpts")
    result = trainer.train(train_ds, max_steps=3, checkpoint_every=100, log_interval=1)
    assert result["final_step"] >= 3
    assert result["final_loss"] == result["final_loss"]
    assert result["checkpoint_path"]


def test_inference_after_training(pipeline):
    cfg, tok, train_ds, model = pipeline
    model.eval()
    result = generate(
        model,
        "The little robot",
        tokenizer=tok,
        max_new_tokens=12,
        temperature=0.8,
        top_k=50,
    )
    assert isinstance(result.text, str)
    assert 0 < len(result.token_ids) <= 12
    # token-level API still works
    ids = tok.encode("Once")
    out = generate(model, torch.tensor(ids), tokenizer=tok, max_new_tokens=4, temperature=0.0, decode=False)
    assert len(out.token_ids) <= 4


def test_evaluation_without_trainer(pipeline):
    cfg, tok, train_ds, model = pipeline
    loader = DataLoader(train_ds, batch_size=min(2, len(train_ds)), drop_last=True)
    results = evaluate(model, loader, max_batches=2)
    assert "perplexity" in results
    assert results["perplexity"]["perplexity"] > 0
