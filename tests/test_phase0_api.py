"""Real API lifespan and completion-contract regressions for Phase 0."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.routes.completions import _stream_safe_text
from xrfm.config.schema import ModelConfig
from xrfm.models.gpt import GPTModel
from xrfm.tokenization.bpe import BytePairEncoder
from xrfm.training.checkpoint import CheckpointLoader
from xrfm.training.optimizer import OptimizerLoader


def make_tiny_service_artifacts(tmp_path: Path) -> tuple[Path, Path, BytePairEncoder]:
    tokenizer = BytePairEncoder(vocab_size_target=300)
    tokenizer.train_on_text("hello world. this is a tiny service checkpoint. " * 12)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    model = GPTModel(
        ModelConfig(
            vocab_size=tokenizer.vocab_size(),
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_seq_len=8,
            dropout=0.0,
            use_rope=False,
            pad_token_id=tokenizer.pad_id,
        )
    )
    optimizer = OptimizerLoader(model.parameters(), learning_rate=1e-3)
    checkpoint_path = CheckpointLoader(str(tmp_path)).save_checkpoint(
        model,
        optimizer,
        step=0,
        filename="tiny-service.pt",
        extra={"model_config": model.config.to_dict()},
    )
    return tokenizer_path, Path(checkpoint_path), tokenizer


def configure_service(monkeypatch: pytest.MonkeyPatch, tokenizer_path: Path, checkpoint_path: Path) -> None:
    monkeypatch.setenv("XRFM_TOKENIZER_PATH", str(tokenizer_path))
    monkeypatch.setenv("XRFM_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("XRFM_AGENT_ENABLED", "false")


def test_real_lifespan_models_and_completion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_path, checkpoint_path, _ = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    from api.main import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert len(models.json()) == 1
        response = client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_new_tokens": 2, "temperature": 0},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["choices"][0]["text"] is not None
        assert 1 <= body["usage"]["completion_tokens"] <= 2
        assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]


def test_completion_usage_uses_original_prompt_and_actual_generated_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenizer_path, checkpoint_path, tokenizer = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    from api.main import app

    prompt = "hello world " * 20
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"prompt": prompt, "max_new_tokens": 2, "temperature": 0},
        )
    assert response.status_code == 200, response.text
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == len(tokenizer.encode(prompt))
    assert 1 <= usage["completion_tokens"] <= 2
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_missing_checkpoint_is_liveness_only_not_readiness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XRFM_TOKENIZER_PATH", raising=False)
    monkeypatch.delenv("XRFM_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("XRFM_AGENT_ENABLED", "false")
    from api.main import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is False
        assert client.get("/health/ready").status_code == 503
        assert client.get("/v1/models").status_code == 503
        assert client.post("/v1/completions", json={"prompt": "hello"}).status_code == 503


def test_stream_stop_sequences_hold_partial_suffix_and_flush() -> None:
    assert _stream_safe_text("helloE", ["END"]) == "hello"
    assert _stream_safe_text("helloEN", ["END"]) == "hello"
    assert _stream_safe_text("helloEND", ["END"]) == "hello"
    assert _stream_safe_text("helloE", ["END"], terminal=True) == "helloE"
    assert _stream_safe_text("hello world", ["END"]) == "hello world"
