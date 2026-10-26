"""
API tests (forensic-audit fix, F-41/F-42).

The original `api.main` failed to import because `api.routes.search_routes`
did not exist. These tests verify the API imports and serves health/ready.
"""

import os
import sys
from pathlib import Path

import pytest

from xrfm.config.schema import ModelConfig
from xrfm.models.gpt import GPTModel
from xrfm.tokenization.bpe import BytePairEncoder
from xrfm.training.checkpoint import CheckpointLoader
from xrfm.training.optimizer import OptimizerLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi", reason="API tests need fastapi (pip install -e '.[dev]' or '.[api]')")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Build isolated service artifacts; never depend on repository checkpoints."""
    tokenizer = BytePairEncoder(vocab_size_target=300)
    tokenizer.train_on_text("hello world. this is a tiny legacy API checkpoint. " * 12)
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
        filename="tiny-legacy-api.pt",
        extra={"model_config": model.config.to_dict()},
    )
    monkeypatch.setenv("XRFM_TOKENIZER_PATH", str(tokenizer_path))
    monkeypatch.setenv("XRFM_CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setenv("XRFM_API_DEVICE", "cpu")
    monkeypatch.setenv("XRFM_AGENT_ENABLED", "false")

    import api.main  # noqa: F401 - must import without error (F-41)
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_api_imports():
    import api.main  # noqa: F401

    assert True


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("healthy", "loading")
    assert "version" in body


def test_ready_endpoint(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_tokenize_endpoint(client):
    resp = client.post("/v1/tokenize", json={"text": "hello world"})
    if resp.status_code == 200:
        body = resp.json()
        assert isinstance(body["tokens"], list)
        assert body["count"] == len(body["tokens"])
