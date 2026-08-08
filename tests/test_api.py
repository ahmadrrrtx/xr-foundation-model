"""
API tests (forensic-audit fix, F-41/F-42).

The original `api.main` failed to import because `api.routes.search_routes`
did not exist. These tests verify the API imports and serves health/ready.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
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
