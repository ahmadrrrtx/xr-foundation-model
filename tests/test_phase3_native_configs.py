"""
Phase 3 (`GATE-3`) Unit & Preset Tests:
Track A Native Model Configurations (`native-25m.yaml`, `native-85m.yaml`)
and Tied-Embedding `SafeTensors` Export/Import Parity.
"""

from __future__ import annotations

import json

import torch

from xrfm.config.loader import load_config
from xrfm.config.schema import ModelConfig
from xrfm.inference import generate
from xrfm.models.gpt import XRFMModel
from xrfm.tokenization.hf_bridge import HFTokenizerAdapter
from xrfm.training.safetensors_io import export_model_safetensors, load_model_safetensors


class TestPhase3NativePresetsAndSafeTensors:
    def test_native_25m_preset_parameter_budget_and_forward(self):
        cfg = load_config("config/native-25m.yaml")
        assert cfg.model.vocab_size == 32768
        assert cfg.model.n_heads == 6
        assert cfg.model.n_kv_heads == 2

        model = XRFMModel(cfg)
        unique_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
        assert 22_000_000 <= unique_params <= 28_000_000, f"Expected ~25M params, got {unique_params:,}"

        x = torch.randint(0, 32768, (2, 32))
        logits, _ = model(x)
        assert logits.shape == (2, 32, 32768)

    def test_native_85m_preset_parameter_budget(self):
        cfg = load_config("config/native-85m.yaml")
        assert cfg.model.vocab_size == 32768
        assert cfg.model.n_heads == 10
        assert cfg.model.n_kv_heads == 2
        assert cfg.model.gradient_checkpointing is True

        with torch.device("meta"):
            model = XRFMModel(cfg)
        unique_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
        assert 80_000_000 <= unique_params <= 92_000_000, f"Expected ~85M params, got {unique_params:,}"

    def test_safetensors_roundtrip_preserves_exact_logits_and_tied_weights(self, tmp_path):
        mcfg = ModelConfig(
            vocab_size=1024,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            max_seq_len=64,
            dropout=0.0,
        )
        m1 = XRFMModel(mcfg)
        m1.eval()
        st_path = str(tmp_path / "model.safetensors")
        export_model_safetensors(m1, st_path, metadata={"track": "native", "step": "100"})

        m2 = XRFMModel(mcfg)
        m2.eval()
        meta = load_model_safetensors(m2, st_path)
        assert meta.get("track") == "native"
        assert m2.lm_head.weight.data_ptr() == m2.embedding.embedding.weight.data_ptr()

        probe = torch.randint(0, 1024, (2, 16))
        with torch.no_grad():
            out1, _ = m1(probe)
            out2, _ = m2(probe)
        assert torch.equal(out1, out2)

    def test_generation_engine_relocates_prompt_to_model_device(self):
        mcfg = ModelConfig(
            vocab_size=32768,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            max_seq_len=64,
            dropout=0.0,
        )
        m = XRFMModel(mcfg)
        tok = HFTokenizerAdapter.from_file("src/xrfm/tokenization/tokenizer_32k_multilingual.json")
        res = generate(m, "اردو اور Python ٹیسٹ", tokenizer=tok, max_new_tokens=6, temperature=0.0)
        assert len(res.token_ids) == 6

    def test_gate3_kaggle_training_artifact_invariants(self):
        with open("docs/experiments/track_a_native_training_v1.json", encoding="utf-8") as f:
            art = json.load(f)
        assert art["gate_status"] == "PASS"
        expected_domains = ("en", "code", "ur", "ur-Latn", "ar", "hi", "es", "fr")
        for run_key in ("xrfm_native_25m", "xrfm_native_85m"):
            run = art["runs"][run_key]
            assert len(run["hf_commit"]) == 40
            assert len(run["safetensors_sha256"]) == 64
            tel = run["telemetry"]
            assert tel["gate3_pass"] is True
            assert tel["cuda_exact_resume_probe"]["device"] == "cuda"
            assert tel["cuda_exact_resume_probe"]["max_weight_abs_diff"] == 0.0
            assert tel["cuda_exact_resume_probe"]["exact_match"] is True
            traj = tel["bpb_trajectory"]
            assert len(traj) >= 3
            for dom in expected_domains:
                vals = [pt["bpb"][dom] for pt in traj]
                for i in range(1, len(vals)):
                    assert vals[i] < vals[i - 1], f"Non-monotonic BPB in {run_key}/{dom}: {vals}"
