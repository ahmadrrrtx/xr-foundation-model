"""Phase 0 defect regression suite (GATE-0).

Every test in this file reproduces an empirically confirmed defect from
``docs/research/XRFM_FULL_REPOSITORY_AUDIT.md`` and ``XRFM_TARGET_ARCHITECTURE.md``:

- F-04: ``MultiHeadAttention`` materialized a dense causal mask on unmasked
  full-sequence training, disabling ``is_causal=True`` FlashAttention/MemEfficient
  SDPA; also adds GQA (``n_kv_heads``) support.
- F-05: ``RoPE`` recomputed uncached FP32 trig tables every step and upcast
  ``fp16``/``bf16`` Q & K to ``float32``, breaking mixed-precision SDPA.
- F-06: ``RMSNorm`` squared inputs in ``float16``, overflowing to ``inf``/``NaN``
  at ``|x| > 256`` (critical on Kaggle P100/T4).
- F-07: ``ModelConfig(use_swiglu=False)`` was silently ignored by
  ``TransformerBlock``/``XRFMModel``.
- F-08: ``XRFMModel.forward`` unconditionally ran ``(input_ids >= vocab_size).any()``
  on every forward pass (host-device sync); now controlled by ``check_input_bounds``.
- F-09: ``OptimizerLoader`` applied ``weight_decay`` to 1-D ``RMSNorm`` weights
  and biases when constructed via ``TrainingLoop``.
- F-10: ``TrainingLoop`` wrapped DDP/FSDP *before* resolving ``self.device`` and
  moving ``self.model.to(self.device)``.
- F-11: ``SchedulerLoader.get_lr`` oscillated past ``max_steps`` instead of
  clamping at ``min_lr_ratio * base_lr``.
- F-12: ``compute_perplexity``, ``TextCompletionAccuracy``, and ``TopKAccuracy``
  failed to move batches to ``model``'s device, failed on 3-tuple ``(inp, tgt, mask)``
  dataloaders, counted ``ignore_index=-100`` padding targets as wrong predictions,
  and lacked ``compute_bits_per_byte`` for fair multilingual tokenizer evaluation.
- F-17: ``README.md`` quickstart (``load_config("config/tiny.yaml")`` +
  ``BPETokenizer.pretrained()``) crashed with ``IndexError`` because ``tiny.yaml``
  set ``vocab_size: 1024`` while ``BPETokenizer.pretrained()`` emits IDs up to ``2047``.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import xrfm
from xrfm.config.schema import DatasetConfig, ModelConfig, TrainingConfig, XRFMConfig
from xrfm.evaluation import (
    TextCompletionAccuracy,
    TopKAccuracy,
    compute_perplexity,
    compute_perplexity_strided,
)
from xrfm.models.attention.multi_head import MultiHeadAttention
from xrfm.models.attention.rope import RoPE
from xrfm.models.gpt import XRFMModel
from xrfm.models.layers.rmsnorm import RMSNorm
from xrfm.models.layers.swiglu import SwiGLU
from xrfm.training.loop import TrainingLoop
from xrfm.training.scheduler import SchedulerLoader


# ---------------------------------------------------------------------------
# F-05 & F-06: Mixed-Precision RoPE & RMSNorm Stability
# ---------------------------------------------------------------------------
class TestMixedPrecisionLayers:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_rope_preserves_input_dtype_and_caches_tables(self, dtype: torch.dtype) -> None:
        rope = RoPE(d_model=32, max_seq_len=64, base=10000.0)
        q = torch.randn(2, 4, 16, 32, dtype=dtype)
        k = torch.randn(2, 4, 16, 32, dtype=dtype)
        q_rot, k_rot = rope(q, k, seq_len=16, offset=0)

        assert q_rot.dtype == dtype, f"RoPE changed Q dtype from {dtype} to {q_rot.dtype}"
        assert k_rot.dtype == dtype, f"RoPE changed K dtype from {dtype} to {k_rot.dtype}"
        assert hasattr(rope, "_cos_cached") and rope._cos_cached is not None
        assert rope._cos_cached.shape[0] >= 64

    def test_rope_dynamically_extends_cache_past_initial_max_seq_len(self) -> None:
        rope = RoPE(d_model=16, max_seq_len=8)
        q = torch.randn(1, 2, 12, 16, dtype=torch.float16)
        k = torch.randn(1, 2, 12, 16, dtype=torch.float16)
        q_rot, k_rot = rope(q, k, seq_len=12, offset=4)
        assert q_rot.shape == (1, 2, 12, 16)
        assert q_rot.dtype == torch.float16
        assert torch.isfinite(q_rot).all() and torch.isfinite(k_rot).all()

    def test_rmsnorm_fp16_does_not_overflow_large_activations(self) -> None:
        """In fp16, 300.0**2 = 90000 > 65504 (fp16 max) overflows without fp32 variance."""
        norm = RMSNorm(dim=64, eps=1e-6).to(torch.float16)
        x = torch.full((2, 8, 64), 300.0, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all(), "RMSNorm overflowed to NaN/Inf in float16"
        assert torch.allclose(out.float(), torch.ones_like(out, dtype=torch.float32), atol=1e-2)

    def test_full_model_bf16_forward_backward_succeeds(self) -> None:
        cfg = ModelConfig(
            vocab_size=128,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            max_seq_len=32,
            dropout=0.0,
            gradient_checkpointing=True,
        )
        model = XRFMModel(cfg).to(torch.bfloat16)
        ids = torch.randint(0, 128, (2, 16))
        logits, _ = model(ids)
        assert logits.dtype == torch.bfloat16
        loss = logits.float().sum()
        loss.backward()
        assert model.embedding.embedding.weight.grad is not None


# ---------------------------------------------------------------------------
# F-04, F-07 & GQA: Attention Native Causal SDPA, GQA & SwiGLU Toggle
# ---------------------------------------------------------------------------
class TestAttentionGQAAndSwiGLU:
    def test_unmasked_attention_dispatches_sdpa_with_none_mask(self) -> None:
        """Unmasked full-sequence forward MUST pass mask=None so SDPA uses is_causal=True."""
        import xrfm.optimization.flash_attention as fa

        attn = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
        x = torch.randn(2, 12, 32)
        captured: dict[str, object] = {}

        real_fn = fa.flash_attention_forward

        def spy_flash(q, k, v, mask=None, dropout_p=0.0, training=False, force_backend=None):
            captured["mask"] = mask
            return real_fn(q, k, v, mask=mask, dropout_p=dropout_p, training=training, force_backend=force_backend)

        with patch("xrfm.optimization.flash_attention.flash_attention_forward", side_effect=spy_flash):
            out, _ = attn(x, mask=None, use_cache=False)

        assert out.shape == (2, 12, 32)
        assert "mask" in captured
        assert captured["mask"] is None, "Unmasked full-sequence attention materialized a dense mask!"

    def test_gqa_kv_cache_matches_full_sequence_forward(self) -> None:
        torch.manual_seed(7)
        cfg = ModelConfig(
            vocab_size=64,
            d_model=64,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,
            d_ff=128,
            max_seq_len=32,
            dropout=0.0,
            rope_theta=50000.0,
        )
        model = XRFMModel(cfg).eval()
        ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            full_logits, _ = model(ids, use_cache=False)
            prefill_logits, kv = model(ids[:, :6], use_cache=True)
            assert kv is not None
            assert kv[0][0].shape == (1, 2, 6, 8)  # (B, n_kv_heads=2, seq=6, d_head=8)
            step_logits = [prefill_logits]
            for pos in range(6, 10):
                tok_logits, kv = model(ids[:, pos : pos + 1], past_key_values=kv, use_cache=True)
                step_logits.append(tok_logits)
            cached_logits = torch.cat(step_logits, dim=1)

        assert torch.allclose(full_logits, cached_logits, atol=1e-5)

    def test_use_swiglu_false_builds_standard_gelu_ffn(self) -> None:
        cfg_swiglu = ModelConfig(
            vocab_size=64, d_model=32, n_layers=1, n_heads=4, d_ff=128, max_seq_len=16, dropout=0.0, use_swiglu=True
        )
        cfg_gelu = ModelConfig(
            vocab_size=64, d_model=32, n_layers=1, n_heads=4, d_ff=128, max_seq_len=16, dropout=0.0, use_swiglu=False
        )
        m_swiglu = XRFMModel(cfg_swiglu)
        m_gelu = XRFMModel(cfg_gelu)
        assert isinstance(m_swiglu.blocks[0].ffn, SwiGLU)
        assert not isinstance(m_gelu.blocks[0].ffn, SwiGLU)
        # SwiGLU has 3 matrices (W_1, W_2, W_3); standard GELU FFN has 2 matrices -> strictly fewer params
        assert m_gelu.parameter_count() < m_swiglu.parameter_count()


# ---------------------------------------------------------------------------
# F-09, F-10 & F-11: Optimizer Grouping, Device Order & Cosine LR Clamping
# ---------------------------------------------------------------------------
class TestTrainingFixes:
    def test_training_loop_excludes_1d_norms_and_biases_from_weight_decay(self) -> None:
        cfg = XRFMConfig(
            model=ModelConfig(
                vocab_size=64,
                d_model=32,
                n_layers=2,
                n_heads=4,
                d_ff=64,
                max_seq_len=16,
                dropout=0.0,
                use_bias=True,
            ),
            training=TrainingConfig(batch_size=2, max_steps=20, warmup_steps=2, weight_decay=0.1),
            data=DatasetConfig(max_seq_len=16),
        )
        model = XRFMModel(cfg)
        loop = TrainingLoop(config=cfg, model=model)
        groups = loop.optimizer.optimizer.param_groups
        assert len(groups) == 2, "Expected decay and no_decay parameter groups"
        decay_group = next(g for g in groups if g["weight_decay"] > 0.0)
        nodecay_group = next(g for g in groups if g["weight_decay"] == 0.0)
        assert all(p.dim() >= 2 for p in decay_group["params"])
        assert all(p.dim() < 2 for p in nodecay_group["params"])
        # RMSNorm weights and Linear biases must be in the no_decay group
        nodecay_ids = {id(p) for p in nodecay_group["params"]}
        assert id(model.norm_final.weight) in nodecay_ids
        assert id(model.blocks[0].attn.W_q.bias) in nodecay_ids

    def test_training_loop_moves_model_to_device_before_ddp_wrapping(self) -> None:
        cfg = XRFMConfig(
            model=ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=8, dropout=0.0),
            training=TrainingConfig(batch_size=2, max_steps=10, warmup_steps=2, use_ddp=True, device="meta"),
            data=DatasetConfig(max_seq_len=8),
        )
        model = XRFMModel(cfg.model)  # starts on CPU
        observed_devices: list[str] = []

        def fake_wrap_ddp(m: torch.nn.Module) -> torch.nn.Module:
            observed_devices.append(next(m.parameters()).device.type)
            return m

        with (
            patch("xrfm.training.loop.is_distributed", return_value=True),
            patch("xrfm.training.loop.wrap_model_ddp", side_effect=fake_wrap_ddp),
        ):
            TrainingLoop(config=cfg, model=model)

        assert observed_devices == ["meta"], f"Model was wrapped before moving to target device: {observed_devices}"

    def test_scheduler_clamps_cosine_decay_past_max_steps_and_honors_min_lr_ratio(self) -> None:
        p = torch.nn.Parameter(torch.zeros(4))
        opt = torch.optim.AdamW([p], lr=1e-3)
        sched = SchedulerLoader(opt, base_lr=1e-3, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
        lr_at_max = sched.get_lr(step=100)
        lr_past_max = sched.get_lr(step=150)
        assert math.isclose(lr_at_max, 1e-4, rel_tol=1e-6)
        assert math.isclose(lr_past_max, 1e-4, rel_tol=1e-6), "Scheduler rebounded upward past max_steps!"


# ---------------------------------------------------------------------------
# F-12: Evaluation Padding Masking, Device Placement, 3-Tuple & BPB
# ---------------------------------------------------------------------------
class TestEvaluationFixes:
    def test_accuracy_benchmarks_ignore_minus_100_padding_and_accept_3_tuple_batches(self) -> None:
        cfg = ModelConfig(vocab_size=16, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=8, dropout=0.0)
        model = XRFMModel(cfg).eval()
        inp = torch.tensor([[1, 2, 3, 0, 0, 0]])
        with torch.no_grad():
            logits, _ = model(inp)
            greedy = logits.argmax(dim=-1)
        # First 3 tokens match model's greedy top-1 prediction; last 3 are -100 padding!
        tgt = greedy.clone()
        tgt[:, 3:] = -100
        mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        loader = DataLoader(TensorDataset(inp, tgt, mask), batch_size=1)

        top1 = TextCompletionAccuracy().compute(model, loader)
        top5 = TopKAccuracy(k=5).compute(model, loader)
        ppl = compute_perplexity(model, loader)

        assert top1["total_tokens"] == 3, f"Counted -100 padding tokens: {top1}"
        assert math.isclose(top1["accuracy"], 1.0)
        assert top5["total_tokens"] == 3
        assert math.isclose(top5["top5_accuracy"], 1.0)
        assert ppl["total_tokens"] == 3

    def test_evaluation_moves_cpu_batches_to_model_device(self) -> None:
        cfg = ModelConfig(vocab_size=16, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=8, dropout=0.0)
        model = XRFMModel(cfg).eval()
        seen_devices: list[torch.device] = []

        def tracking_forward(input_ids, *args, **kwargs):
            seen_devices.append(input_ids.device)
            b, s = input_ids.shape
            return torch.zeros((b, s, 16), dtype=torch.float32), None

        # Fake _model_device returning meta to prove evaluation calls .to(device) on inputs
        with (
            patch("xrfm.evaluation.perplexity._get_model_device", return_value=torch.device("meta")),
            patch("xrfm.evaluation.benchmarks._get_model_device", return_value=torch.device("meta")),
            patch.object(model, "forward", side_effect=tracking_forward),
        ):
            inp = torch.tensor([[1, 2, 3, 4]])
            tgt = torch.tensor([[2, 3, 4, 5]])
            loader = DataLoader(TensorDataset(inp, tgt), batch_size=1)
            compute_perplexity(model, loader)
            TextCompletionAccuracy().compute(model, loader)
            TopKAccuracy(k=3).compute(model, loader)
            compute_perplexity_strided(model, torch.tensor([1, 2, 3, 4, 5, 6]), stride=2, max_seq_len=4)

        assert all(d.type == "meta" for d in seen_devices), f"Inputs were not moved to model device: {seen_devices}"

    def test_compute_bits_per_byte_multilingual_roundtrip(self) -> None:
        from xrfm.evaluation import compute_bits_per_byte

        tok = xrfm.BPETokenizer.pretrained()
        cfg = ModelConfig(
            vocab_size=tok.vocab_size(), d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=64, dropout=0.0
        )
        model = XRFMModel(cfg).eval()
        texts = [
            "Secure coding prevents SQL injection.",
            "محفوظ کوڈنگ سیکیورٹی کے لیے ضروری ہے۔",
            "Yeh code SQL injection ko block karta hai.",
        ]
        res = compute_bits_per_byte(model, texts, tok)
        assert res["bpb"] > 0.0 and math.isfinite(res["bpb"])
        assert res["total_bytes"] == sum(len(t.encode("utf-8")) for t in texts)
        assert res["total_tokens"] > 0


# ---------------------------------------------------------------------------
# F-17: README Quickstart Compatibility Out-of-the-Box
# ---------------------------------------------------------------------------
def test_readme_quickstart_tiny_yaml_matches_pretrained_tokenizer(tmp_path) -> None:
    cfg = xrfm.load_config("config/tiny.yaml")
    tok = xrfm.BPETokenizer.pretrained()
    assert cfg.model.vocab_size == tok.vocab_size(), (
        f"config/tiny.yaml vocab_size ({cfg.model.vocab_size}) != "
        f"BPETokenizer.pretrained().vocab_size() ({tok.vocab_size()})"
    )
    model = xrfm.XRFM(cfg)
    train_ds = xrfm.TextDataset(cfg.data, tok, split="train", dataset_path="data/datasets/sample.txt")
    trainer = xrfm.Trainer(model, config=cfg, checkpoint_dir=str(tmp_path / "ckpt"))
    result = trainer.train(train_ds, max_steps=2, checkpoint_every=2)
    assert result["final_step"] == 2
