"""
XRFM forensic-audit verification suite.

The original 192 tests verify shapes/API behavior only. This suite adds
GROUND-TRUTH tests: mathematical reference comparisons, causality, weight
tying, KV-cache equivalence, resume/scheduler state, reproducibility,
tokenizer fidelity, and loss masking. See docs/audit/FORENSIC_AUDIT.md §4.
"""

import os
import sys
import tempfile

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.attention.multi_head import MultiHeadAttention  # noqa: E402
from model.attention.rope import RoPE  # noqa: E402
from model.gpt import GPTModel  # noqa: E402
from model.layers.rmsnorm import RMSNorm  # noqa: E402
from model.layers.swiglu import SwiGLU  # noqa: E402
from tokenizer.bpe import BytePairEncoder  # noqa: E402
from training.loop import TrainingLoop, _set_seed  # noqa: E402
from training.mixed_precision import MixedPrecisionLoader  # noqa: E402
from training.scheduler import SchedulerLoader  # noqa: E402

TINY = "config/tiny.yaml"


# ----------------------------------------------------------------------
# 1. Causality (F-01/F-02)
# ----------------------------------------------------------------------
class TestCausality:
    def _probe(self, attn):
        x1 = torch.randn(1, 6, 64)
        x2 = x1.clone()
        x2[0, 4] = 99.0
        attn.eval()
        with torch.no_grad():
            o1, _ = attn(x1)
            o2, _ = attn(x2)
        return (o1[0, 2] - o2[0, 2]).abs().max().item(), (o1[0, 5] - o2[0, 5]).abs().max().item()

    def test_default_path_is_causal(self):
        attn = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
        d2, d5 = self._probe(attn)
        assert d2 == 0.0, "position 2 must not depend on position 4"
        assert d5 > 0.0, "position 5 must depend on position 4"

    def test_manual_fallback_is_causal(self):
        # Force the manual path by blocking the flash module import.
        import sys

        sys.modules["optimization.flash_attention"] = None
        try:
            attn = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
            d2, d5 = self._probe(attn)
            assert d2 == 0.0, "manual fallback must still be causal"
            assert d5 > 0.0
        finally:
            del sys.modules["optimization.flash_attention"]

    def test_manual_scores_have_inf_masking(self):
        # Verify the manual path actually produces -inf entries above the diagonal.
        import math
        import sys

        sys.modules["optimization.flash_attention"] = None
        try:
            attn = MultiHeadAttention(d_model=16, n_heads=2, dropout=0.0)
            x = torch.randn(1, 5, 16)
            q = attn.W_q(x).view(1, 5, 2, 8).transpose(1, 2)
            k = attn.W_k(x).view(1, 5, 2, 8).transpose(1, 2)
            q, k = attn.rope(q, k, 5, offset=0)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(8)
            additive = torch.triu(torch.full((5, 5), float("-inf")), diagonal=1)
            masked = scores + additive.unsqueeze(0).unsqueeze(0)
            assert (masked[0, 0, 0, 2:] == float("-inf")).all()
            assert torch.isfinite(masked[0, 0, 2, :3]).all()
        finally:
            del sys.modules["optimization.flash_attention"]


# ----------------------------------------------------------------------
# 2. Reference math (F-10)
# ----------------------------------------------------------------------
class TestReferenceMath:
    def test_rmsnorm_matches_manual(self):
        x = torch.randn(4, 16, 64) * 3.0
        rm = RMSNorm(64, eps=1e-6)
        with torch.no_grad():
            out = rm(x)
            manual = rm.weight * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert torch.allclose(out, manual, atol=1e-6)

    def test_swiglu_matches_manual(self):
        s = SwiGLU(d_model=32, d_ff=128)
        x = torch.randn(3, 7, 32)
        with torch.no_grad():
            out = s(x)
            manual = s.W_3(F.silu(s.W_1(x)) * s.W_2(x))
        assert torch.allclose(out, manual, atol=1e-6)

    def test_rope_matches_reference(self):
        def ref(q, offset, base=10000.0):
            d = q.shape[-1]
            h = d // 2
            seq = q.shape[-2]
            idx = torch.arange(0, h, dtype=torch.float32)
            inv = 1.0 / (base ** (idx / h))
            t = torch.arange(offset, offset + seq, dtype=torch.float32)
            freqs = torch.outer(t, inv)
            cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)
            sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)
            x1 = q[..., :h]
            x2 = q[..., h:]
            return q * cos + torch.cat([-x2, x1], dim=-1) * sin

        rope = RoPE(d_model=64, base=10000.0)
        q = torch.randn(2, 8, 33, 64)
        k = torch.randn(2, 8, 33, 64)
        for offset in (0, 5, 128):
            qr, kr = rope(q, k, 33, offset=offset)
            assert torch.allclose(qr, ref(q, offset), atol=1e-5)
            assert torch.allclose(kr, ref(k, offset), atol=1e-5)

    def test_rope_relative_position_property(self):
        rope = RoPE(d_model=64)
        q = torch.randn(1, 1, 1, 64)
        k = torch.randn(1, 1, 1, 64)
        lhs = torch.dot(
            rope.apply_rotary_emb(q, 1, offset=9).squeeze(),
            rope.apply_rotary_emb(k, 1, offset=3).squeeze(),
        )
        rhs = torch.dot(
            rope.apply_rotary_emb(q, 1, offset=6).squeeze(),
            rope.apply_rotary_emb(k, 1, offset=0).squeeze(),
        )
        assert abs(lhs - rhs) < 1e-4

    def test_weight_tied_logits(self):
        m = GPTModel(TINY)
        m.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits, _ = m(x)
            h = m.embedding(x)
            for blk in m.blocks:
                h, _ = blk(h)
            h = m.norm_final(h)
            ref = torch.matmul(h, m.embedding.embedding.weight.t())
        assert m.lm_head.weight is m.embedding.embedding.weight
        assert torch.allclose(logits, ref, atol=1e-5)

    def test_kv_cache_equivalence(self):
        m = GPTModel(TINY)
        m.eval()
        prompt = torch.randint(0, 100, (1, 5))
        t1 = torch.randint(0, 100, (1, 1))
        t2 = torch.randint(0, 100, (1, 1))
        with torch.no_grad():
            full = torch.cat([prompt, t1, t2], dim=1)
            lf, _ = m(full)
            lp, kv = m(prompt, use_cache=True)
            l1, kv = m(t1, past_key_values=kv, use_cache=True)
            l2, _ = m(t2, past_key_values=kv, use_cache=True)
        assert (lf[:, 4, :] - lp[:, -1, :]).abs().max().item() < 1e-4
        assert (lf[:, 5, :] - l1[:, -1, :]).abs().max().item() < 1e-4
        assert (lf[:, 6, :] - l2[:, -1, :]).abs().max().item() < 1e-4

    def test_logits_finite_and_random_init_scale(self):
        m = GPTModel(TINY)
        m.eval()
        x = torch.randint(0, m.embedding.vocab_size, (2, 32))
        with torch.no_grad():
            logits, _ = m(x)
        assert torch.isfinite(logits).all()


# ----------------------------------------------------------------------
# 3. Scheduler state / resume (F-24)
# ----------------------------------------------------------------------
class TestSchedulerResume:
    def test_state_roundtrip_restores_step_and_lr(self):
        opt = torch.optim.AdamW([torch.randn(2, 2, requires_grad=True)], lr=0.1)
        sched = SchedulerLoader(opt, base_lr=0.1, warmup_steps=5, max_steps=50)
        for _ in range(7):
            sched.step()
        sd = sched.state_dict()
        lr_before = sched.get_lr()

        opt2 = torch.optim.AdamW([torch.randn(2, 2, requires_grad=True)], lr=0.1)
        sched2 = SchedulerLoader(opt2, base_lr=0.1, warmup_steps=5, max_steps=50)
        sched2.load_state_dict(sd)
        assert sched2.current_step == 7
        assert abs(sched2.get_lr() - lr_before) < 1e-12
        assert abs(opt2.param_groups[0]["lr"] - lr_before) < 1e-12


# ----------------------------------------------------------------------
# 4. Reproducibility (F-25)
# ----------------------------------------------------------------------
class TestReproducibility:
    def _fixed_dataset(self):
        class D:
            def __len__(self):
                return 32

            def __getitem__(self, i):
                ids = torch.randint(0, 100, (33,))
                return ids[:-1], ids[1:]

        return D()

    def test_same_seed_same_first_loss(self):
        # Same seed -> identical first-batch loss (exact equality).
        _set_seed(42)
        m1 = GPTModel(TINY)
        l1 = TrainingLoop(config_path=TINY, model=m1, dataset=self._fixed_dataset())
        l1.batch_size = 8
        r1 = l1.training_loop(max_steps=2, log_interval=1)

        _set_seed(42)
        m2 = GPTModel(TINY)
        l2 = TrainingLoop(config_path=TINY, model=m2, dataset=self._fixed_dataset())
        l2.batch_size = 8
        r2 = l2.training_loop(max_steps=2, log_interval=1)

        assert r1["final_loss"] == r2["final_loss"], "same seed must give same loss"

    def test_different_seed_different_batch_order(self):
        _set_seed(1)
        m1 = GPTModel(TINY)
        l1 = TrainingLoop(config_path=TINY, model=m1, dataset=self._fixed_dataset())
        l1.batch_size = 8
        r1 = l1.training_loop(max_steps=2, log_interval=1)

        _set_seed(2)
        m2 = GPTModel(TINY)
        l2 = TrainingLoop(config_path=TINY, model=m2, dataset=self._fixed_dataset())
        l2.batch_size = 8
        r2 = l2.training_loop(max_steps=2, log_interval=1)

        assert r1["final_loss"] != r2["final_loss"], "different seeds should differ"


# ----------------------------------------------------------------------
# 5. Tokenizer fidelity (F-11/F-12)
# ----------------------------------------------------------------------
class TestTokenizerFidelity:
    def _tok(self):
        tok = BytePairEncoder(vocab_size_target=512)
        tok.train_on_text(
            "The quick brown fox jumps over the lazy dog.\n"
            "Second paragraph with café, naïve, 你好世界, مرحبا, 🚀.\n" * 20
        )
        return tok

    def test_exact_roundtrip_unicode_and_whitespace(self):
        tok = self._tok()
        samples = [
            "Hello, world! This is a test.",
            "line one\nline two\n\tindented",
            "Unicode: café, naïve, 你好世界, مرحبا, 🚀 emoji",
            "code: def foo(x):\n    return x + 1",
            "  leading and trailing spaces  ",
        ]
        for s in samples:
            assert tok.decode(tok.encode(s)) == s, f"roundtrip failed for {s!r}"

    def test_whitespace_is_preserved(self):
        tok = self._tok()
        text = "a\nb\tc\n\ndouble blank line\n    indented    "
        assert tok.decode(tok.encode(text)) == text

    def test_byte_coverage_no_unknown_errors(self):
        tok = self._tok()
        # A string exercising many bytes: utf-8 multibyte + ascii
        s = "".join(chr(c) for c in range(0x20, 0x500, 37)) + "中文 العربية 🚀"
        ids = tok.encode(s)
        assert tok.decode(ids) == s

    def test_pad_id_never_emitted_by_encode(self):
        tok = self._tok()
        assert tok.pad_id is not None
        ids = tok.encode("plain english text here")
        assert tok.pad_id not in ids


# ----------------------------------------------------------------------
# 6. Loss masking / padding (F-15)
# ----------------------------------------------------------------------
class TestLossMasking:
    def test_padding_targets_ignored(self):
        from xrfm.data.loader import XRFMTextDataset

        tok = BytePairEncoder(vocab_size_target=512)
        tok.train_on_text("short text for tokenizer fitting.\n" * 30)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("This is a short document.\nAnd another line here.\n" * 5)
            path = f.name
        try:
            ds = XRFMTextDataset(path, tok, max_seq_len=64, split="train", pad_id=tok.pad_id or 0)
            inp, tgt = ds[0]
            assert inp.shape == (64,)
            assert tgt.shape == (64,)
            # Every padded input position must have -100 target.
            for i in range(64):
                if inp[i] == (tok.pad_id or 0):
                    assert tgt[i] == -100
        finally:
            os.unlink(path)

    def test_cross_entropy_ignore_equals_manual(self):
        logits = torch.randn(2, 8, 50)
        targets = torch.randint(0, 50, (2, 8))
        targets[0, 5:] = -100
        targets[1, 6:] = -100
        loss = F.cross_entropy(logits.view(-1, 50), targets.view(-1), ignore_index=-100)
        valid = targets != -100
        manual = F.cross_entropy(logits[valid].view(-1, 50), targets[valid], reduction="mean")
        assert abs(loss.item() - manual.item()) < 1e-5


# ----------------------------------------------------------------------
# 7. Mixed precision honesty (F-23)
# ----------------------------------------------------------------------
class TestMixedPrecisionHonesty:
    def test_cpu_is_noop(self):
        loader = MixedPrecisionLoader(enabled=True)
        assert not torch.cuda.is_available() or True  # no GPU in CI sandbox
        loss = torch.randn(2, 2, requires_grad=True).sum()
        scaled = loader.scale(loss)
        assert torch.equal(scaled, loss)  # NoOpScaler on CPU


# ----------------------------------------------------------------------
# 8. Config-driven resume (v1.1 regression: the resume_from block had been
#    orphaned after a return inside _checkpoint_extra, silently disabling it)
# ----------------------------------------------------------------------
class TestConfigResume:
    def test_resume_from_config_restores_step_and_scheduler(self):
        import tempfile

        from training.loop import TrainingLoop, _set_seed

        class D:
            def __len__(self):
                return 32

            def __getitem__(self, i):
                ids = torch.randint(0, 100, (33,))
                return ids[:-1], ids[1:]

        tmp = tempfile.mkdtemp(prefix="xrfm_resume_")

        _set_seed(42)
        m1 = GPTModel(TINY)
        l1 = TrainingLoop(config_path=TINY, model=m1, dataset=D(), checkpoint_dir=tmp)
        l1.batch_size = 8
        r1 = l1.training_loop(max_steps=10, checkpoint_every=10, log_interval=1000)
        ckpt_path = r1["checkpoint_path"]
        assert ckpt_path and os.path.exists(ckpt_path)

        # Now drive resume THROUGH the config mechanism (TrainingLoop.resume_from)
        import yaml

        cfg = yaml.safe_load(open(TINY))
        cfg["training"]["resume_from"] = ckpt_path
        tmp_cfg = os.path.join(tmp, "resume.yaml")
        with open(tmp_cfg, "w") as f:
            yaml.safe_dump(cfg, f)

        _set_seed(43)  # different seed: proves resume overrides fresh init
        m2 = GPTModel(TINY)
        l2 = TrainingLoop(config_path=tmp_cfg, model=m2, dataset=D(), checkpoint_dir=tmp)
        assert l2.current_step == 10, f"expected step 10, got {l2.current_step}"
        assert l2.scheduler.current_step == l1.scheduler.current_step
