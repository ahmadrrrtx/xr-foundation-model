"""Phase 12 tests: 5-state uncertainty head.

GATE: the head must distinguish known / unknown / ambiguous / insufficient-context
/ conflicting-evidence better than random on synthetic examples.
"""

from __future__ import annotations

import torch

from xrfm.uncertainty import STATE_NAMES, UncertaintyHead, effective_rank, routing_entropy


def test_state_names_and_shapes():
    head = UncertaintyHead()
    ev = torch.randn(4, 9)
    logits, calib = head(ev)
    assert logits.shape == (4, 5)
    assert calib.shape == (4,)
    assert STATE_NAMES == ("ANSWER", "QUALIFY", "ABSTAIN", "REQUEST_CONTEXT", "REQUEST_EVIDENCE")


def test_effective_rank_bounds():
    H = torch.eye(8).unsqueeze(0).expand(3, -1, -1)
    r = effective_rank(H)
    assert (r > 0).all() and (r <= 1.0001).all()
    H_low = torch.randn(3, 8, 1) @ torch.randn(3, 1, 8)
    assert effective_rank(H_low).mean() < r.mean()


def test_routing_entropy_bounds():
    ei = torch.tensor([[0, 0, 1, 1], [1, 2, 0, 2]])
    w = torch.full((2, 4), 0.5)
    ent = routing_entropy(w, ei, 2)
    assert (ent >= 0).all() and (ent <= 1.0001).all()


def test_evidence_vector_uses_graph_and_memory():
    head = UncertaintyHead()
    H = torch.randn(2, 8, 8)
    ei = torch.randint(0, 8, (2, 20))
    w = torch.sigmoid(torch.randn(2, 20))
    mi = {"surprise": torch.rand(2), "margin": torch.rand(2)}
    ev = head.evidence(H, w, ei, mi)
    assert ev.shape == (2, 9)
    assert torch.isfinite(ev).all()


def test_gate_distinguishes_states_better_than_random():
    """Synthetic 5-way evidence classification. We construct evidence vectors
    characteristic of each state and train the head; it must exceed chance (0.2)."""
    torch.manual_seed(0)
    head = UncertaintyHead()
    opt = torch.optim.Adam(head.parameters(), lr=3e-2)

    def sample(label):
        # Build a 9-dim evidence vector with class-conditional signal.
        v = torch.randn(9) * 0.3
        if label == 0:   # ANSWER: high support, low conflict, high margin
            v[0] += 1.0; v[2] -= 1.0; v[8] += 1.0
        elif label == 1: # QUALIFY: moderate conflict
            v[2] += 0.5; v[0] += 0.3
        elif label == 2: # ABSTAIN: low support, high surprise
            v[0] -= 1.0; v[7] += 1.0
        elif label == 3: # REQUEST_CONTEXT: low margin, low edge mass
            v[0] -= 0.5; v[8] -= 1.0
        else:            # REQUEST_EVIDENCE: high conflict
            v[2] += 1.5; v[3] -= 0.5
        return v

    for _ in range(400):
        labels = torch.randint(0, 5, (64,))
        X = torch.stack([sample(int(l)) for l in labels])
        logits, _ = head(X)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    with torch.no_grad():
        labels = torch.randint(0, 5, (500,))
        X = torch.stack([sample(int(l)) for l in labels])
        logits, _ = head(X)
        acc = (logits.argmax(-1) == labels).float().mean().item()
    assert acc > 0.6, f"uncertainty head did not separate states: {acc:.2f}"


def test_decide_threshold():
    head = UncertaintyHead()
    ev = torch.randn(3, 9)
    logits, _ = head(ev)
    decisions = head.decide(logits, threshold=0.5)
    assert all(d in STATE_NAMES + ("QUALIFY",) for d in decisions)
