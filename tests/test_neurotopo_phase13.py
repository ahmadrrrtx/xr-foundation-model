"""Phase 13 tests: calibration metrics + token-prob baseline comparison."""

from __future__ import annotations

import torch

from xrfm.nt_evaluation import (
    abstention_rate,
    brier_score,
    evaluate_calibration,
    expected_calibration_error,
    false_confidence_rate,
    selective_accuracy_coverage,
    token_prob_confidence,
)


def test_ece_perfect_and_worst():
    conf = torch.tensor([0.9, 0.9, 0.1, 0.1])
    correct = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(conf, correct) < 0.15
    # Perfect confidence, always wrong -> high ECE
    bad = torch.tensor([0.99, 0.99])
    bad_correct = torch.tensor([0.0, 0.0])
    assert expected_calibration_error(bad, bad_correct) > 0.5


def test_brier_score():
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    assert brier_score(probs, targets) < 1e-6


def test_selective_coverage_monotonic():
    torch.manual_seed(0)
    conf = torch.rand(100)
    correct = (torch.rand(100) > 0.3).float()
    covs, accs = selective_accuracy_coverage(conf, correct, thresholds=10)
    # Returned sorted by increasing coverage (for AUC integration).
    assert covs[0] <= covs[-1]
    assert 0.0 <= covs[0] and covs[-1] >= 0.99
    assert all(0 <= c <= 1 for c in covs)


def test_abstention_and_false_confidence():
    conf = torch.tensor([0.9, 0.2, 0.9, 0.1])
    correct = torch.tensor([1.0, 0.0, 0.0, 1.0])
    assert abstention_rate(conf, 0.5) == 0.5
    # Two high-confidence predictions, one wrong -> 0.5 FCR
    assert false_confidence_rate(conf, correct, 0.8) == 0.5


def test_token_prob_baseline():
    logits = torch.randn(4, 50)
    conf = token_prob_confidence(logits)
    assert conf.shape == (4,)
    assert (conf > 0).all() and (conf <= 1).all()


def test_graph_confidence_beats_token_prob_on_synthetic():
    """GATE: on a synthetic set where graph evidence separates known from
    unknown, a graph-evidence confidence must give a better selective-AUC than
    the token-probability baseline (which is uninformative here)."""
    torch.manual_seed(0)
    from xrfm.uncertainty import UncertaintyHead

    head = UncertaintyHead()
    opt = torch.optim.Adam(head.parameters(), lr=3e-2)

    def sample(label):
        v = torch.randn(9) * 0.3
        if label == 0:
            v[0] += 1.0; v[2] -= 1.0; v[8] += 1.0
        else:
            v[0] -= 1.0; v[7] += 1.0
        return v

    for _ in range(400):
        y = torch.randint(0, 2, (64,))
        X = torch.stack([sample(int(i)) for i in y])
        logits, _ = head(X)
        loss = torch.nn.functional.cross_entropy(logits[:, [0, 2]], y)
        opt.zero_grad(); loss.backward(); opt.step()

    # Eval: label 0 = known/correct, label 1 = unknown/abstain (incorrect if answered)
    head.eval()
    N = 1000
    y = torch.randint(0, 2, (N,))
    X = torch.stack([sample(int(i)) for i in y])
    with torch.no_grad():
        logits, _ = head(X)
        p = torch.softmax(logits, dim=-1)
        graph_conf = p[:, 0]   # confidence in ANSWER
        correct = (y == 0)
        # token-prob baseline: uninformative random confidence
        torch.manual_seed(1)
        token_conf = torch.rand(N)

    g_metrics = evaluate_calibration(graph_conf, correct, p, y.clamp_max(p.shape[1] - 1))
    t_metrics = evaluate_calibration(token_conf, correct,
                                     torch.softmax(torch.randn(N, 5), -1), y.clamp_max(4))
    assert g_metrics["selective_auc"] > t_metrics["selective_auc"], (
        g_metrics["selective_auc"], t_metrics["selective_auc"]
    )


def test_evaluate_calibration_returns_all_keys():
    conf = torch.rand(50)
    correct = (torch.rand(50) > 0.4).float()
    probs = torch.softmax(torch.randn(50, 5), -1)
    targets = torch.randint(0, 5, (50,))
    m = evaluate_calibration(conf, correct, probs, targets)
    for k in ["ece", "brier", "abstention@0.5", "false_confidence@0.8",
              "selective_auc", "mean_conf", "accuracy"]:
        assert k in m
