"""
XRFM-NeuroTopo — confidence calibration metrics (Phase 13).

Implements ECE, Brier score, selective accuracy-coverage curve, abstention rate,
and false-confidence rate, plus a token-probability confidence baseline for
comparison.
"""

from __future__ import annotations

import torch


def expected_calibration_error(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    """ECE: |accuracy - confidence| weighted by bin count.

    conf: (N,) predicted confidence in [0,1].
    correct: (N,) bool/0-1 whether the prediction was correct.
    """
    conf = conf.detach().float().cpu().clamp(0, 1)
    correct = correct.detach().float().cpu()
    # Equal-mass binning (quantiles) is robust when confidences are clustered.
    n = conf.numel()
    edges = torch.quantile(conf, torch.linspace(0, 1, n_bins + 1))
    edges[0] = -0.01  # include the lowest confidences
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean().item()
        c = conf[mask].mean().item()
        ece += (mask.sum().item() / n) * abs(acc - c)
    return ece


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Multi-class Brier score: mean ||p - onehot||^2."""
    probs = probs.detach().float().cpu()
    targets = targets.detach().cpu()
    onehot = torch.nn.functional.one_hot(targets, num_classes=probs.shape[-1]).float()
    return ((probs - onehot) ** 2).sum(dim=-1).mean().item()


def selective_accuracy_coverage(
    conf: torch.Tensor, correct: torch.Tensor, thresholds: int = 20
) -> tuple[list[float], list[float]]:
    """Return (coverage_list, accuracy_list) sweeping a confidence threshold.

    At each threshold, abstain on low-confidence examples; coverage is the
    fraction answered; accuracy is among answered.
    """
    conf = conf.detach().float().cpu()
    correct = correct.detach().float().cpu()
    covs, accs = [], []
    for tau in torch.linspace(0, 1, thresholds + 1):
        answered = conf >= tau.item()
        n_ans = int(answered.sum().item())
        covs.append(n_ans / max(conf.numel(), 1))
        accs.append(float(correct[answered].mean().item()) if n_ans > 0 else 1.0)
    # Sort by increasing coverage so the AUC integrates correctly.
    order = torch.argsort(torch.tensor(covs))
    covs = [covs[i] for i in order.tolist()]
    accs = [accs[i] for i in order.tolist()]
    return covs, accs


def abstention_rate(conf: torch.Tensor, threshold: float) -> float:
    return float((conf < threshold).float().mean().item())


def false_confidence_rate(
    conf: torch.Tensor, correct: torch.Tensor, threshold: float = 0.8
) -> float:
    """Fraction of high-confidence predictions that are wrong."""
    high = conf >= threshold
    if high.sum() == 0:
        return 0.0
    return float((~correct[high].bool()).float().mean().item())


def token_prob_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Baseline confidence from next-token max probability."""
    p = torch.softmax(logits.detach(), dim=-1)
    return p.max(dim=-1).values


def evaluate_calibration(
    conf: torch.Tensor, correct: torch.Tensor, probs: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    covs, accs = selective_accuracy_coverage(conf, correct)
    # AUCC: area under coverage-accuracy curve (higher is better).
    auc = float(torch.trapz(torch.tensor(accs), torch.tensor(covs)).item())
    return {
        "ece": expected_calibration_error(conf, correct),
        "brier": brier_score(probs, targets),
        "abstention@0.5": abstention_rate(conf, 0.5),
        "false_confidence@0.8": false_confidence_rate(conf, correct, 0.8),
        "selective_auc": auc,
        "mean_conf": float(conf.mean().item()),
        "accuracy": float(correct.float().mean().item()),
    }
