"""Persistent calibration tracking and calibration-weighted voting.

Each agent accumulates a track record of (confidence, was_correct) pairs.
Calibration is measured with the Brier score (mean squared error between
stated confidence and the 0/1 outcome). Votes are weighted by
``exp(-lambda * brier)``: well-calibrated agents dominate, miscalibrated
loudmouths are downweighted automatically — no hand-tuning per agent.
"""

from __future__ import annotations

import numpy as np


class CalibrationTracker:
    """Track record of stated confidences vs outcomes per agent."""

    def __init__(self) -> None:
        self._history: dict[str, list[tuple[float, int]]] = {}

    def record(self, agent: str, confidence: float, correct: bool) -> None:
        self._history.setdefault(agent, []).append(
            (float(confidence), 1 if correct else 0)
        )

    def n(self, agent: str) -> int:
        return len(self._history.get(agent, []))

    def agents(self) -> list[str]:
        return list(self._history.keys())

    def brier(self, agent: str) -> float:
        """Mean squared error between confidence and outcome. Lower is better."""
        hist = self._history.get(agent, [])
        if not hist:
            return 0.25  # maximum-uncertainty prior for unseen agents
        conf = np.array([c for c, _ in hist])
        outcome = np.array([o for _, o in hist])
        return float(np.mean((conf - outcome) ** 2))

    def ece(self, agent: str, n_bins: int = 10) -> float:
        """Expected calibration error. Lower is better."""
        hist = self._history.get(agent, [])
        if not hist:
            return 0.0
        conf = np.array([c for c, _ in hist])
        outcome = np.array([o for _, o in hist])
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf <= hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(conf)) * abs(conf[mask].mean() - outcome[mask].mean())
        return float(ece)

    def accuracy(self, agent: str) -> float:
        hist = self._history.get(agent, [])
        if not hist:
            return 0.0
        return float(np.mean([o for _, o in hist]))

    def reliability(self, agent: str, n_bins: int = 10, alpha: float = 1.0) -> list[tuple[float, float, int]]:
        """Reliability curve: per confidence bin, (bin_center, empirical_accuracy, n).

        Laplace smoothing (alpha) keeps small bins near 0.5 instead of 0/1.
        """
        hist = self._history.get(agent, [])
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        conf = np.array([c for c, _ in hist]) if hist else np.array([])
        outcome = np.array([o for _, o in hist]) if hist else np.array([])
        curve = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if len(conf) == 0:
                mask = np.array([], dtype=bool)
            elif lo == 0:
                mask = conf <= hi
            else:
                mask = (conf > lo) & (conf <= hi)
            n = int(mask.sum())
            hits = float(outcome[mask].sum()) if n else 0.0
            acc = (hits + alpha) / (n + 2 * alpha)  # Laplace-smoothed
            curve.append(((lo + hi) / 2.0, acc, n))
        return curve

    def calibrated_prob(self, agent: str, confidence: float, n_bins: int = 10) -> float:
        """P(correct | this agent stated this confidence), from its track record.

        This is the voting weight: it corrects overconfidence DOWN (loud:
        0.95 stated -> ~0.40 empirical) and underconfidence UP (meek: 0.55
        stated -> ~0.66 empirical). Unseen agents / empty bins fall back to
        the smoothed prior, so cold debates start uniform.
        """
        curve = self.reliability(agent, n_bins=n_bins)
        idx = min(n_bins - 1, max(0, int(confidence * n_bins)))
        return float(curve[idx][1])

    def weights(self, lam: float = 5.0, agents: list[str] | None = None) -> dict[str, float]:
        """Brier-exponential weights (diagnostic / ablation baseline).

        The live debate uses calibrated_prob(); this remains as the naive
        calibration-weighting baseline for comparison.
        """
        if lam < 0:
            raise ValueError("lam must be non-negative")
        names = list(agents) if agents is not None else list(self._history)
        raw = {a: float(np.exp(-lam * self.brier(a))) for a in names}
        total = sum(raw.values())
        if total <= 0 or not raw:
            return {}
        return {a: v / total for a, v in raw.items()}

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            a: {
                "n": float(self.n(a)),
                "accuracy": self.accuracy(a),
                "brier": self.brier(a),
                "ece": self.ece(a),
            }
            for a in self._history
        }
