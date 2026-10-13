"""The adversaria debate protocol.

One round per task:
1. Each agent proposes (answer, confidence, rationale) in a fixed order.
2. The novelty critic scores every rationale against earlier ones.
3. Proposals below the novelty threshold are discounted (repeated angles
   carry less weight even when correct).
4. Remaining weight is the agent's calibration weight (from its persistent
   track record) times the novelty factor.
5. The answer with the most effective weight wins; debate confidence is the
   winner's share of total effective weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents import Proposal, SimulatedAgent, Task
from .calibration import CalibrationTracker
from .novelty import NoveltyCritic


@dataclass
class DebateResult:
    winner: str
    confidence: float
    proposals: list[Proposal] = field(default_factory=list)
    novelties: list[float] = field(default_factory=list)
    cal_weights: dict[str, float] = field(default_factory=dict)
    eff_weights: dict[str, float] = field(default_factory=dict)
    vote_totals: dict[str, float] = field(default_factory=dict)


class Debate:
    def __init__(
        self,
        agents: list[SimulatedAgent],
        tracker: CalibrationTracker,
        lam: float = 5.0,
        novelty_threshold: float = 0.25,
        novelty_discount: float = 0.5,
        use_novelty: bool = True,
    ):
        if not agents:
            raise ValueError("need at least one agent")
        self.agents = agents
        self.tracker = tracker
        self.lam = lam
        self.novelty_threshold = novelty_threshold
        self.novelty_discount = novelty_discount
        self.use_novelty = use_novelty
        self.critic = NoveltyCritic()

    def run(self, task: Task, work: str, record: bool = True) -> DebateResult:
        proposals: list[Proposal] = []
        peer: dict[str, Proposal] = {}
        for agent in self.agents:
            prop = agent.propose(task, work, peer_proposals=peer)
            proposals.append(prop)
            peer[agent.name] = prop

        self.critic.reset()
        novelties = [self.critic.score(p.rationale) for p in proposals]

        # Calibration weight = P(correct | stated confidence) from the
        # agent's own track record (recalibrated, not raw confidence).
        names = [a.name for a in self.agents]
        raw_w = {p.agent: self.tracker.calibrated_prob(p.agent, p.confidence)
                 for p in proposals}
        tot = sum(raw_w.values())
        cal_w = {a: (raw_w[a] / tot if tot > 0 else 1.0 / len(names)) for a in names}
        # keep the Brier-exponential around for the ablation baseline
        self._brier_w = self.tracker.weights(lam=self.lam, agents=names)
        eff: dict[str, float] = {}
        for prop, nov in zip(proposals, novelties):
            w = cal_w.get(prop.agent, 0.0)
            if self.use_novelty and nov < self.novelty_threshold:
                w *= self.novelty_discount
            eff[prop.agent] = w

        totals: dict[str, float] = {}
        for prop in proposals:
            totals[prop.answer] = totals.get(prop.answer, 0.0) + eff[prop.agent]
        winner = max(sorted(totals), key=lambda a: totals[a])
        total_w = sum(totals.values())
        confidence = totals[winner] / total_w if total_w > 0 else 0.0

        if record:
            for prop in proposals:
                self.tracker.record(prop.agent, prop.confidence, prop.answer == task.answer)

        return DebateResult(
            winner=winner,
            confidence=float(confidence),
            proposals=proposals,
            novelties=novelties,
            cal_weights=cal_w,
            eff_weights=eff,
            vote_totals=totals,
        )
