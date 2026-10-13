"""Simulated debate agents with known skill / calibration profiles.

Real LLM agents are replaced by simulators so the debate protocol itself can
be evaluated with ground truth: each profile has a true skill P(correct) and
a confidence behaviour (how confidence relates to being right). This lets us
measure whether calibration-weighted voting actually neutralises a
loud-but-wrong agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Task:
    qid: str
    question: str
    options: list  # list[str], e.g. ["A) ...", ...]
    answer: str  # the correct option letter, e.g. "B"


@dataclass(frozen=True)
class Proposal:
    agent: str
    answer: str
    confidence: float  # in [0, 1]
    rationale: str


@dataclass
class AgentProfile:
    """Behavioural profile of a simulated agent."""

    name: str
    skill: float  # P(answer correct), in [0, 1]
    conf_correct: tuple = (0.85, 0.08)  # (mean, std) confidence when right
    conf_wrong: tuple = (0.60, 0.12)  # (mean, std) confidence when wrong
    style: str = "direct"  # rationale template style
    parrot_of: str | None = None  # name of agent whose answer this one copies

    def __post_init__(self) -> None:
        if not 0.0 <= self.skill <= 1.0:
            raise ValueError("skill must be in [0, 1]")


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


_RATIONALE_TEMPLATES = {
    "direct": [
        "Computing step by step, {work} leads to option {ans}.",
        "Working through it: {work}, so the answer is {ans}.",
        "The calculation {work} gives {ans} directly.",
    ],
    "elimination": [
        "Ruling out the distractors leaves {ans}: {work}.",
        "Options other than {ans} fail because {work}; answer {ans}.",
        "By elimination, {ans} is the only consistent choice ({work}).",
    ],
    "analogy": [
        "This is like the standard pattern where {work}; hence {ans}.",
        "Same structure as the textbook case ({work}), so {ans}.",
    ],
    "terse": [
        "{work} -> {ans}.",
        "Answer {ans} ({work}).",
    ],
}


def _reword(text: str) -> str:
    """Near-verbatim copy with a discourse marker — the classic parrot move.

    Keeps every content word identical so the argument is truly repeated;
    only the framing changes. This is exactly what the novelty critic must
    catch: same angle, restated.
    """
    text = text.strip()
    return "In other words, " + text[0].lower() + text[1:]


class SimulatedAgent:
    """An agent with a fixed skill and confidence behaviour."""

    def __init__(self, profile: AgentProfile, rng: np.random.Generator):
        self.profile = profile
        self.rng = rng

    @property
    def name(self) -> str:
        return self.profile.name

    def _draw_confidence(self, correct: bool) -> float:
        mean, std = self.profile.conf_correct if correct else self.profile.conf_wrong
        return _clip01(float(self.rng.normal(mean, std)))

    def _rationale(self, task: Task, answer: str, work: str) -> str:
        templates = _RATIONALE_TEMPLATES[self.profile.style]
        t = templates[int(self.rng.integers(len(templates)))]
        return t.format(work=work, ans=answer)

    def propose(self, task: Task, work: str, peer_proposals: dict | None = None) -> Proposal:
        """Produce a proposal. Parrots copy another agent's answer (reworded)."""
        p = self.profile
        if p.parrot_of is not None:
            src = (peer_proposals or {}).get(p.parrot_of)
            if src is not None:
                return Proposal(
                    agent=p.name,
                    answer=src.answer,
                    confidence=_clip01(float(self.rng.normal(0.80, 0.08))),
                    rationale=_reword(src.rationale),
                )
        correct = bool(self.rng.random() < p.skill)
        if correct:
            answer = task.answer
        else:
            wrong = [o for o in ("A", "B", "C", "D") if o != task.answer]
            answer = str(self.rng.choice(wrong))
        return Proposal(
            agent=p.name,
            answer=answer,
            confidence=self._draw_confidence(correct),
            rationale=self._rationale(task, answer, work),
        )
