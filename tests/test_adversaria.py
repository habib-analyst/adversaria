"""Tests for adversaria. All deterministic; no network; no LLM calls."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from adversaria import (  # noqa: E402
    AgentProfile,
    CalibrationTracker,
    Debate,
    NoveltyCritic,
    SimulatedAgent,
    Task,
    novelty_scores,
)


def _task(ans="B"):
    return Task(qid="t", question="q?",
                options=["A) 1", "B) 2", "C) 3", "D) 4"], answer=ans)


# ---------- novelty ----------

def test_novelty_identical_is_zero():
    r = ["the cat sat on the mat"] * 3
    s = novelty_scores(r)
    assert s[0] == 1.0
    assert s[1] == pytest.approx(0.0, abs=1e-9)
    assert s[2] == pytest.approx(0.0, abs=1e-9)


def test_novelty_orthogonal_is_high():
    r = ["alpha beta gamma delta", "one two three four", "red blue green yellow"]
    s = novelty_scores(r)
    assert all(v > 0.9 for v in s[1:])


def test_novelty_partial_repeat():
    base = "computing the sum step by step gives forty two"
    r = [base, base + " and therefore the answer holds", "completely unrelated zebra quantum"]
    s = novelty_scores(r)
    assert s[1] < 0.6  # near-repeat detected
    assert s[2] > 0.8  # genuinely new angle


def test_novelty_empty():
    assert novelty_scores([]) == []
    assert novelty_scores([""]) == [1.0]


def test_critic_stateful():
    c = NoveltyCritic()
    assert c.score("hello world") == 1.0
    assert c.score("hello world") == pytest.approx(0.0, abs=1e-9)
    c.reset()
    assert c.score("hello world") == 1.0


# ---------- calibration ----------

def test_brier_perfect_agent():
    t = CalibrationTracker()
    for _ in range(20):
        t.record("a", 1.0, True)
    assert t.brier("a") == pytest.approx(0.0)


def test_brier_worst_agent():
    t = CalibrationTracker()
    for _ in range(20):
        t.record("a", 1.0, False)
    assert t.brier("a") == pytest.approx(1.0)


def test_brier_unseen_agent_prior():
    t = CalibrationTracker()
    assert t.brier("ghost") == pytest.approx(0.25)


def test_weights_normalise_and_favour_calibrated():
    t = CalibrationTracker()
    rng = np.random.default_rng(0)
    for _ in range(50):
        t.record("good", float(rng.normal(0.85, 0.05)), rng.random() < 0.85)
        t.record("loud", 0.95, rng.random() < 0.55)
    w = t.weights(lam=5.0)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["good"] > w["loud"]


def test_weights_uniform_when_no_history():
    t = CalibrationTracker()
    assert t.weights() == {}


def test_weights_negative_lam_rejected():
    t = CalibrationTracker()
    t.record("a", 0.5, True)
    with pytest.raises(ValueError):
        t.weights(lam=-1.0)


def test_ece_well_calibrated_low():
    t = CalibrationTracker()
    rng = np.random.default_rng(1)
    for _ in range(500):
        p = rng.random()
        t.record("a", p, rng.random() < p)
    assert t.ece("a") < 0.08


# ---------- agents ----------

def test_skill_respected_statistically():
    rng = np.random.default_rng(7)
    agent = SimulatedAgent(AgentProfile(name="x", skill=0.8), rng)
    task = _task()
    hits = sum(agent.propose(task, "work").answer == "B" for _ in range(2000))
    assert 0.75 < hits / 2000 < 0.85


def test_parrot_copies_answer():
    rng = np.random.default_rng(3)
    sage = SimulatedAgent(AgentProfile(name="sage", skill=1.0), rng)
    parrot = SimulatedAgent(AgentProfile(name="parrot", skill=0.0, parrot_of="sage"), rng)
    task = _task()
    src = sage.propose(task, "work")
    got = parrot.propose(task, "work", peer_proposals={"sage": src})
    assert got.answer == src.answer
    assert got.rationale != src.rationale  # reworded, not identical


def test_parrot_without_peer_falls_back():
    rng = np.random.default_rng(3)
    parrot = SimulatedAgent(AgentProfile(name="parrot", skill=1.0, parrot_of="sage"), rng)
    got = parrot.propose(_task(), "work")
    assert got.answer == "B"


def test_confidence_clipped():
    rng = np.random.default_rng(3)
    a = SimulatedAgent(AgentProfile(name="x", skill=0.5,
                                    conf_correct=(5.0, 5.0), conf_wrong=(-5.0, 5.0)), rng)
    for _ in range(100):
        p = a.propose(_task(), "work")
        assert 0.0 <= p.confidence <= 1.0


def test_bad_skill_rejected():
    with pytest.raises(ValueError):
        AgentProfile(name="x", skill=1.5)


# ---------- debate ----------

def _debate_agents(seed=0):
    rng = np.random.default_rng(seed)
    profiles = [
        AgentProfile(name="a1", skill=1.0, conf_correct=(0.9, 0.01), style="direct"),
        AgentProfile(name="a2", skill=1.0, conf_correct=(0.9, 0.01), style="elimination"),
    ]
    return [SimulatedAgent(p, np.random.default_rng(seed + i)) for i, p in enumerate(profiles)]


def test_debate_unanimous_winner():
    agents = _debate_agents()
    tracker = CalibrationTracker()
    d = Debate(agents, tracker)
    res = d.run(_task("B"), "work")
    assert res.winner == "B"
    assert res.confidence == pytest.approx(1.0)


def test_debate_records_track_record():
    agents = _debate_agents()
    tracker = CalibrationTracker()
    d = Debate(agents, tracker)
    d.run(_task("B"), "work", record=True)
    assert tracker.n("a1") == 1
    assert tracker.accuracy("a1") == 1.0


def test_debate_no_record_when_disabled():
    agents = _debate_agents()
    tracker = CalibrationTracker()
    d = Debate(agents, tracker)
    d.run(_task("B"), "work", record=False)
    assert tracker.n("a1") == 0


def test_novelty_discount_applies():
    # parrot repeats sage -> its novelty is low -> its effective weight shrinks
    rng = np.random.default_rng(0)
    agents = [
        SimulatedAgent(AgentProfile(name="sage", skill=1.0, style="direct"),
                       np.random.default_rng(0)),
        SimulatedAgent(AgentProfile(name="parrot", skill=0.0, parrot_of="sage"),
                       np.random.default_rng(1)),
    ]
    tracker = CalibrationTracker()
    for _ in range(30):
        tracker.record("sage", 0.9, True)
        tracker.record("parrot", 0.8, True)
    d = Debate(agents, tracker, lam=5.0, use_novelty=True)
    res = d.run(_task("B"), "work", record=False)
    sage_w = res.eff_weights["sage"]
    parrot_w = res.eff_weights["parrot"]
    assert res.novelties[1] < 0.25
    assert parrot_w < sage_w * 0.75  # discounted


def test_debate_needs_agents():
    with pytest.raises(ValueError):
        Debate([], CalibrationTracker())


def test_result_fields_populated():
    agents = _debate_agents()
    d = Debate(agents, CalibrationTracker())
    res = d.run(_task("C"), "work")
    assert len(res.proposals) == 2
    assert len(res.novelties) == 2
    assert set(res.cal_weights) == {"a1", "a2"}
    assert abs(sum(res.vote_totals.values()) - sum(res.eff_weights.values())) < 1e-9
