"""adversaria quickstart: a 3-agent debate with a loud-but-wrong agent.

Run:  PYTHONPATH=src python examples/quickstart.py
"""

from adversaria import (
    AgentProfile,
    CalibrationTracker,
    Debate,
    SimulatedAgent,
    Task,
)
import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)
    profiles = [
        AgentProfile(name="sage", skill=0.85,
                     conf_correct=(0.88, 0.07), conf_wrong=(0.52, 0.12)),
        AgentProfile(name="loud", skill=0.55,          # confident, usually wrong
                     conf_correct=(0.60, 0.08), conf_wrong=(0.95, 0.03)),
        AgentProfile(name="meek", skill=0.66,          # underconfident but useful
                     conf_correct=(0.55, 0.08), conf_wrong=(0.45, 0.08)),
    ]
    agents = [SimulatedAgent(p, np.random.default_rng(100 + i))
              for i, p in enumerate(profiles)]
    tracker = CalibrationTracker()

    # warm up track records on 30 synthetic tasks
    for i in range(30):
        t = Task(qid=f"w{i}", question=f"warmup {i}?",
                 options=["A) 1", "B) 2", "C) 3", "D) 4"],
                 answer=["A", "B", "C", "D"][i % 4])
        Debate(agents, tracker).run(t, "warmup work", record=True)

    print("track records after warmup:")
    for a, s in tracker.summary().items():
        print(f"  {a:6s} acc={s['accuracy']:.2f} brier={s['brier']:.3f}")

    task = Task(qid="demo", question="What is 17 + 25 - 8?",
                options=["A) 32", "B) 34", "C) 36", "D) 30"], answer="B")
    res = Debate(agents, tracker).run(task, "17+25-8=34", record=False)
    print("\nproposals:")
    for p, nov in zip(res.proposals, res.novelties):
        print(f"  {p.agent:6s} -> {p.answer} (conf {p.confidence:.2f}, "
              f"novelty {nov:.2f}, weight {res.eff_weights[p.agent]:.3f}): {p.rationale}")
    print(f"\ndebate winner: {res.winner} (truth: {task.answer}), "
          f"confidence {res.confidence:.2f}")


if __name__ == "__main__":
    main()
