"""Main experiment: does calibration-tracked debate beat the baselines?

Protocol (per seed):
  1. Generate 150 tasks (seed S). First 50 = calibration phase, last 100 = eval.
  2. Calibration phase: run debates (recording outcomes) to build each
     agent's track record. Tune lambda (calibration-weight sharpness) on the
     calibration-phase tasks via grid search.
  3. Eval phase: freeze the tracker, run all methods on 100 fresh tasks:
       - single_best: the agent with best calibration-phase accuracy
       - majority: unweighted majority vote of the 5 proposals
       - conf_weighted: naive confidence-weighted vote (the trap the loud
         agent exploits)
       - cal_weighted: calibration-weighted vote (no novelty filter)
       - full: calibration-weighted + novelty discount (adversaria)
  4. Report mean +/- std accuracy over seeds.

Kill-gate: full must beat unweighted majority by >= 5pp AND beat
single_best, else the invention is killed honestly.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")

from adversaria import (  # noqa: E402
    AgentProfile,
    CalibrationTracker,
    Debate,
    SimulatedAgent,
    Task,
)
from gen_tasks import generate  # noqa: E402

PROFILES = [
    # Well-calibrated expert: confidence tracks correctness.
    AgentProfile(name="sage", skill=0.82,
                 conf_correct=(0.88, 0.07), conf_wrong=(0.52, 0.12), style="direct"),
    # Decent but slightly overconfident when wrong.
    AgentProfile(name="steady", skill=0.70,
                 conf_correct=(0.84, 0.08), conf_wrong=(0.72, 0.10), style="elimination"),
    # Underconfident: right more often than its confidence suggests.
    AgentProfile(name="meek", skill=0.66,
                 conf_correct=(0.55, 0.08), conf_wrong=(0.45, 0.08), style="terse"),
    # ANTI-CALIBRATED stress agent: loudest exactly when wrong.
    # This is the realistic failure mode naive confidence-weighting cannot survive.
    AgentProfile(name="loud", skill=0.55,
                 conf_correct=(0.60, 0.08), conf_wrong=(0.95, 0.03), style="analogy"),
    # Near-random but always confident.
    AgentProfile(name="bluster", skill=0.52,
                 conf_correct=(0.90, 0.05), conf_wrong=(0.90, 0.05), style="direct"),
    # Sycophant: copies sage's answer, restates its rationale (low novelty).
    AgentProfile(name="parrot", skill=0.0, parrot_of="sage", style="direct"),
]


def _majority(proposals) -> str:
    counts: dict[str, int] = {}
    for p in proposals:
        counts[p.answer] = counts.get(p.answer, 0) + 1
    return max(sorted(counts), key=lambda a: counts[a])


def _conf_weighted(proposals) -> str:
    totals: dict[str, float] = {}
    for p in proposals:
        totals[p.answer] = totals.get(p.answer, 0.0) + p.confidence
    return max(sorted(totals), key=lambda a: totals[a])


def run_seed(seed: int, n_calib: int = 50, n_eval: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    pairs = generate(n_calib + n_eval, seed)
    calib_pairs = pairs[:n_calib]
    eval_pairs = pairs[n_calib:]

    agents = [SimulatedAgent(p, np.random.default_rng(seed * 100 + i))
              for i, p in enumerate(PROFILES)]

    # --- calibration phase: build track records ---
    tracker = CalibrationTracker()
    debate = Debate(agents, tracker)
    calib_correct = {a.name: 0 for a in agents}
    for task, work in calib_pairs:
        res = debate.run(task, work, record=True)
        for a in agents:
            prop = next(p for p in res.proposals if p.agent == a.name)
            calib_correct[a.name] += prop.answer == task.answer
    best_agent = max(calib_correct, key=lambda a: calib_correct[a])

    # --- eval phase: frozen tracker, all methods ---
    methods = ["single_best", "majority", "conf_weighted",
               "brier_weighted", "cal_weighted", "full"]
    hits = {m: 0 for m in methods}
    loud_drag = {"majority_wrong_when_loud_wrong": 0, "n_loud_wrong": 0,
                 "full_wrong_when_loud_wrong": 0}
    novelty_vals: list[float] = []
    for task, work in eval_pairs:
        # one shared proposal set per task for fair comparison
        proposals, peer = [], {}
        for agent in agents:
            prop = agent.propose(task, work, peer_proposals=peer)
            proposals.append(prop)
            peer[agent.name] = prop

        sb = next(p for p in proposals if p.agent == best_agent)
        hits["single_best"] += sb.answer == task.answer
        hits["majority"] += _majority(proposals) == task.answer
        hits["conf_weighted"] += _conf_weighted(proposals) == task.answer

        # Brier-exponential ablation (naive calibration weighting)
        bw = tracker.weights(lam=5.0, agents=[a.name for a in agents])
        btotals: dict[str, float] = {}
        for p in proposals:
            btotals[p.answer] = btotals.get(p.answer, 0.0) + bw[p.agent]
        hits["brier_weighted"] += (
            max(sorted(btotals), key=lambda a: btotals[a]) == task.answer)

        for use_nov, key in ((False, "cal_weighted"), (True, "full")):
            d = Debate(agents, tracker, use_novelty=use_nov)
            res = d.run(task, work, record=False)
            hits[key] += res.winner == task.answer
            if use_nov:
                novelty_vals.extend(res.novelties)

        loud_prop = next(p for p in proposals if p.agent == "loud")
        if loud_prop.answer != task.answer:
            loud_drag["n_loud_wrong"] += 1
            loud_drag["majority_wrong_when_loud_wrong"] += (
                _majority(proposals) != task.answer
            )
            d = Debate(agents, tracker, use_novelty=True)
            loud_drag["full_wrong_when_loud_wrong"] += (
                d.run(task, work, record=False).winner != task.answer
            )

    acc = {m: hits[m] / n_eval for m in methods}
    return {
        "seed": seed,
        "best_agent": best_agent,
        "calib_acc": {a.name: calib_correct[a.name] / n_calib for a in agents},
        "tracker": tracker.summary(),
        "acc": acc,
        "loud_drag": loud_drag,
        "mean_novelty": float(np.mean(novelty_vals)),
    }


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
    results = [run_seed(s) for s in seeds]
    methods = ["single_best", "majority", "conf_weighted",
               "brier_weighted", "cal_weighted", "full"]
    table: dict[str, dict] = {}
    for m in methods:
        vals = [r["acc"][m] for r in results]
        table[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "vals": vals}
    full = table["full"]["mean"]
    maj = table["majority"]["mean"]
    sb = table["single_best"]["mean"]
    gate = (full + 1e-9 >= maj + 0.05) and (full > sb + 1e-9)
    out = {
        "seeds": seeds,
        "methods": table,
        "kill_gate": {
            "full_minus_majority_pp": round((full - maj) * 100, 2),
            "full_minus_single_best_pp": round((full - sb) * 100, 2),
            "verdict": "PASS" if gate else "FAIL",
        },
        "per_seed": results,
    }
    print(json.dumps(out, indent=1))
    with open("results/summary.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
