[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen.svg)]()

# adversaria

**Multi-agent debate where votes are weighted by proven calibration — not volume, not loudness.**

Every debate framework lets agents vote. None track whether an agent's confidence means anything. adversaria gives each agent a persistent track record of (stated confidence, actual outcome) pairs and weights its vote by the *recalibrated* probability it is right — so a loud-but-wrong agent (0.95 confidence, 55% accuracy) is automatically downweighted, and an underconfident-but-useful agent is corrected *up*. A novelty critic separately discounts proposals that merely restate earlier arguments.

## Quickstart

```bash
pip install adversaria
```

```python
from adversaria import AgentProfile, CalibrationTracker, Debate, SimulatedAgent, Task
import numpy as np

agents = [SimulatedAgent(p, np.random.default_rng(100 + i)) for i, p in enumerate([
    AgentProfile(name="sage", skill=0.85,
                 conf_correct=(0.88, 0.07), conf_wrong=(0.52, 0.12)),
    AgentProfile(name="loud", skill=0.55,   # confident, usually wrong
                 conf_correct=(0.60, 0.08), conf_wrong=(0.95, 0.03)),
    AgentProfile(name="meek", skill=0.66,   # underconfident but useful
                 conf_correct=(0.55, 0.08), conf_wrong=(0.45, 0.08)),
])]
tracker = CalibrationTracker()
task = Task(qid="demo", question="What is 17 + 25 - 8?",
            options=["A) 32", "B) 34", "C) 36", "D) 30"], answer="B")
res = Debate(agents, tracker).run(task, "17+25-8=34")
print(res.winner, f"{res.confidence:.2f}")   # B 1.00
print(res.eff_weights)  # loud downweighted by its own track record
```

Expected output (after a 30-task warmup; exact numbers vary by seed):

```
track records after warmup:
  sage   acc=0.80 brier=0.076
  loud   acc=0.57 brier=0.488
  meek   acc=0.67 brier=0.210
debate winner: B (truth: B), confidence 1.00
```

Or run the bundled demo: `PYTHONPATH=src python examples/quickstart.py`

## Measured results

500 seeded multi-choice reasoning tasks, 6-agent panel (well-calibrated expert, overconfident, underconfident, **anti-calibrated loud agent**, blusterer, parrot). 5 seeds × 100 held-out tasks.

| method | accuracy |
|---|---|
| single best agent | 0.838 ± 0.029 |
| unweighted majority | 0.900 ± 0.006 |
| naive confidence-weighted | 0.912 ± 0.008 |
| Brier-exponential weighting (ablation) | 0.842 ± 0.028 |
| recalibrated voting (no novelty) | 0.946 ± 0.017 |
| **full adversaria** | **0.962 ± 0.010** |

On tasks where the loud agent is wrong, majority vote fails 9–21% of the time; adversaria fails 2–8%. Full paper with gap analysis, ablations and honest limitations: [`paper/PAPER.md`](paper/PAPER.md).

## How it works

```
proposals → novelty critic (TF-IDF dissimilarity vs earlier proposals)
          → calibration tracker (reliability curve per agent)
          → weight = P̂(correct | stated confidence) × novelty factor
          → argmax answer, confidence = winner's weight share
```

Two moving parts, both in `src/adversaria/`:
- `calibration.py` — persistent track records, Brier/ECE diagnostics, Laplace-smoothed reliability curves, recalibrated vote weights.
- `novelty.py` — pure-numpy TF-IDF novelty critic (zero heavy dependencies).
- `debate.py` — the protocol; `agents.py` — simulated agents with known skill/calibration profiles for honest evaluation.

## Citation

```bibtex
@software{rehman2026adversaria,
  author = {Habib Ur Rehman},
  title = {adversaria: calibration-tracked multi-agent debate},
  year = {2026},
  url = {https://github.com/habib-analyst/adversaria}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
