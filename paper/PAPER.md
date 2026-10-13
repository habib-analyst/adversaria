# adversaria: Calibration-Tracked Debate — votes weighted by proven calibration, arguments scored for novelty

**Habib Ur Rehman** · 2026-10-12 · github.com/habib-analyst/adversaria

## Abstract

Multi-agent debate (parallel proposals + voting) is a standard pattern, but every
open implementation we surveyed weights votes by *stated confidence* or one-agent-one-vote.
Both fail against a **loud-but-wrong agent**: an agent that states 0.95 confidence
whether right or wrong hijacks confidence-weighted votes, and one-agent-one-vote
lets it drag majorities. adversaria's invention is twofold: (1) **calibration-tracked
voting** — each agent carries a persistent track record of (stated confidence, outcome)
pairs, and its vote is weighted by the *recalibrated* probability
P(correct | stated confidence) learned from that record, which pushes
overconfident agents down *and* corrects underconfident agents up; (2) an
**argument-novelty critic** that scores each proposal's rationale by TF-IDF
dissimilarity to earlier proposals and discounts near-repeats, so a parrot
restating the leader's angle cannot double-count it. On 500 seeded multi-choice
reasoning tasks with a 6-agent panel including an anti-calibrated loud agent,
a sycophantic parrot and an underconfident agent, full adversaria reaches
**0.962 accuracy vs 0.900 unweighted majority (+6.2pp) and 0.838 single-best
(+12.4pp)**. Naive confidence weighting (0.912) and Brier-exponential weighting
(0.842, ablation) both fall short of recalibrated voting (0.946).

## 1. Gap analysis

Role-assigned debate with voting is crowded: llm-council (opinion-giver/reviewer/
synthesizer/arbiter + 7 voting methods including weighted/confidence),
agentic-consensus-framework, MASLab's taxonomy (arXiv 2505.16988), helix ADR-008.
We found **nothing** that (a) weights votes by each agent's *measured past
calibration* from a persistent track record, or (b) scores proposals for
*argument novelty* rather than correctness. Confidence-weighted voting exists
(llm-council has it) — and our experiments show exactly why it is insufficient.

## 2. Method

**Track record.** Every debate outcome appends (confidence, correct∈{0,1}) per
agent. From the record we compute Brier score and ECE (diagnostics) and a
**reliability curve**: 10 confidence bins with Laplace-smoothed empirical
accuracy. The voting weight for a proposal is the curve's value at the stated
confidence: `w = P̂(correct | confidence)`.

**Why not Brier-exponential.** Weighting by `exp(−λ·Brier)` (the obvious first
idea) conflates miscalibration with uselessness: our underconfident agent
(Brier 0.22, accuracy 0.66) gets downweighted although its votes carry signal.
Recalibration instead *corrects* it upward. The ablation (§4) confirms:
Brier-exponential reaches only 0.842 vs 0.946 for recalibrated voting.

**Novelty critic.** Pure-numpy TF-IDF + cosine dissimilarity over rationales;
`novelty_i = 1 − max_{j<i} cos(r_i, r_j)`. Proposals below 0.25 novelty get
their weight halved. First proposal always scores 1.0.

**Debate protocol.** Fixed agent order → proposals → novelty scoring →
effective weight = recalibrated weight × novelty factor → argmax answer;
debate confidence = winner's weight share.

## 3. Experimental setup

- **Tasks:** 150 seeded synthetic multi-choice tasks/run (arithmetic, logic,
  sequences, word problems), 50 for calibration warmup, 100 held-out for eval,
  5 seeds (500 eval tasks total). Deterministic generator; no LLM calls, so
  the *protocol* is evaluated with ground truth.
- **Panel (6 simulated agents):** sage (skill 0.82, well-calibrated), steady
  (0.70, mildly overconfident), meek (0.66, underconfident), loud (0.55,
  **anti-calibrated**: 0.95 confidence when wrong, 0.60 when right), bluster
  (0.52, always 0.90), parrot (copies sage's answer, restates its rationale).
- **Methods:** single-best (best warmup agent), unweighted majority, naive
  confidence-weighted, Brier-exponential (λ=5, ablation), recalibrated
  (no novelty), full adversaria (recalibrated + novelty).

## 4. Results (mean ± std over 5 seeds, 100 tasks each)

| method | accuracy |
|---|---|
| single best agent | 0.838 ± 0.029 |
| unweighted majority | 0.900 ± 0.006 |
| naive confidence-weighted | 0.912 ± 0.008 |
| Brier-exponential (ablation) | 0.842 ± 0.028 |
| recalibrated voting (no novelty) | 0.946 ± 0.017 |
| **full adversaria** | **0.962 ± 0.010** |

Kill-gate: full ≥ majority + 5pp (6.2pp ✓) and full > single-best (+12.4pp ✓) → **PASS**.

**The loud agent, neutralised.** On tasks where loud is wrong (38–56 per 100),
majority vote is wrong 9–21% of the time; full adversaria only 2–8%.
Naive confidence weighting (0.912) looks fine on average but cannot fully
discount the anti-calibrated agent; recalibration does (−5.0pp gap closed).

**Novelty's contribution.** Full vs recalibrated-only: 0.962 vs 0.946 (+1.6pp,
consistent in 4/5 seeds). The parrot's mean novelty is 0.06 (caught); honest
note: when the repeated angle is *correct*, discounting it costs a little —
seed 0 shows full 0.95 vs recalibrated 0.96. The critic trades a small
correct-duplicate penalty for robustness against wrong-duplicate cascades.

## 5. Limitations (honest)

1. **Simulated agents, not LLMs.** Skill/calibration profiles are stipulated;
   real LLM agents' miscalibration patterns are messier. The protocol is
   validated; the agent models are illustrative.
2. **Synthetic tasks.** Four template families; no claim about transfer to
   open-ended reasoning.
3. **Novelty is lexical.** TF-IDF dissimilarity catches restatement, not
   semantic redundancy between differently-worded identical arguments.
4. **Cold start.** New agents begin at the smoothed prior (uniform); the
   track record needs ~30+ outcomes before weights differentiate (measured:
   warmup is 50 tasks).

## 6. What this is not

Not another debate framework with roles and a judge. The claim is narrow and
measured: *persistent per-agent calibration records + recalibrated vote
weights + a novelty critic beat majority and confidence-weighted voting when
the panel contains miscalibrated agents.*

## Reproduce

```
PYTHONPATH=src python experiments/run_eval.py 0 1 2 3 4   # ~3 s, CPU
python -m pytest tests/                                   # 23 tests
```

## Citation

```bibtex
@software{rehman2026adversaria,
  author = {Habib Ur Rehman},
  title = {adversaria: calibration-tracked multi-agent debate},
  year = {2026},
  url = {https://github.com/habib-analyst/adversaria}
}
```
