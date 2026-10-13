"""Deterministic synthetic multi-choice reasoning task generator.

Four task families (arithmetic, logic, sequences, word problems), each with
a correct answer and three plausible distractors. Fully seeded: the same
seed always yields the same task set, so experiments are reproducible.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from adversaria import Task


def _opts(correct: int, distractors: list[int], rng) -> tuple[list[str], str]:
    seen = {correct}
    picks: list[int] = []
    for d in distractors:
        if d != correct and d not in seen:
            seen.add(d)
            picks.append(d)
        if len(picks) == 3:
            break
    letters = ["A", "B", "C", "D"]
    order = [correct] + picks
    perm = list(range(4))
    rng.shuffle(perm)
    ordered = [order[j] for j in perm]
    options = [f"{letters[k]}) {v}" for k, v in enumerate(ordered)]
    return options, letters[ordered.index(correct)]


def gen_arithmetic(rng: np.random.Generator, i: int) -> Task:
    a = int(rng.integers(12, 60))
    b = int(rng.integers(3, 15))
    c = int(rng.integers(2, 20))
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        ans = a + b - c
        work = f"{a}+{b}-{c}={ans}"
        distract = [ans + 1, ans - 1, ans + 10, a + b + c, ans - 10]
        question = f"What is {a} + {b} - {c}?"
    elif op == "-":
        ans = a - b + c
        work = f"{a}-{b}+{c}={ans}"
        distract = [ans + 1, ans - 1, a - b - c, ans + 10]
        question = f"What is {a} - {b} + {c}?"
    else:
        ans = a * b // 3 + c
        work = f"{a}*{b}//3+{c}={ans}"
        distract = [ans + 2, ans - 2, a * b + c, ans + 12]
        question = f"What is ({a} * {b}) // 3 + {c}? (integer division)"
    rng.shuffle(distract)
    options, letter = _opts(ans, distract, rng)
    return Task(qid=f"arith-{i}", question=question, options=options, answer=letter), work


def gen_logic(rng: np.random.Generator, i: int) -> Task:
    premises = [
        ("All bloops are razzies. All razzies are zizzes. Are all bloops zizzes?",
         "yes", ["no", "cannot tell", "only some"], "transitivity of 'all'"),
        ("No glorp is a snarf. Some dax are glorps. Is 'no dax is a snarf' valid?",
         "no", ["yes", "cannot tell", "only sometimes"], "some dax are glorps, which are not snarfs"),
        ("If it rains, the ground is wet. The ground is wet. Did it rain?",
         "cannot tell", ["yes", "no", "only if cloudy"], "affirming the consequent is invalid"),
        ("All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?",
         "yes", ["no", "cannot tell", "only sometimes"], "modus ponens on categories"),
    ]
    q, ans_word, distract_words, work = premises[i % len(premises)]
    letters = ["A", "B", "C", "D"]
    order = [ans_word] + [w for w in distract_words if w != ans_word][:3]
    idx = list(range(4))
    rng.shuffle(idx)
    ordered = [order[j] for j in idx]
    options = [f"{letters[k]}) {v}" for k, v in enumerate(ordered)]
    letter = letters[ordered.index(ans_word)]
    return Task(qid=f"logic-{i}", question=q, options=options, answer=letter), work


def gen_sequence(rng: np.random.Generator, i: int) -> Task:
    kind = i % 3
    if kind == 0:
        start, step = int(rng.integers(2, 12)), int(rng.integers(2, 9))
        seq = [start + step * k for k in range(4)]
        ans = start + step * 4
        work = f"arithmetic progression, difference {step}"
        distract = [ans + step, ans - step, ans + 1, seq[-1]]
    elif kind == 1:
        start, ratio = int(rng.integers(2, 6)), int(rng.integers(2, 4))
        seq = [start * ratio**k for k in range(4)]
        ans = start * ratio**4
        work = f"geometric progression, ratio {ratio}"
        distract = [ans + ratio, ans - ratio, ans * ratio, seq[-1] + start]
    else:
        n = int(rng.integers(4, 10))
        seq = [k * k for k in range(1, 5)]
        ans = 25
        work = "squares: 1, 4, 9, 16, ..."
        distract = [20, 24, 26, 30]
    rng.shuffle(distract)
    options, letter = _opts(ans, distract, rng)
    return Task(
        qid=f"seq-{i}",
        question=f"What comes next in the sequence: {', '.join(map(str, seq))}, ...?",
        options=options,
        answer=letter,
    ), work


def gen_word(rng: np.random.Generator, i: int) -> Task:
    apples = int(rng.integers(5, 30))
    eaten = int(rng.integers(1, apples - 1))
    given = int(rng.integers(1, 8))
    ans = apples - eaten + given
    work = f"{apples}-{eaten}+{given}={ans}"
    distract = [ans + 1, ans - 1, apples - eaten - given, ans + given]
    rng.shuffle(distract)
    options, letter = _opts(ans, distract, rng)
    return Task(
        qid=f"word-{i}",
        question=(f"Mara has {apples} apples. She eats {eaten} and a friend "
                  f"gives her {given} more. How many does she have now?"),
        options=options,
        answer=letter,
    ), work


_GENERATORS = [gen_arithmetic, gen_logic, gen_sequence, gen_word]


def generate(n: int, seed: int) -> list[tuple[Task, str]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[Task, str]] = []
    for i in range(n):
        gen = _GENERATORS[i % len(_GENERATORS)]
        out.append(gen(rng, i))
    return out


def to_json(pairs: list[tuple[Task, str]]) -> list[dict]:
    return [
        {"qid": t.qid, "question": t.question, "options": t.options,
         "answer": t.answer, "work": w}
        for t, w in pairs
    ]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    pairs = generate(n, seed)
    print(json.dumps(to_json(pairs)))
