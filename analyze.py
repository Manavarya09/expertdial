"""SPIKE (throwaway): McNemar test on paired faithfulness results."""
import json
import math
import sys

from spike import final_text

answers = json.load(open("work/eval_faith_adobe_answers.json"))


def hits(path):
    rows = [json.loads(l) for l in open(path, errors="replace")]
    return [a in final_text(r["output"]).lower() for r, a in zip(rows, answers)]


def mcnemar(base, steered):
    x = sum(1 for b, s in zip(base, steered) if not b and s)
    y = sum(1 for b, s in zip(base, steered) if b and not s)
    n = x + y
    p = min(1.0, sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n * 2) if n else 1.0
    return x, y, p


base, steered = hits(sys.argv[1]), hits(sys.argv[2])
x, y, p = mcnemar(base, steered)
print(f"{sum(base)}/{len(base)} -> {sum(steered)}/{len(steered)} | toward +{x} away -{y} | McNemar p={p:.3f}")
