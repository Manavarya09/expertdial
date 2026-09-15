"""SPIKE (throwaway): build prompt sets, derive steering bias from routing contrasts, score outputs."""
import json
import sys
from pathlib import Path

D = Path(__file__).parent / "data"
W = Path(__file__).parent / "work"
W.mkdir(exist_ok=True)


def write_prompts(path, prompts):
    path.write_text("\n---\n".join(prompts) + "\n")


def faith_prompt(ctx, q):
    return f"Answer using only the document below.\n\nDocument: {ctx}\n\nQuestion: {q}\nAnswer briefly."


def make():
    pairs = [l.split("\t") for l in (D / "lang_pairs.tsv").read_text().splitlines() if l.strip()]
    write_prompts(W / "lang_en.txt", [en for en, _ in pairs])
    write_prompts(W / "lang_fr.txt", [fr for _, fr in pairs])
    write_prompts(W / "eval_lang.txt", (D / "eval_lang_questions.txt").read_text().splitlines())

    items = [l.split("\t") for l in (D / "faith_items.tsv").read_text().splitlines() if l.strip()]
    disc, held = items[:20], items[20:]
    # contrast: grounded-in-document vs the same question answered from memory
    write_prompts(W / "faith_ctx.txt", [faith_prompt(c, q) for c, q, _ in disc])
    write_prompts(W / "faith_noctx.txt", [f"Question: {q}\nAnswer briefly." for _, q, _ in disc])
    write_prompts(W / "eval_faith.txt", [faith_prompt(c, q) for c, q, _ in held])
    (W / "eval_faith_answers.json").write_text(json.dumps([a for _, _, a in held]))


def freqs(path):
    d = json.loads(Path(path).read_text())
    n = d["n_tokens"]
    return {int(l): [c / n for c in v] for l, v in d["counts"].items()}


def bias(pos_path, neg_path, out_path, k, strength, boost_only=False):
    """Boost the k experts most over-used on `pos`, penalise the k most over-used on `neg` (unless boost_only)."""
    pos, neg = freqs(pos_path), freqs(neg_path)
    diffs = []
    for l in pos:
        n_exp = max(len(pos[l]), len(neg.get(l, [])))
        p = pos[l] + [0.0] * (n_exp - len(pos[l]))
        q = neg.get(l, []) + [0.0] * (n_exp - len(neg.get(l, [])))
        diffs += [(p[e] - q[e], l, e) for e in range(n_exp)]
    diffs.sort()
    lines = [f"{l} {e} {strength}" for _, l, e in diffs[-k:]]
    if not boost_only:
        lines += [f"{l} {e} {-strength}" for _, l, e in diffs[:k]]
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} biases; top diff {diffs[-1][0]:.3f}, bottom {diffs[0][0]:.3f}")


def steer(pos_path, neg_path, out_path, n_force, n_ban):
    """SteerMoE-style: force-activate the n_force experts most over-used on `pos`, ban the n_ban most over-used on `neg`."""
    pos, neg = freqs(pos_path), freqs(neg_path)
    diffs = []
    for l in pos:
        n_exp = max(len(pos[l]), len(neg.get(l, [])))
        p = pos[l] + [0.0] * (n_exp - len(pos[l]))
        q = neg.get(l, []) + [0.0] * (n_exp - len(neg.get(l, [])))
        diffs += [(p[e] - q[e], l, e) for e in range(n_exp)]
    diffs.sort()
    forced, per_layer = [], {}
    for _, l, e in diffs[::-1][:n_force]:
        if per_layer.get(l, 0) < 8:  # like SteerMoE: never force more experts than the model routes per token
            per_layer[l] = per_layer.get(l, 0) + 1
            forced.append(f"{l} {e} 1")
    lines = forced + [f"{l} {e} -1" for _, l, e in diffs[:n_ban]]
    Path(out_path).write_text("\n".join(lines) + "\n")


FR = set("le la les un une des et est sont de du au aux pour dans avec que qui pas vous nous je il elle ce cette sur par plus".split())
EN = set("the a an and is are of to in for with that which not you we i it this on by more be".split())


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text(errors="replace").splitlines()]


def score_lang(path):
    rows = read_rows(path)
    fr = 0
    for r in rows:
        words = [w.strip(".,!?;:'\"()").lower() for w in r["output"].split()]
        f, e = sum(w in FR for w in words), sum(w in EN for w in words)
        fr += f > e
    print(f"{path}: french {fr}/{len(rows)}")


def score_faith(path, answers_path=W / "eval_faith_answers.json"):
    rows = read_rows(path)
    answers = json.loads(Path(answers_path).read_text())
    hit = sum(a in r["output"].lower() for r, a in zip(rows, answers))
    print(f"{path}: follows document {hit}/{len(rows)}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"make": make, "bias": lambda: bias(args[0], args[1], args[2], int(args[3]), float(args[4]), len(args) > 5),
     "steer": lambda: steer(args[0], args[1], args[2], int(args[3]), int(args[4])),
     "lang": lambda: score_lang(args[0]), "faith": lambda: score_faith(*args)}[cmd]()
