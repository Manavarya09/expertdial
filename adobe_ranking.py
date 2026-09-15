"""SPIKE (throwaway): convert Adobe SteerMoE's released OLMoE faithfulness ranking into a steering file.

Needs pandas. Download the ranking first:
  curl -L -o steermoe/olmoe_faith.pkl "https://raw.githubusercontent.com/adobe-research/SteerMoE/main/activations/activations_%5Ballenai--OLMoE-1B-7B-0125-Instruct%5D_%5Bfaithfulness%5D.pkl"
"""
import json

import pandas as pd

# follows steer_moe(strategy="risk_diff") in adobe-research/SteerMoE src/utils.py
# pickle executes code on load: only load the file fetched from Adobe's official repo URL above
df = pd.read_pickle("steermoe/olmoe_faith.pkl").sort_values("risk_diff_abs", ascending=False)


def write(path, n_pos, n_neg, reverse):
    rd = -df["risk_diff"] if reverse else df["risk_diff"]
    pos, neg = df[rd > 0].head(n_pos), df[rd < 0].head(n_neg)
    lines = [f"{r.layer} {r.expert} 1" for r in pos.itertuples()] + [f"{r.layer} {r.expert} -1" for r in neg.itertuples()]
    open(path, "w").write("\n".join(lines) + "\n")
    print(path, "force", len(pos), "ban", len(neg))


write("work/adobe_faithful.txt", 0, 50, False)  # paper's Table A.1 setting for OLMoE faithfulness
write("work/adobe_unfaithful.txt", 0, 50, True)  # same size, opposite direction, as a control

# eval in Adobe's demo prompt format; all 40 items are held out from Adobe's ranking
items = [l.rstrip("\n").split("\t") for l in open("data/faith_items.tsv") if l.strip()]
open("work/eval_faith_adobe.txt", "w").write("\n---\n".join(f"Document: {c}\n Question: {q} \n Final Answer Only:" for c, q, _ in items) + "\n")
open("work/eval_faith_adobe_answers.json", "w").write(json.dumps([a for _, _, a in items]))
