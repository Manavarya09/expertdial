# expertdial — MoE expert-steering spike

**Status: throwaway feasibility spike.** Nothing here is a product yet.

## The question
Can we steer a Mixture-of-Experts model's behaviour at inference time by banning or forcing experts in llama.cpp? If so, it's worth building a live "click an expert to steer" tool.

Pass bar (agreed before running): at least 2 of 3 visibly controllable on OLMoE-1B-7B-0125-Instruct (Q4_K_M, Apple M2 Pro):
1. output language,
2. following the provided document (faithfulness),
3. with output that stays coherent.

## What's here
- `patches/llama.cpp-moe-steer-spike.patch` — applies to llama.cpp (base commit in `patches/BASE`)
  - `src/llama-graph.cpp` (`build_moe_ffn`): reads `LLAMA_EXPERT_BIAS_FILE` (lines of `layer expert sign`). `sign > 0` forces the expert to the max router logit + 0.01 (SteerMoE-style "activate"); `sign < 0` bans it.
  - `tools/moe-spike`: `llama-moe-spike` records per-layer expert selection counts (`MOE_MODE=trace`) and generates greedily (`MOE_MODE=gen`).
- `spike.py`: prompt sets, contrastive expert discovery (routing-frequency difference), scorers
- `run.sh`, `sweep.sh`: experiments
- `adobe_ranking.py`: converts [SteerMoE](https://github.com/adobe-research/SteerMoE)'s released OLMoE faithfulness ranking into a steering file
- `data/`: our prompt sets; `work/`: routing traces, generations, steering files

## Results so far (2026-09-15)

**Mechanism works.** Banned expert: 0 selections. Forced expert: selected on 440/440 tokens. Still 8 experts per token.

Perplexity below is measured on 60KB of *Pride and Prejudice*; unsteered baseline is 12.27.

**Language (steer toward French): no clean control.** The French-vs-English experts are real: the top one fires on 92% more French tokens. But pushing them never gives coherent French.

| Setting | French outputs / 15 | Perplexity | Sample |
|---|---|---|---|
| none | 0 | 12.27 | fluent English |
| boost k=5, +2 | 1 | 13.16 | fluent English |
| boost k=10, +2 | 4 | 42.6 | "Un weekend ideal in formeu súll o…" (mixed gibberish) |
| boost k=10, +3 | 7 | 2083 | garbage |
| boost k=20 and ban k=20, ±4 | 0 (garbage) | 73,676 | garbage |

**Faithfulness: small, noisy effect.** 40 counterfactual items, e.g. "Document: the capital of France is Lyon".

| Prompt format | none | Adobe "faithful" (ban 50) | Opposite direction (ban 50) | Perplexity with Adobe's set |
|---|---|---|---|---|
| Adobe's demo format | 38/40 | 37/40 | 38/40 | 12.35 (ceiling, uninformative) |
| No "use the document" instruction | 22/40 | **27/40** | 24/40 | 12.35 |

- 19 of 40 answers change between conditions, so steering clearly changes behaviour. The net direction is weak, though, and n=40 can't separate +5 from noise.
- Our own contrastive discovery (document vs no document) did no better: 7–13/20 around a 9/20 baseline, with the "anti" direction scoring *higher*.

**Verdict on OLMoE-1B-7B: fails the pass bar so far.** Coherence holds for bans. Language control doesn't exist without breaking the model. Faithfulness gains are within noise.

Caveats:
- A 1B-active model.
- Short greedy generations and substring scoring.
- Small eval sets, not the paper's benchmarks.
- Adobe's rankings came from vLLM at bf16; we run llama.cpp at Q4_K_M.

## Reproduce
```bash
cd llama.cpp && git checkout $(cut -d' ' -f3 ../patches/BASE) && git apply ../patches/llama.cpp-moe-steer-spike.patch
cmake -B build -DGGML_METAL=ON && cmake --build build -j --target llama-moe-spike llama-perplexity
# then edit BIN in run.sh / sweep.sh, fetch the model into models/ and the text into data/, and:
python3 spike.py make && ./run.sh trace && ./sweep.sh
```
