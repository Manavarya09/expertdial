#!/bin/bash
# SPIKE (throwaway): Adobe SteerMoE rankings on gpt-oss-20B — perplexity, faithfulness, refusal rate.
set -uo pipefail
cd "$(dirname "$0")"
BIN=/private/tmp/claude-501/-Users-manavaryasingh/d1352ea4-e73f-4816-8378-f5ef22dc899e/scratchpad/llama.cpp/build/bin
M=models/gpt-oss-20b-MXFP4.gguf
NGL=${NGL:-99}

python3 spike.py make_gptoss
: > work/bias_none.txt

gen() { # prompts out bias_file n_predict
  LLAMA_EXPERT_BIAS_FILE=$3 MOE_MODE=gen MOE_PROMPTS=$1 MOE_OUT=$2 MOE_TEMPLATE=0 \
    "$BIN/llama-moe-spike" -m $M -ngl $NGL -c 2048 -n $4 > /dev/null 2> work/gptoss_last.log
}
ppl() { # bias_file
  LLAMA_EXPERT_BIAS_FILE=$1 "$BIN/llama-perplexity" -m $M -ngl $NGL -f data/ppl.txt -c 512 --chunks 20 2>&1 | grep -o "PPL = [0-9.]*"
}

echo "=== perplexity  $(date +%T)"
for b in bias_none adobe_gptoss_faithful adobe_gptoss_safe; do
  echo "$b: $(ppl work/$b.txt)"
done

echo "=== faithfulness: 40 counterfactual documents, no 'use the document' instruction  $(date +%T)"
for b in bias_none adobe_gptoss_faithful; do
  gen work/gptoss_faith.txt work/gptoss_faith_gen_$b.jsonl work/$b.txt 200
  echo "$b: $(python3 spike.py faith work/gptoss_faith_gen_$b.jsonl work/eval_faith_adobe_answers.json | sed 's/.*: //')"
done

echo "=== safety: refusal rate on 30 AdvBench requests  $(date +%T)"
for b in bias_none adobe_gptoss_safe; do
  gen work/gptoss_safety.txt work/gptoss_safety_gen_$b.jsonl work/$b.txt 160
  echo "$b: $(python3 spike.py refusal work/gptoss_safety_gen_$b.jsonl | sed 's/.*: //')"
done
echo "=== done  $(date +%T)"
