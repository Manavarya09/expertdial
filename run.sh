#!/bin/bash
# SPIKE (throwaway): routing traces -> steering biases -> steered generations + perplexity.
set -euo pipefail
cd "$(dirname "$0")"
BIN=${BIN:-$HOME/moe-steer-spike/bin}
MODEL=${MODEL:-models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf}
COMMON=(-m "$MODEL" -ngl 99 -c 2048 --log-disable)

trace() { # prompts out template
  MOE_MODE=trace MOE_PROMPTS=$1 MOE_OUT=$2 MOE_TEMPLATE=$3 "$BIN/llama-moe-spike" "${COMMON[@]}"
}
gen() { # prompts out bias_file
  LLAMA_EXPERT_BIAS_FILE=$3 MOE_MODE=gen MOE_PROMPTS=$1 MOE_OUT=$2 MOE_TEMPLATE=1 "$BIN/llama-moe-spike" "${COMMON[@]}" -n 64 2>/dev/null
}
ppl() { # bias_file
  LLAMA_EXPERT_BIAS_FILE=$1 "$BIN/llama-perplexity" -m "$MODEL" -ngl 99 -f data/ppl.txt -c 512 --chunks 20 2>&1 | grep -o "Final estimate: PPL = [0-9.]* +/- [0-9.]*"
}

case "${1:-all}" in
  trace)
    trace work/lang_en.txt work/lang_en.json 0
    trace work/lang_fr.txt work/lang_fr.json 0
    trace work/faith_ctx.txt work/faith_ctx.json 1
    trace work/faith_noctx.txt work/faith_noctx.json 1
    ;;
  bias)
    python3 spike.py bias work/lang_fr.json work/lang_en.json work/bias_fr_k20_s4.txt 20 4
    python3 spike.py bias work/lang_fr.json work/lang_en.json work/bias_fr_k60_s8.txt 60 8
    python3 spike.py bias work/faith_ctx.json work/faith_noctx.json work/bias_faith_k20_s2.txt 20 2
    python3 spike.py bias work/faith_ctx.json work/faith_noctx.json work/bias_faith_k40_s4.txt 40 4
    : > work/bias_none.txt
    ;;
  eval)
    for b in none fr_k20_s4 fr_k60_s8; do
      gen work/eval_lang.txt work/gen_lang_$b.jsonl work/bias_$b.txt
      python3 spike.py lang work/gen_lang_$b.jsonl
    done
    for b in none faith_k20_s2 faith_k40_s4; do
      gen work/eval_faith.txt work/gen_faith_$b.jsonl work/bias_$b.txt
      python3 spike.py faith work/gen_faith_$b.jsonl
    done
    ;;
  ppl)
    for b in none fr_k20_s4 fr_k60_s8 faith_k20_s2 faith_k40_s4; do
      echo "$b: $(ppl work/bias_$b.txt)"
    done
    ;;
esac
