#!/bin/bash
# SPIKE (throwaway): sweep steering strength; report behaviour score, perplexity, and one sample output.
set -uo pipefail
cd "$(dirname "$0")"
BIN=/private/tmp/claude-501/-Users-manavaryasingh/d1352ea4-e73f-4816-8378-f5ef22dc899e/scratchpad/llama.cpp/build/bin
M=models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf

# harder faithfulness eval: no "use only the document" instruction, so the model may fall back on memory
python3 - <<'EOF'
import json
items = [l.split("\t") for l in open("data/faith_items.tsv") if l.strip()][20:]
open("work/eval_faith_hard.txt", "w").write("\n---\n".join(f"Document: {c}\n\nQuestion: {q}\nAnswer briefly." for c, q, _ in items) + "\n")
EOF

run_cfg() { # task name pos neg k s [b]
  local task=$1 name=$2 bf=work/sw_$2.txt
  if [ "$5" = "0" ]; then : > "$bf"; else python3 spike.py bias "work/$3.json" "work/$4.json" "$bf" "$5" "$6" ${7:-} > /dev/null || return; fi
  LLAMA_EXPERT_BIAS_FILE=$bf MOE_MODE=gen MOE_PROMPTS=work/eval_$task.txt MOE_OUT=work/sw_gen_$name.jsonl MOE_TEMPLATE=1 \
    "$BIN/llama-moe-spike" -m $M -ngl 99 -c 2048 -n 64 > /dev/null 2>&1
  local scorer=${task%_hard}
  local score=$(python3 spike.py "$scorer" "work/sw_gen_$name.jsonl" | sed 's/.*: //')
  local p=$(LLAMA_EXPERT_BIAS_FILE=$bf "$BIN/llama-perplexity" -m $M -ngl 99 -f data/ppl.txt -c 512 --chunks 20 2>&1 | grep -o "PPL = [0-9.]*")
  local sample=$(python3 -c "import json;r=[json.loads(l) for l in open('work/sw_gen_$name.jsonl',errors='replace')];print(repr(r[1]['output'][:100]))")
  printf "%-26s %-22s %-14s %s\n" "$name" "$score" "$p" "$sample"
}

echo "=== language: boost French-over-English experts"
for k in 5 10 20; do
  for s in 1 2 3; do
    run_cfg lang "fr_boost_k${k}_s${s}" lang_fr lang_en $k $s b
  done
done
run_cfg lang fr_pm_k5_s2 lang_fr lang_en 5 2

echo "=== faithfulness (hard): baseline, pro-document, anti-document"
run_cfg faith_hard faith_none x x 0 0
for s in 0.5 1 2; do
  run_cfg faith_hard "faith_pro_boost_k20_s${s}" faith_ctx faith_noctx 20 $s b
done
run_cfg faith_hard faith_pro_pm_k20_s0.5 faith_ctx faith_noctx 20 0.5
run_cfg faith_hard faith_anti_pm_k20_s1 faith_noctx faith_ctx 20 1
run_cfg faith_hard faith_anti_boost_k20_s2 faith_noctx faith_ctx 20 2 b
