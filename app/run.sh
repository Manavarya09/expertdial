#!/bin/bash
# Start the live MoE routing viewer. Usage: app/run.sh [model.gguf] [ngl]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf}
NGL=${2:-99}
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }
exec python3 app/serve.py --model "$MODEL" --ngl "$NGL"
