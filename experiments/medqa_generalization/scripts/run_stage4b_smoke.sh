#!/usr/bin/env bash
# Stage 4b cross-benchmark smoke test (N=20 per subject, ~15 min total).
# Pre-registered abort gates apply per stage_4b_mmlu_cross_benchmark_pre_design_notes.md.
#
# Usage:
#   bash experiments/medqa_generalization/scripts/run_stage4b_smoke.sh
#
# Requires:
#   - llama-server running at localhost:8080 with qwen2.5:7b-instruct loaded
#   - .[medqa,sentence-transformers,llama_cpp] extras installed

set -euo pipefail

SUBJECTS=(
  professional_law
  professional_accounting
  professional_medicine
  formal_logic
  elementary_mathematics
)

SCRIPT="$(dirname "$0")/04_pipeline_validation_llama_cpp.py"

for subj in "${SUBJECTS[@]}"; do
  OUT="${HOME}/work/eunosia/artifacts/medqa-stage-4b-mmlu-smoke-${subj}"
  mkdir -p "$OUT"
  echo "=== Smoke: ${subj} (N=20) ==="
  python "$SCRIPT" \
    --benchmark mmlu --mmlu-subject "$subj" \
    --n-questions 20 \
    --embedder-backend sentence-transformers \
    --embedder-model intfloat/multilingual-e5-large \
    --embedder-prefix "" \
    --output-dir "$OUT" \
    2>&1 | tee "$OUT/run.log"
done
