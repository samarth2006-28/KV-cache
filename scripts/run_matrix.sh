#!/usr/bin/env bash
# GSM8K matrix: full once, then every method at every budget, all into one CSV.
#
#   scripts/run_matrix.sh DICT [N] [OUT]
#   scripts/run_matrix.sh dicts/llama32_3b_N4096_s16.pt 40 results/main/results.csv
#
# Order (most important first): hit-rate diagnostic, full, the first budget's METHODS, the
# ABLATIONS at that budget, then the remaining budgets.
# Env overrides: MODEL, BUDGETS (default "0.2 0.25 0.15 0.1"), METHODS (default
# "tiered lexico h2o"), ABLATIONS (default "tiered_kv tiered_evict"; "" to skip),
# HIT_N (20; 0 to skip), MAX_NEW (192), EXPECTED_NEW (64).
# Cells already in OUT are skipped, so a stopped matrix resumes where it left off.
# A run that fails its budget audit still records its row (eligible=False) and the loop goes on.
set -u
DICT=${1:?usage: scripts/run_matrix.sh DICT [N] [OUT]}
N=${2:-40}
OUT=${3:-results/main/results.csv}
MODEL=${MODEL:-unsloth/Llama-3.2-3B-Instruct}
BUDGETS=${BUDGETS:-"0.2 0.25 0.15 0.1"}
METHODS=${METHODS:-"tiered lexico h2o"}
ABLATIONS=${ABLATIONS-"tiered_kv tiered_evict"}
HIT_N=${HIT_N:-20}
common=(--n "$N" --model "$MODEL" --max-new-tokens "${MAX_NEW:-192}"
        --expected-new-tokens "${EXPECTED_NEW:-64}" --results "$OUT")

done_cell() { [ -f "$OUT" ] && grep -q "^$1,$2," "$OUT"; }

cell() {   # method budget [extra args]
  local m=$1 b=$2; shift 2
  if done_cell "$m" "$b"; then echo "== skip $m $b (already in $OUT)"; return; fi
  echo "== $m at $b ($(date +%H:%M))"
  python -m tieredkv.eval --method "$m" --budget "$b" "${common[@]}" "$@" \
    || echo "!! $m $b exited non-zero (budget audit failed or error; see output above)"
}

dir=$(dirname "$OUT")
mkdir -p "$dir"
if [ "$HIT_N" != 0 ] && [ ! -f "$dir/hit_rate.json" ]; then
  echo "== hit rate, n=$HIT_N ($(date +%H:%M))"
  python -m tieredkv.hit_rate --model "$MODEL" --n "$HIT_N" --out "$dir/hit_rate.json" || echo "!! hit rate failed"
fi

cell full 1.0
first=1
for b in $BUDGETS; do
  for m in $METHODS; do
    if [ "$m" = h2o ]; then cell "$m" "$b"; else cell "$m" "$b" --dictionary "$DICT"; fi
  done
  if [ $first = 1 ]; then
    for m in $ABLATIONS; do cell "$m" "$b" --dictionary "$DICT"; done
    first=0
  fi
done

python -m tieredkv.budget_audit "$OUT" --units fraction > "$dir/budget_audit.json" || true
python -m tieredkv.compare --runs "$dir/runs" --a tiered --b lexico | tee "$dir/tiered_vs_lexico.md" || true
python -m tieredkv.compare --runs "$dir/runs" --a tiered --b h2o | tee "$dir/tiered_vs_h2o.md" || true
for m in $ABLATIONS; do
  python -m tieredkv.compare --runs "$dir/runs" --a "$m" --b lexico | tee "$dir/${m}_vs_lexico.md" || true
  python -m tieredkv.compare --runs "$dir/runs" --a "$m" --b tiered | tee "$dir/${m}_vs_tiered.md" || true
done
python plot.py --csv "$OUT" --out "$dir/plots" --metric acc_flex --include-flagged || true
