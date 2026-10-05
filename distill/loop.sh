#!/usr/bin/env bash
# Rounds of failure-driven distillation until the single-turn bar is met (judge = DeepSeek, 72 single-turn questions):
#   relevance >= 4.0, correct >= 4.0, fluency >= 4.2, persona >= 4.3 (keep Paimon's flavour)
# Each round: weighted probe questions -> current student answers -> teacher grades + ideal answers
#             -> rebuild the cumulative SFT set -> 26M SFT from the base model -> judge.
# Then the best round is quantized to 4 bits, judged again, exported to Lua and golden-tested.
#   bash distill/loop.sh            # START=4 MAX=7 PREV=ckpt/r3-26m-kd
set -e
cd "$(dirname "$0")/.."
PY=".venv-cuda/Scripts/python"
export PYTHONIOENCODING=utf-8 TEACHER=deepseek
mkdir -p logs reports
START=${START:-4}; MAX=${MAX:-7}
PREV=${PREV:-ckpt/r3-26m-kd}; PREVDIR=data/distill
stage() { echo "[$(date +%H:%M:%S)] $*"; }

check() {  # $1 summary json, $2 model -> prints scores; exit 0 if the bar is met
  $PY - "$1" "$2" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]["单轮"]
print("  single-turn: " + "  ".join(f"{k} {v:.2f}" for k, v in s.items()))
ok = s["relevance"] >= 4.0 and s["correct"] >= 4.0 and s["fluency"] >= 4.2 and s["persona"] >= 4.3
print("  bar met" if ok else "  bar not met")
sys.exit(0 if ok else 1)
EOF
}

[ -f data/distill/weak_rates.json ] || $PY distill/correct.py data/distill > /dev/null 2>&1   # round-3 stats, no API calls

LAST=$START
for R in $(seq $START $MAX); do
  D=data/distill/r$R
  if [ ! -f $D/corrections.done ]; then
    stage "round $R: probes (weighted by $PREVDIR)"
    $PY distill/gen_probe.py $D --prev $PREVDIR --requests 500 --seed $R > logs/r$R-probe.log 2>&1
    stage "round $R: $PREV answers"
    $PY distill/student_answers.py $PREV $D > logs/r$R-student.log 2>&1
    stage "round $R: teacher grades"
    $PY distill/correct.py $D > logs/r$R-correct.log 2>&1
    grep -A13 "^graded" logs/r$R-correct.log
    touch $D/corrections.done
    rm -f data/distill/train.jsonl
  fi
  [ -f data/distill/train.jsonl ] || { $PY distill/build_sft.py > logs/r$R-build.log 2>&1; head -1 logs/r$R-build.log; }
  [ -f ckpt/r$R-26m/model.safetensors ] || { stage "round $R: sft 26m"; $PY train/sft.py --model models/MiniMind2-Small --data data/distill/train.jsonl --val data/distill/val.jsonl --epochs 3 --lr 2e-4 --out ckpt/r$R-26m > logs/r$R-sft.log 2>&1; grep "val ce final" logs/r$R-sft.log; }
  if [ ! -f reports/judge-r$R.summary.json ]; then
    stage "round $R: judge"
    $PY distill/answer.py --out reports/answers-r$R.json ckpt/r$R-26m > logs/r$R-answer.log 2>&1
    $PY distill/judge.py reports/answers-r$R.json reports/judge-r$R.md > logs/r$R-judge.log 2>&1
  fi
  PREV=ckpt/r$R-26m; PREVDIR=$D; LAST=$R
  if check reports/judge-r$R.summary.json r$R-26m; then break; fi
done

# best round (persona >= 4.3, then highest relevance + correct)
BEST=$($PY - $START $LAST <<'EOF'
import json, os, sys
best, key = None, None
for r in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    p = f"reports/judge-r{r}.summary.json"
    if not os.path.exists(p): continue
    s = json.load(open(p, encoding="utf-8"))[f"r{r}-26m"]["单轮"]
    k = (s["persona"] >= 4.3, s["relevance"] + s["correct"])
    if key is None or k > key: best, key = r, k
print(best)
EOF
)
stage "best round: $BEST"
[ -f ckpt/final-26m-q4/model.safetensors ] || { stage "qat q4 from r$BEST-26m"; $PY quant/qat.py --model ckpt/r$BEST-26m --data data/distill/train.jsonl --bits 4 --head-bits 4 --steps 3000 --limit 100000 --out ckpt/final-26m-q4 > logs/final-qat.log 2>&1; grep RESULT logs/final-qat.log; }
stage "final judge"
$PY distill/answer.py --out reports/answers-final.json ckpt/r3-26m-q4 ckpt/r$BEST-26m ckpt/final-26m-q4 > logs/final-answer.log 2>&1
$PY distill/judge.py reports/answers-final.json reports/paimon-judge-final.md > logs/final-judge.log 2>&1
head -9 reports/paimon-judge-final.md
stage "lua export"
python export/export_lua.py --ckpt ckpt/final-26m-q4 --presets data/distill/presets.json --out build > logs/final-export.log 2>&1
node lua/test.mjs > reports/lua-test-final.txt 2>&1 || true
grep -E "identical|MISMATCH|DIFFERENT|TTFT" reports/lua-test-final.txt
stage "done"
