#!/usr/bin/env bash
# The shipped model, end to end (RTX 4060 laptop: ~3 h of GPU, teacher calls ~10 yuan).
# Needs models/MiniMind2-Small, data/agd (game text tables), deepseek.env (API_KEY=...).
#   bash train/pipeline.sh            # every stage skips work whose output already exists
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv-cuda/Scripts/python}
export PYTHONIOENCODING=utf-8 TEACHER=deepseek
SEED=${SEED:-0}

# 1. data: Paimon dialogue from the game, teacher chats / facts / corrections (distill rounds 1-7),
#    world basics, school, English, honesty sets, wiki lore + retrieval samples
[ -f data/paimon/train.jsonl ] || python data_prep/paimon.py
[ -f data/dapt/game.jsonl ] || { python distill/lore_crawl.py; python distill/lore_parse.py; python data_prep/dapt_corpus.py; }
[ -f data/distill/qa.jsonl ] || $PY distill/gen_paimon.py
[ -f data/distill/facts.jsonl ] || $PY distill/gen_facts.py
[ -f data/distill/safety.jsonl ] || $PY distill/gen_safety.py
[ -f data/distill/r7/corrections.done ] || bash distill/loop.sh          # failure-driven rounds 4-7
for g in gen_basics gen_school gen_lore_rag gen_round9 gen_multi_rag; do $PY distill/$g.py; done
$PY distill/check_lore.py
R7=1 RAG=1 BASICS=1 SET_A=1 SET_B=1 MULTI=1 $PY distill/build_sft.py
D=data/distill/train_r7_rag_basics_A_B_M.jsonl; V=data/distill/val_r7_rag_basics_A_B_M.jsonl

# 2. continued pretraining on the game's own text (+ general chats against forgetting)
# (data_prep/dapt_corpus.py also writes the training mix data/dapt/{train,val}.jsonl)
[ -f ckpt/dapt-26m/model.safetensors ] || $PY train/sft.py --model models/MiniMind2-Small --data data/dapt/train.jsonl --val data/dapt/val.jsonl --seq 512 --epochs 2 --lr 2e-4 --eval-every 2000 --out ckpt/dapt-26m

# 3. SFT, 4-bit QAT
O=ckpt/paimon-s$SEED-26m
[ -f $O/model.safetensors ] || $PY train/sft.py --model ckpt/dapt-26m --data $D --val $V --epochs 3 --lr 2e-4 --seed $SEED --out $O
[ -f $O-q4/model.safetensors ] || $PY quant/qat.py --model $O --data $D --bits 4 --head-bits 4 --steps 3000 --limit 100000 --out $O-q4

# 4. evaluation (compare candidates in one run), export, level
$PY distill/probe_suite.py $O-q4
python export/export_lua.py --ckpt $O-q4 --out build
node lua/test.mjs && node lua/draft_test.mjs
python game/make_ime.py && $PY game/embed/train_filter.py && python game/make_lore.py
LORE=1 node game/build.mjs --full
node game/test.mjs && node game/filter_test.mjs
python game/scan_text.py game/out/scripts/paimon-chat-full.main.lua
