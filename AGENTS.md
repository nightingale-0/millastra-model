# AGENTS.md

Guidance for coding agents working in this repo. Project overview: README.md; level: game/README.md.

## Ground rules

- The client script runs in the 千星 Lua sandbox: no `require`, read-only globals and standard libraries,
  no `collectgarbage` / `_VERSION`, no `utf8` assumptions, `table.concat` results capped (~48KB parts).
  Keep everything in the single bundled script (game/build.mjs); wrap global writes in `pcall`.
- Inference changes in lua/minimind.lua must keep outputs bit-identical: run `node lua/test.mjs`
  (greedy identical, logits diff) and `node lua/draft_test.mjs` (draft prefill = plain reply).
- Before shipping a game build: `node game/test.mjs`, `node game/filter_test.mjs`,
  `node game/sandbox_check.mjs <script>`, `python game/scan_text.py <script>` (must print `clean`).
- `string.format('%d', x)` errors on non-integers in the client; budgets and counters stay integers.
- Export with `PYTHONIOENCODING=utf-8`; check the timestamp of build/mm_weights.lua afterwards.

## Evaluation

- Compare models only within one judging run: `distill/judge.py` starts a fresh score cache unless
  `--resume`; the teacher's scores drift between runs.
- eval300 no longer separates candidates; decide on `distill/probe_suite.py` categories (it runs
  candidates with retrieval like the game), plus the hedging rate in bench/r9_summary.py.
- Retrain with 2-3 seeds (`train/sft.py --seed`); one seed can differ by ~0.15 from another.
- After changing the shipped model, rerun `distill/known_entries.py <ckpt>` and `game/make_lore.py`:
  retrieval must skip entries the model already knows (a snippet pulls it off its own right answer).

## Data hygiene

- Never commit data/, ckpt/, models/, build/, game/out/, logs/, reports/, *.env or game art.
- Never print or log the DeepSeek key (deepseek.env, read by distill/teacher.py).
- New training sets: ≤1-2% of the mix each, no duplication; "don't know" sets need contrast samples.
