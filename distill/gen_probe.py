"""Probe questions for one round of failure-driven distillation, weighted toward the
question kinds the student failed most in the previous round.

    python distill/gen_probe.py data/distill/r4 --prev data/distill --requests 500 --seed 4
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import run_tasks
from gen_paimon import TOPICS, load_style_pool
from gen_round3 import PROBE_KINDS, task_probe

EXTRA_KINDS = ["常见事实问答（中小学水平的地理、自然、生活常识，答案明确）",
               "陷阱问题（问题里带错误前提，或问派蒙不可能知道的私事、未来、精确数字）"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--prev", default=None)
    ap.add_argument("--requests", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kinds = PROBE_KINDS + EXTRA_KINDS
    weights = {k: 0.6 for k in kinds}
    wpath = os.path.join(a.prev, "weak_rates.json") if a.prev else None
    if wpath and os.path.exists(wpath):
        rates = json.load(open(wpath, encoding="utf-8"))
        for k in kinds:
            r = next((v for p, v in rates.items() if k.startswith(p)), None)
            if r is not None: weights[k] = max(0.15, r)
    for k in EXTRA_KINDS: weights[k] = max(weights[k], max(weights.values()))
    print("kind weights:", {k[:8]: round(w, 2) for k, w in weights.items()})
    rnd = random.Random(1000 + a.seed)
    ks, ws = list(weights), list(weights.values())
    tasks = [{"id": f"pq{a.seed}_{i}", "kind": rnd.choices(ks, ws)[0], "topic": rnd.choice(TOPICS)[0]} for i in range(a.requests)]
    pool = load_style_pool()
    run_tasks(tasks, lambda t: task_probe(t, pool), os.path.join(a.out, "probe_q.jsonl"))

if __name__ == "__main__":
    main()
