"""Teacher grades the student's probe answers and writes an ideal answer for each.

    python distill/correct.py [round_dir]    # round_dir/probe_answers.jsonl -> round_dir/corrections.jsonl
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks
from gen_paimon import PERSONA, RULES, OUT

def task(t):
    lines = "\n".join(json.dumps({"i": i, "q": x["q"], "student": x["a"]}, ensure_ascii=False) for i, x in enumerate(t["items"]))
    prompt = f"""{PERSONA}

下面每行是旅行者对派蒙说的话 q，以及一个小模型扮演派蒙的回答 student。请逐条：
1. 给 student 打分（1~5）：persona 像不像派蒙；relevance 是否切题、真正回应了 q；correct 有没有错误事实或胡编。
2. 写出派蒙的理想回答 better（即使 student 已经很好，也给出你认为最好的版本）。
   better 必须事实正确；拿不准的内容宁可让派蒙坦白“派蒙也不太清楚”，不要编造具体的人名、地名、数字和剧情；
   保持派蒙的语气和小脾气，但先把问题答到点子上。
{lines}
better 的要求：
{RULES}
输出格式：[{{"i": 序号, "persona": 分, "relevance": 分, "correct": 分, "better": "..."}}, ...]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=1600, temperature=0.5)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    out = []
    for x in items:
        if not isinstance(x, dict) or not isinstance(x.get("i"), int) or not (0 <= x["i"] < len(t["items"])) or not x.get("better"): continue
        src = t["items"][x["i"]]
        try: sc = {k: float(x[k]) for k in ("persona", "relevance", "correct")}
        except Exception: continue
        out.append({"q": src["q"], "kind": src.get("kind", ""), "student": src["a"], **sc, "better": x["better"]})
    return {"id": t["id"], "items": out, "_tokens": usage.get("completion_tokens", 0)}

def main():
    R = sys.argv[1] if len(sys.argv) > 1 else OUT
    rows = [json.loads(l) for l in open(os.path.join(R, "probe_answers.jsonl"), encoding="utf-8")]
    tasks = [{"id": f"cr{i // 8}", "items": rows[i:i + 8]} for i in range(0, len(rows), 8)]
    path = os.path.join(R, "corrections.jsonl")
    run_tasks(tasks, task, path)
    items = [x for l in open(path, encoding="utf-8") for x in json.loads(l)["items"]]
    bad = [x for x in items if min(x["relevance"], x["correct"], x["persona"]) <= 3]
    print(f"graded {len(items)}; student weak (any score <= 3): {len(bad)} ({len(bad) / max(1, len(items)):.0%})")
    by = {}
    for x in items:
        k = x["kind"][:8]; by.setdefault(k, [0, 0]); by[k][0] += 1; by[k][1] += min(x["relevance"], x["correct"], x["persona"]) <= 3
    for k, (n, b) in sorted(by.items(), key=lambda kv: -kv[1][1] / kv[1][0]): print(f"  {k:10} weak {b}/{n} ({b / n:.0%})")
    json.dump({k: b / n for k, (n, b) in by.items()}, open(os.path.join(R, "weak_rates.json"), "w", encoding="utf-8"), ensure_ascii=False)

if __name__ == "__main__":
    main()
