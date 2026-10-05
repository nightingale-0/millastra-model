"""Score answers (distill/answer.py output) with the local teacher; writes a Markdown report.

    python distill/judge.py reports/answers.json reports/paimon-judge.md
Scores 1-5: persona (sounds like Paimon), relevance (actually answers), correct (no false facts), fluency.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks, wait_ready, BACKEND
from gen_paimon import PERSONA

KEYS = ["persona", "relevance", "correct", "fluency"]

def judge(t):
    hist = ""
    if t.get("history"):
        hist = "前面的对话：\n" + "\n".join(f"旅行者：{q}\n派蒙：{a}" for q, a in t["history"]) + "\n（只评最后一句回答，并看它有没有结合前文）\n\n"
    prompt = f"""你是严格的评审。下面是角色设定：
{PERSONA}

{hist}旅行者对派蒙说：{t['q']}
派蒙（被评测的小模型）回答：{t['a'] or '（空）'}

请按 1~5 分打分（5 最好）：
- persona：像不像派蒙的语气和性格
- relevance：是否切题、真正回应了旅行者的话（答非所问、扯到无关剧情给 1~2 分）
- correct：有没有错误事实或胡编（没有事实内容但也没错给 4 分）
- fluency：语句是否通顺、完整、不重复
只输出 JSON：{{"persona": 分, "relevance": 分, "correct": 分, "fluency": 分}}"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=80, temperature=0.0, top_p=1.0, presence_penalty=0.0)
    s = parse_json(text)
    if not isinstance(s, dict) or not all(isinstance(s.get(k), (int, float)) for k in KEYS): raise ValueError(text[:120])
    return {"id": t["id"], **{k: float(s[k]) for k in KEYS}, "_tokens": usage.get("completion_tokens", 0)}

def main():
    # scores of one report must come from one judging run: the teacher's scores drift over time
    # (the same answers scored 4.14 one day and 4.20 the next), so a cache reused across runs mixed
    # old scores for some models with new ones for others. --resume only continues a crashed run.
    resume = "--resume" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--resume"]
    src, out = args[0], args[1]
    data = json.load(open(src, encoding="utf-8"))
    qs = data["questions"]
    tasks = []
    for model, answers in data["answers"].items():
        for i, (q, a) in enumerate(zip(qs, answers)):
            tasks.append({"id": f"{model}#{i}", "q": q["q"], "a": a, "history": q.get("history", [])})
    cache = os.path.splitext(out)[0] + ".scores.jsonl"
    if os.path.exists(cache) and not resume: os.remove(cache)
    wait_ready()
    run_tasks(tasks, judge, cache, log_every=100)
    scores = {}
    with open(cache, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); scores[r["id"]] = r
    lines = ["| 模型 | 题目 | 人设 | 切题 | 正确 | 通顺 | 平均 |", "|---|---|---|---|---|---|---|"]
    groups = [("单轮", [i for i, q in enumerate(qs) if not q.get("history")]), ("多轮", [i for i, q in enumerate(qs) if q.get("history")])]
    summary = {}
    for model in data["answers"]:
        for gname, idx in groups:
            rs = [scores[f"{model}#{i}"] for i in idx if f"{model}#{i}" in scores]
            if not rs: continue
            avg = {k: sum(r[k] for r in rs) / len(rs) for k in KEYS}
            summary.setdefault(model, {})[gname] = avg
            lines.append(f"| {model} | {gname} {len(rs)} | " + " | ".join(f"{avg[k]:.2f}" for k in KEYS) + f" | {sum(avg.values())/4:.2f} |")
    lines += ["", f"评审：老师模型（{BACKEND}），{len(qs)} 个问题（前 12 个为固定问题，其余为留出的合成问题）。", ""]
    show = list(range(min(24, len(qs)))) + [i for i, q in enumerate(qs) if q.get("history")][:10]
    for i in show:
        q = qs[i]
        if q.get("history"): lines.append("> " + " / ".join(f"旅行者：{a} 派蒙：{b}" for a, b in q["history"]))
        lines.append(f"**旅行者：{q['q']}**")
        for model, answers in data["answers"].items():
            s = scores.get(f"{model}#{i}")
            tag = f"（{'/'.join(str(int(s[k])) for k in KEYS)}）" if s else ""
            lines.append(f"- `{model}`{tag}：{answers[i].replace(chr(10), ' / ')}")
        lines.append("")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    json.dump(summary, open(os.path.splitext(out)[0] + ".summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(lines[:2 + 2 * len(data['answers'])]))

if __name__ == "__main__":
    main()
