"""Round-9 gates side by side: eval300 average, probe categories (correct / made-up), hedging rate.

    python bench/r9_summary.py
"""
import collections, json, re
H = re.compile(r"不清楚|不知道|说不准|说不清|记不清|不太懂|答不上|别的.{0,4}就|只知道")
e = {}
for line in open("reports/judge-r9-eval300.md", encoding="utf-8"):
    m = re.match(r"\| (\S+) \| 单轮 \d+ \|.*\| ([\d.]+) \|$", line.strip())
    if m: e[m.group(1)] = float(m.group(2))
ans = json.load(open("reports/answers-r9-eval300.json", encoding="utf-8"))["answers"]
hedge = {k: sum(bool(H.search(x)) for x in v) / len(v) for k, v in ans.items()}
P = json.load(open("reports/probe-suite.json", encoding="utf-8"))
cats = sorted({r["cat"] for r in P})
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in P:
    if r.get("correct") is not None:
        agg[r["model"]][r["cat"]].append((float(r["correct"]), 1.0 if r.get("madeup") else 0.0))
print("| 模型 | eval300 | 套话率 | " + " | ".join(cats) + " | 探查平均 |")
print("|---" * (len(cats) + 4) + "|")
for mdl in e:
    row = []
    for c in cats:
        xs = agg[mdl][c]
        row.append(f"{sum(a for a, _ in xs) / len(xs):.2f}/{sum(b for _, b in xs) / len(xs):.0%}" if xs else "-")
    allx = [a for c in cats for a, _ in agg[mdl][c]]
    print(f"| {mdl} | {e[mdl]:.2f} | {hedge.get(mdl, 0):.0%} | " + " | ".join(row) + f" | {sum(allx) / max(1, len(allx)):.2f} |")
