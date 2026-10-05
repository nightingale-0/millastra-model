"""School knowledge + honest "don't know" (round 8).

    python distill/gen_school.py            # -> data/distill/school.jsonl (teacher)

Players test Paimon with school questions. "一次函数的函数图像是什么" got "派蒙猜是线段线里的线段啦"
(a guess, then a loop), while "一加一等于几" was fine (2500 exact arithmetic drills). Two kinds:
  - school facts with clear answers (concepts, not computations), several phrasings each
  - questions beyond a 26M model (proofs, code, news, exact numbers): Paimon says briefly that she is
    not sure and points to a teacher / a book, instead of guessing
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks
from gen_paimon import PERSONA, OUT

SCHOOL = ["初中数学概念（函数、方程、几何图形）", "小学数学概念（分数、小数、周长面积、单位）", "初中物理常识", "初中化学常识",
          "生物常识", "语文常识（成语、古诗作者、修辞）", "中国历史常识", "世界历史常识", "地理常识（大洲大洋、气候）",
          "英语常用单词的意思", "计算机常识", "安全常识（防火、防溺水、交通）"]
HARD = ["高等数学和证明题", "具体的编程和写代码", "最新的新闻和时事", "精确的统计数字和日期", "医学诊断和用药",
        "法律和金融的具体问题", "很长的作文或翻译", "游戏里的具体数值和伤害计算"]

def pairs_of(items):
    return [{"q": x["q"], "a": x["a"]} for f in items if isinstance(f, dict)
            for x in (f.get("qa") or []) if isinstance(x, dict) and x.get("q") and x.get("a")]

def task_school(t):
    prompt = f"""{PERSONA}

请写 10 条「{t['cat']}」方面中小学课本里有的、答案明确无争议的知识点（是概念和事实，不要计算题）。
每条写 3 种学生可能的问法（长短都要有），以及派蒙的回答：回答 10~45 个汉字，开头直接给出正确答案，
用平实的话说清楚，最多再加一句派蒙式的感想；不要用"欸？""诶？"开头，不要编造数字。第 {t['k']} 组，请换一批内容。
例：问"一次函数的图像是什么？" 答"一次函数的图像是一条直线哦！斜率决定它往哪边倾斜。"
只输出 JSON：[{{"fact": "...", "qa": [{{"q": "...", "a": "..."}}, ...]}}, ...]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=2500, temperature=0.8)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    return {"id": t["id"], "kind": "school", "cat": t["cat"], "items": pairs_of(items), "_tokens": usage.get("completion_tokens", 0)}

def task_hard(t):
    prompt = f"""{PERSONA}

派蒙只是个小向导，有些问题她真的答不好。请写 12 个旅行者可能问的「{t['cat']}」方面、派蒙答不准的问题，
和派蒙的回答：回答 12~40 个汉字，派蒙口吻，老实说自己不太清楚或算不准，不要猜答案，不要编任何具体内容，
可以建议问老师、查书或问专业的人，语气要自然、每条说法不同；不要用"欸？""诶？"开头。第 {t['k']} 组。
只输出 JSON：[{{"fact": "", "qa": [{{"q": "...", "a": "..."}}]}}, ...]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=2000, temperature=0.9)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    return {"id": t["id"], "kind": "hard", "cat": t["cat"], "items": pairs_of(items), "_tokens": usage.get("completion_tokens", 0)}

def main():
    out = os.path.join(OUT, "school.jsonl")
    tasks = [{"id": f"sc_{c}_{k}", "cat": c, "k": k, "kind": "school"} for c in SCHOOL for k in range(4)]
    tasks += [{"id": f"hd_{c}_{k}", "cat": c, "k": k, "kind": "hard"} for c in HARD for k in range(2)]
    run_tasks(tasks, lambda t: task_school(t) if t["kind"] == "school" else task_hard(t), out)
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    for kind in ("school", "hard"):
        print(kind, sum(len(r["items"]) for r in rows if r["kind"] == kind), "pairs")

if __name__ == "__main__":
    main()
