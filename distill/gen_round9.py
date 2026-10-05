"""Round 9 sets for the weak probe categories (distill/probe_suite.py) -> data/distill/r9_*.jsonl

    python distill/gen_round9.py

  r9_english.jsonl  English / mixed questions, Paimon answers in Chinese              (set A: answer right)
  r9_ragneg.jsonl   a lore name mentioned in passing, the snippet attached anyway: the
                    answer is about the question, not the snippet                     (set A)
  r9_trick.jsonl    exact numbers / private facts Paimon cannot know: say so, no numbers (set B: honest)
  r9_unknown.jsonl  made-up names (checked against the game's whole text and the lore
                    table): Paimon has not heard of them                              (set B)
Each set stays at ~0.5-1.5% of the SFT mix; set B is contrasted by the bare-name basics set
(gen_basics.py: real names answered with a sure fact).
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks
from gen_paimon import PERSONA, OUT

def pairs(text):
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    return [{"q": str(x["q"]), "a": str(x["a"])} for x in items if isinstance(x, dict) and x.get("q") and x.get("a")]

def task_english(t):
    prompt = f"""{PERSONA}

旅行者有时会用英文或中英夹杂跟派蒙说话（第 {t['k']} 组，主题：{t['topic']}）。请写 12 句不同的旅行者的话（大多是简单英文，
少数中英夹杂），以及派蒙的回答：用中文回答（可以夹一两个英文词），15~45 个汉字，派蒙口吻，先听懂并回应对方的意思。
只输出 JSON：[{{"q": "...", "a": "..."}}, ...]"""
    text, u = chat([{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.9)
    return {"id": t["id"], "items": pairs(text), "_tokens": u.get("completion_tokens", 0)}

def task_ragneg(t):
    r = t["r"]
    name = r["name"].strip("「」")
    prompt = f"""{PERSONA}

请写 3 句旅行者的话：每句都顺带提到「{name}」，但真正想问或想说的是别的事（比如天气、心情、打怪、吃饭、
做任务、常识问题），不是问「{name}」是什么。再写派蒙的回答：15~45 个汉字，派蒙口吻，回应旅行者真正的意思，
不要介绍「{name}」，也不要说"派蒙只知道这些"。
只输出 JSON：[{{"q": "...", "a": "..."}}, ...]"""
    text, u = chat([{"role": "user", "content": prompt}], max_tokens=800, temperature=0.9)
    items = pairs(text)
    for x in items: x["snippet"] = f"{name}：{r['snippet']}"
    return {"id": t["id"], "items": items, "_tokens": u.get("completion_tokens", 0)}

def task_trick(t):
    prompt = f"""{PERSONA}

请写 12 个派蒙不可能准确知道答案的问题（第 {t['k']} 组，类型：{t['kind']}），以及派蒙的回答：15~40 个汉字，派蒙口吻，
老实说不知道或没法精确回答，不给出任何具体数字或编造的细节，可以开个玩笑或提一个不需要编造的建议；每条说法不同。
只输出 JSON：[{{"q": "...", "a": "..."}}, ...]"""
    text, u = chat([{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.9)
    return {"id": t["id"], "items": pairs(text), "_tokens": u.get("completion_tokens", 0)}

def task_unknown(t):
    names = "、".join(t["names"])
    prompt = f"""{PERSONA}

下面这些名字在《原神》里并不存在：{names}
对每个名字写 1 个旅行者的问题（问是什么 / 是谁 / 在哪，问法各不相同），以及派蒙的回答：15~40 个汉字，派蒙口吻，
说自己没听说过这个名字，不要猜它是什么，可以反问旅行者是在哪听到的；每条说法不同。
只输出 JSON：[{{"q": "...", "a": "..."}}, ...]"""
    text, u = chat([{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.8)
    return {"id": t["id"], "items": pairs(text), "_tokens": u.get("completion_tokens", 0)}

def fake_names(n, seed=0):
    """transliteration-style names that appear nowhere in the game's text or the lore table"""
    rnd = random.Random(seed)
    syl = list("阿卡瑟洛维斯特拉尼亚伦尔德克芬莉丝蒂诺贝塔萨菲奥兰赫茨米雅格里欧珀伊多")
    suf = ["", "", "城", "港", "山", "镇", "之森", "遗迹", "王国", "岛"]
    corpus = []
    for f in ("data/agd/TextMapCHS.json", "data/agd/TextMap_MediumCHS.json"):
        corpus += [v for v in json.load(open(f, encoding="utf-8")).values() if isinstance(v, str)]
    corpus += [json.loads(l)["name"] for l in open("data/lore/entries.jsonl", encoding="utf-8")]
    text = "\n".join(corpus)
    out = set()
    while len(out) < n:
        base = "".join(rnd.choice(syl) for _ in range(rnd.choice((3, 3, 4))))
        name = base + rnd.choice(suf)
        if base not in text and name not in out: out.add(name)
    return sorted(out)

def main():
    topics = ["打招呼和闲聊", "问派蒙喜欢吃什么", "冒险和打怪", "问路和找东西", "心情和天气", "提瓦特的角色和地方"]
    run_tasks([{"id": f"en_{k}", "k": k, "topic": topics[k % len(topics)]} for k in range(50)], task_english, os.path.join(OUT, "r9_english.jsonl"))
    lore = [json.loads(l) for l in open(os.path.join(OUT, "lore_rag.jsonl"), encoding="utf-8")]
    random.seed(9)
    run_tasks([{"id": f"rn_{r['name']}", "r": r} for r in random.sample(lore, 170)], task_ragneg, os.path.join(OUT, "r9_ragneg.jsonl"))
    kinds = ["派蒙自己的精确数据（体重、身高、年龄、心跳）", "旅行者的私事和未来", "精确的统计数字（面积、人口、距离）",
             "游戏里不确定的未来剧情", "需要精确计算的游戏数值", "现实世界的最新消息"]
    run_tasks([{"id": f"tr_{k}", "k": k, "kind": kinds[k % len(kinds)]} for k in range(36)], task_trick, os.path.join(OUT, "r9_trick.jsonl"))
    names = fake_names(400)
    run_tasks([{"id": f"un_{i}", "names": names[i:i + 10]} for i in range(0, len(names), 10)], task_unknown, os.path.join(OUT, "r9_unknown.jsonl"))
    for f in ("r9_english", "r9_ragneg", "r9_trick", "r9_unknown"):
        rows = [json.loads(l) for l in open(os.path.join(OUT, f + ".jsonl"), encoding="utf-8")]
        print(f, sum(len(r["items"]) for r in rows), "pairs")

if __name__ == "__main__":
    main()
