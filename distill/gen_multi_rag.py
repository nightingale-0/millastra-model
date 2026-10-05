"""Retrieval with two snippets / yes-no questions about one entry -> data/distill/r10_multirag.jsonl

    python distill/gen_multi_rag.py

game/chat.lua lookup attaches up to 2 snippets ("（资料：A：…；B：…）"); the round-8 student had only seen
one and mixed the two entries up ("刻晴…冰元素单手剑"). Two kinds:
  - pair: a question naming two entries (relation, comparison, has one been to the other); the answer
    keeps them apart and uses only the two snippets, says what the snippets do not tell
  - yesno: "X 是不是 / 有没有 / 去过…吗" about one entry: answer yes / no when the snippet says so,
    otherwise say what Paimon does know, without guessing
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks
from gen_paimon import PERSONA, OUT

def task(t):
    if t["kind"] == "pair":
        (a, sa), (b, sb) = t["e"]
        prompt = f"""{PERSONA}

派蒙知道的只有这两句：{a}：{sa}；{b}：{sb}
请写 2 个旅行者同时提到「{a}」和「{b}」的问题（比如两者的关系、比较、一方是否去过另一方、谁更厉害），以及派蒙的回答：
20~50 个汉字，派蒙口吻，开头直接回应，分清两者各是什么，只用这两句里的事实；这两句推不出来的部分坦白说不确定，不要猜。
不要提到"资料"二字。只输出 JSON：[{{"q": "...", "a": "..."}}, ...]"""
        snip = f"{a}：{sa}；{b}：{sb}"
    else:
        a, sa = t["e"][0]
        prompt = f"""{PERSONA}

派蒙知道的只有这一句：{a}：{sa}
请写 3 个以「{a}」开头的是非问句（"{a}是不是…""{a}有没有…""{a}去过…吗"之类，有的能用这句话回答、有的不能），
以及派蒙的回答：15~40 个汉字，派蒙口吻。能回答的明确说是或不是并给出依据；回答不了的说派蒙只知道它的哪些情况，不要猜。
不要提到"资料"二字。只输出 JSON：[{{"q": "...", "a": "..."}}, ...]"""
        snip = f"{a}：{sa}"
    text, u = chat([{"role": "user", "content": prompt}], max_tokens=800, temperature=0.8)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    items = [{"q": str(x["q"]), "a": str(x["a"]), "snippet": snip} for x in items if isinstance(x, dict) and x.get("q") and x.get("a")]
    return {"id": t["id"], "kind": t["kind"], "items": items, "_tokens": u.get("completion_tokens", 0)}

def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "game"))
    from make_lore import lookup_keys
    rows = [json.loads(l) for l in open(os.path.join(OUT, "lore_rag.jsonl"), encoding="utf-8")]
    where = {json.loads(l)["name"]: json.loads(l)["where"] for l in open("data/lore/entries.jsonl", encoding="utf-8")}
    ents = [tuple(text.split("：", 1)) for text, _ in lookup_keys(rows, where)]
    region = {n: (where.get(n) or n).split("·")[0] for n, _ in ents}
    random.seed(10)
    by = {}
    for e in ents: by.setdefault(region[e[0]], []).append(e)
    tasks = []
    for k in range(260):                               # related pairs mostly (same region), some across regions
        if k % 4 and len(by.get(region[ents[k % len(ents)][0]], [])) > 1:
            pool = by[region[ents[k % len(ents)][0]]]
            pair = random.sample(pool, 2)
        else:
            pair = random.sample(ents, 2)
        tasks.append({"id": f"mp_{k}", "kind": "pair", "e": pair})
    for k, e in enumerate(random.sample(ents, 200)):
        tasks.append({"id": f"yn_{k}", "kind": "yesno", "e": [e]})
    out = os.path.join(OUT, "r10_multirag.jsonl")
    run_tasks(tasks, task, out)
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    for kind in ("pair", "yesno"):
        print(kind, sum(len(r["items"]) for r in rows if r["kind"] == kind), "pairs")

if __name__ == "__main__":
    main()
