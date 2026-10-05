"""Lore snippets for retrieval + training pairs that use them -> data/distill/lore_rag.jsonl

    python distill/gen_lore_rag.py

Per wiki entry (data/lore/entries.jsonl) the teacher writes, from the source text only:
  - aliases players may type ("芬德尼尔" for "沙尔·芬德尼尔")
  - one snippet of at most 16 characters: what the game puts in front of the question when a name matches
    (every extra token costs ~0.26 s of prefill on the client, so snippets stay short)
  - 3 questions about the entry and Paimon's answers that use only the snippet's facts
The training sample is the user turn "（资料：名字：snippet）\\n问题" (game/chat.lua lookup uses the same format).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks
from gen_paimon import PERSONA, OUT

SRC = "data/lore/entries.jsonl"
SNIP_MAX = 16

def rag_prompt(name, snippet):
    return f"（资料：{name}：{snippet}）\n"

def task(t):
    e = t["e"]
    prompt = f"""{PERSONA}

下面是《原神》wiki 里「{e['name']}」（{e['kind']}，{e['where'] or '无'}）的原文：
{e['text'][:500]}

请只根据这段原文（不要用你自己的记忆补充）输出：
1. aliases：玩家可能用来指代它的其他叫法（简称、去掉前缀的名字等，没有就空列表，不要写原文没出现过的称呼）
2. snippet：一句话概括它最重要的事实，不超过 {SNIP_MAX} 个汉字，不要重复名字本身，要让没看过原文的人也能看懂
只输出 JSON：{{"aliases": [...], "snippet": "..."}}"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=300, temperature=0.3)
    try: d = json.loads(text[text.index("{"):text.rindex("}") + 1])     # an object: parse_json would pick the aliases list
    except ValueError: d = None
    if not isinstance(d, dict) or not d.get("snippet"): raise ValueError("bad json: " + text[:200])
    snip = str(d["snippet"]).strip().rstrip("。")[:SNIP_MAX + 4]
    aliases = [a for a in (d.get("aliases") or []) if isinstance(a, str) and 2 <= len(a) <= 10 and a != e["name"]]
    qa, n2 = make_qa(e, aliases, snip)
    return {"id": t["id"], "name": e["name"], "kind": e["kind"], "aliases": aliases, "snippet": snip, "items": qa,
            "_tokens": usage.get("completion_tokens", 0) + n2}

def make_qa(e, aliases, snip):
    """second call sees only the snippet: answers cannot carry facts the game will not show the model"""
    prompt = f"""{PERSONA}

旅行者问派蒙关于「{e['name']}」{('（也叫：' + '、'.join(aliases) + '）') if aliases else ''}的问题。派蒙知道的只有这一句：{snip}
请写 3 个旅行者可能问的问题（一个只写名字，一个"是什么/是谁/在哪"之类，一个更具体的问法；都要包含它的名字或别名），
以及派蒙的回答：15~40 个汉字，派蒙口吻，开头直接回答，只用这句资料里的事实，绝不添加资料外的内容；
这句话回答不了的部分就坦白说不清楚。三个回答的说法要各不相同，不要都用"派蒙只知道这些"；
不要提到"资料""这句话"之类的字眼，就像派蒙自己知道一样说出来。
只输出 JSON：[{{"q": "...", "a": "..."}}, ...]"""
    text, u2 = chat([{"role": "user", "content": prompt}], max_tokens=600, temperature=0.7)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    qa = [{"q": x["q"], "a": x["a"]} for x in items if isinstance(x, dict) and x.get("q") and x.get("a")]
    return qa, u2.get("completion_tokens", 0)

def main():
    ents = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    ents = [e for e in ents if len(e["text"]) >= 20 and "." not in e["name"]]
    run_tasks([{"id": "lr_" + e["name"], "e": e} for e in ents], task, os.path.join(OUT, "lore_rag.jsonl"))
    rows = [json.loads(l) for l in open(os.path.join(OUT, "lore_rag.jsonl"), encoding="utf-8")]
    print(f"{len(rows)} entries, {sum(len(r['items']) for r in rows)} pairs, "
          f"mean snippet {sum(len(r['snippet']) for r in rows) / max(1, len(rows)):.1f} chars")

if __name__ == "__main__":
    main()
