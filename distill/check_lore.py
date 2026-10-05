"""Quality pass over the retrieval snippets -> rewrites data/distill/lore_rag.jsonl in place

    python distill/check_lore.py            # report in reports/lore-check.json

The teacher checks every snippet against the entry's current source text (data/lore/entries.jsonl):
correct, about the entry itself (what / where / who: not a story event), understandable on its own.
A failed snippet is replaced by the teacher's fix (characters: from the profile fields, i.e. title,
affiliation, occupation, element, weapon), and its Q&A pairs are written again from the new snippet.
Entries whose source cannot support a snippet are dropped.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, run_tasks
from gen_lore_rag import make_qa, SNIP_MAX
from gen_paimon import OUT

RAG = os.path.join(OUT, "lore_rag.jsonl")
CHECK = os.path.join(OUT, "lore_check.jsonl")

def task(t):
    r, e = t["r"], t["e"]
    prompt = f"""下面是《原神》wiki 里「{e['name']}」（{e['kind']}，{e['where'] or '无'}）的原文：
{e['text'][:500]}

游戏里玩家问到这个名字时，会把这句"资料"给一个小模型看：{r['snippet']}
请判断这句资料有没有下面三种问题（只有这三种算问题；信息不完整、措辞可以更好、少了称号元素之类都不算问题）：
A. 有事实错误，或者与原文矛盾；
B. 说的不是「{e['name']}」本身（是什么 / 在哪 / 是谁），而是原文里某段故事、事件或台词的片段；
C. 没看过原文的人完全看不懂它在说什么。
三种都没有：ok 为 true，fix 留空。有问题：ok 为 false，写一句新的资料（不超过 {SNIP_MAX} 个汉字，不要重复名字本身；
角色优先用称号、所属、职业、元素、武器）；如果原文根本说不清它是什么，fix 写空字符串。
只输出 JSON：{{"ok": true/false, "reason": "...", "fix": "..."}}"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=300, temperature=0.1)
    d = json.loads(text[text.index("{"):text.rindex("}") + 1])
    out = {"id": t["id"], "name": r["name"], "ok": bool(d.get("ok")), "reason": str(d.get("reason", ""))[:120],
           "old": r["snippet"], "fix": str(d.get("fix") or "").strip().rstrip("。")}
    if not out["ok"] and out["fix"]:
        out["items"], _ = make_qa(e, r["aliases"], out["fix"])
    return {**out, "_tokens": usage.get("completion_tokens", 0)}

def main():
    ents = {json.loads(l)["name"]: json.loads(l) for l in open("data/lore/entries.jsonl", encoding="utf-8")}
    rows = [json.loads(l) for l in open(RAG, encoding="utf-8")]
    tasks = [{"id": "ck_" + r["name"], "r": r, "e": ents[r["name"]]} for r in rows if r["name"] in ents]
    run_tasks(tasks, task, CHECK)
    chk = {c["name"]: c for c in map(json.loads, open(CHECK, encoding="utf-8"))}
    kept, fixed, dropped = [], 0, 0
    for r in rows:
        c = chk.get(r["name"])
        if c and not c["ok"]:
            if not c["fix"] or not c.get("items"): dropped += 1; continue
            r = {**r, "snippet": c["fix"], "items": c["items"]}; fixed += 1
        kept.append(r)
    with open(RAG, "w", encoding="utf-8") as f:
        for r in kept: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"checked": len(chk), "ok": sum(c["ok"] for c in chk.values()), "fixed": fixed, "dropped": dropped,
           "examples": [{k: c[k] for k in ("name", "old", "fix", "reason")} for c in chk.values() if not c["ok"]][:40]}
    json.dump(rep, open("reports/lore-check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: v for k, v in rep.items() if k != "examples"})

if __name__ == "__main__":
    main()
