"""Which lore entries does the shipped model already know? -> data/lore/known.json (game/make_lore.py skips them)

    python distill/known_entries.py ckpt/r10-s0-26m-q4

Retrieval only helps where the model does not know the entry: for 钟离 / 温迪 / 甘雨 the bare model
answered "真实身份 / 真名" right (岩神摩拉克斯, 风神巴巴托斯) and the snippet ("往生堂客卿") pulled it
off. Each entry is asked "X是谁？/ X是什么？" without a snippet (game decoding); the teacher checks the
answer against the wiki text. Correct and specific -> known -> no trigger keys. Re-run per model.
"""
import json, os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
sys.path.insert(0, os.path.dirname(__file__))
from answer import GenOnlyNoRepeat
from teacher import chat, run_tasks

OUT, CACHE = "data/lore/known.json", "data/lore/known_judge.jsonl"

def main():
    ck = sys.argv[1]
    ents = {json.loads(l)["name"]: json.loads(l) for l in open("data/lore/entries.jsonl", encoding="utf-8")}
    rows = [json.loads(l) for l in open("data/distill/lore_rag.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["name"] in ents and not (ents[r["name"]]["where"] or "").startswith("提瓦特编年史")]
    sysmsg = json.load(open("export/presets.json", encoding="utf-8"))["systems"]["paimon"]
    tok, m = AutoTokenizer.from_pretrained(ck), AutoModelForCausalLM.from_pretrained(ck).eval()
    tasks = []
    for r in rows:
        name = r["name"].strip("「」")
        q = f"{name}是谁？" if ents[r["name"]]["kind"] == "角色" else f"{name}是什么？"
        ids = tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt")
        out = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=60, do_sample=False, repetition_penalty=1.1,
                         pad_token_id=tok.eos_token_id, logits_processor=LogitsProcessorList([GenOnlyNoRepeat(6, ids.shape[1])]))
        a = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        tasks.append({"id": r["name"], "q": q, "a": a, "src": ents[r["name"]]["text"][:400]})
    def judge(t):
        prompt = f"""《原神》wiki 原文：{t['src']}
问：{t['q']}
小模型答：{t['a']}
这个回答是否正确且具体地说出了它是什么 / 是谁（与原文或《原神》公开设定一致，没有错误内容；只说"不知道"或说得很空泛都算否）？
只输出 JSON：{{"known": true/false}}"""
        text, u = chat([{"role": "user", "content": prompt}], max_tokens=30, temperature=0.0)
        return {"id": t["id"], "known": '"known": true' in text.replace(" ", "").replace(":true", ": true") or "true" in text.lower(), "a": t["a"],
                "_tokens": u.get("completion_tokens", 0)}
    if os.path.exists(CACHE): os.remove(CACHE)
    run_tasks(tasks, judge, CACHE)
    res = [json.loads(l) for l in open(CACHE, encoding="utf-8")]
    known = sorted(r["id"] for r in res if r["known"])
    json.dump({"model": os.path.basename(ck.rstrip("/")), "known": known}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(known)} of {len(res)} entries known by {ck} -> {OUT}: {known[:30]}")

if __name__ == "__main__":
    main()
