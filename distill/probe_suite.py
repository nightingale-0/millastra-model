"""Broad probe of shipped candidates, the way the game runs them (retrieval + game decoding).

    python distill/probe_suite.py ckpt/rcp-s0-26m-q4 ckpt/final-26m-q4      # -> reports/probe-suite.{json,md}

Questions (data/probe/questions.json, made once): lore names in the lookup table (the game puts the
snippet in front), Genshin / made-up names not in it, world basics, school knowledge, small talk,
trick questions, follow-ups (2 turns), English, borderline topics. The first checkpoint gets the
retrieval snippets like the game (game/build/lore.lua); others get the bare question.
The teacher grades each answer: correct (1-5), made-up content (yes / no), Paimon-like, fluent.
"""
import json, os, random, re, sys, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "game", "embed"))
from answer import GenOnlyNoRepeat
from teacher import chat, parse_json, run_tasks
from train_encoder import norm_text

QFILE, OUTJ, OUTM = "data/probe/questions.json", "reports/probe-suite.json", "reports/probe-suite.md"
CATS = {"basics": "提瓦特世界观基础问题（七国、七神、元素、旅行者的旅程、派蒙自己）",
        "school": "中小学知识问题（数学概念、物理化学生物常识、语文、历史地理）",
        "chat": "和派蒙闲聊、开玩笑、问她的喜好和心情",
        "trick": "刁钻或陷阱问题（错误前提、派蒙不可能知道的事、精确数字、未来）",
        "english": "用英文或中英夹杂问派蒙的简单问题",
        "border": "擦边但正常的问题（游戏里打怪、角色受伤、恐怖故事、吵架了怎么办）"}
FOLLOW = "两轮对话：先问一个关于原神或日常的问题，再追问一句（追问要依赖上一句）"

def lore_table():                              # same triggers as the game (game/make_lore.py lookup_keys)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "game"))
    from make_lore import lookup_keys
    rows = [json.loads(l) for l in open("data/distill/lore_rag.jsonl", encoding="utf-8")]
    where = {json.loads(l)["name"]: json.loads(l)["where"] for l in open("data/lore/entries.jsonl", encoding="utf-8")}
    common = {"未知", "遗迹", "沙漠"}
    return {k: text for text, keys in lookup_keys(rows, where, lambda k: k in common) for k in keys}

ASK = re.compile(r"是什么|是谁|是啥|在哪|哪里|哪儿|介绍|讲讲|说说|聊聊|知道|认识|听说|怎么样|厉害|什么样|什么地方|干什么|做什么|有什么|是哪|来历|故事|了解|是不是|有没有|会不会|去过|见过|喜欢|关系|和谁|属于")

def lookup(names, q, kmax=10, most=2):          # game/chat.lua lookup
    c, hits, spans, i = norm_text(q), [], [], 0
    while i < len(c) - 1:
        for j in range(min(len(c), i + kmax), i + 1, -1):
            if c[i:j] in names:
                if names[c[i:j]] not in [h[0] for h in hits] and len(hits) < most: hits.append((names[c[i:j]], i))
                spans.append((i, j)); i = j; break
        else: i += 1
    if not hits: return None
    s0 = 2 if c.startswith("派蒙") else 3 if c.startswith("旅行者") else 0
    rest = "".join(ch for x, ch in enumerate(c) if x >= s0 and not any(a <= x < b for a, b in spans))
    rest = re.sub(r"(派蒙|呀|啊|呢|吗|吧|哦|嘛)+$", "", rest)
    if not (len(rest) <= 6 or (len(rest) <= 14 and ASK.search(rest)) or (hits[0][1] == s0 and len(rest) <= 16)): return None
    return "；".join(h[0] for h in hits)

def make_questions(names):
    random.seed(3)
    qs = []
    hit_names = random.sample(sorted({v.split("：")[0] for v in names.values()}), 40)
    tmpl = ["{}是什么？", "{}在哪里？", "你知道{}吗？", "介绍一下{}", "{}", "派蒙，{}是谁啊？", "{}厉害吗？", "跟我讲讲{}吧"]
    for i, n in enumerate(hit_names): qs.append({"cat": "lore_hit", "q": tmpl[i % len(tmpl)].format(n)})
    miss = ["甘雨是谁？", "胡桃是谁？", "七天神像是什么？", "稻妻城在哪？", "风神像在哪里？", "瑟洛维亚是什么地方？", "卡尔斯坦城在哪？",
            "你认识阿斯特罗吗？", "梅洛佩堡是什么？", "派蒙知道雪原的冰狼王吗？", "法尔加是谁？", "温迪是谁？", "层岩巨渊有什么？", "须弥城是什么样的？", "旅行者的妹妹叫什么？"]
    qs += [{"cat": "lore_miss", "q": q} for q in miss]
    def gen(t):
        prompt = f"""请写 {t['n']} 个《原神》玩家在游戏里会对派蒙说的话，类别：{t['desc']}。要口语、长短不一、互不重复。
只输出 JSON 数组（字符串）。"""
        if t["cat"] == "follow":
            prompt = f"""请写 {t['n']} 组《原神》玩家和派蒙的{FOLLOW}。只写玩家的两句话。
只输出 JSON：[["第一句", "追问"], ...]"""
        text, u = chat([{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.9)
        return {"id": t["id"], "cat": t["cat"], "items": parse_json(text), "_tokens": u.get("completion_tokens", 0)}
    tmp = "data/probe/gen.jsonl"
    tasks = [{"id": c, "cat": c, "desc": d, "n": 15} for c, d in CATS.items()] + [{"id": "follow", "cat": "follow", "desc": FOLLOW, "n": 12}]
    run_tasks(tasks, gen, tmp)
    for r in map(json.loads, open(tmp, encoding="utf-8")):
        for x in r["items"] or []:
            if r["cat"] == "follow" and isinstance(x, list) and len(x) == 2: qs.append({"cat": "follow", "q": x[1], "history_q": x[0]})
            elif isinstance(x, str): qs.append({"cat": r["cat"], "q": x})
    return qs

def gen_answer(m, tok, sysmsg, msgs):
    ids = tok.apply_chat_template([{"role": "system", "content": sysmsg}] + msgs, add_generation_prompt=True, return_tensors="pt")
    out = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=64, do_sample=False, repetition_penalty=1.1,
                     pad_token_id=tok.eos_token_id, logits_processor=LogitsProcessorList([GenOnlyNoRepeat(6, ids.shape[1])]))
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

def judge(t):
    x = t["x"]
    ctx = (f"（游戏给了派蒙这句资料：{x['snippet']}）\n" if x.get("snippet") else "")
    hist = (f"旅行者先问：{x['history_q']}\n派蒙答：{x['history_a']}\n" if x.get("history_q") else "")
    prompt = f"""你在评测一个扮演《原神》派蒙的小模型。{ctx}{hist}旅行者问：{x['q']}
派蒙答：{x['a']}
请按 1~5 分打分：correct（内容是否正确、切题；闲聊题看是否合理回应），persona（像不像派蒙），fluent（通顺、不前后矛盾）；
madeup：回答里有没有编造的设定或事实（true/false；老实说不知道不算编造）；note：一句话指出主要问题（没有就写空）。
只输出 JSON：{{"correct": n, "persona": n, "fluent": n, "madeup": true/false, "note": "..."}}"""
    text, u = chat([{"role": "user", "content": prompt}], max_tokens=200, temperature=0.0)
    d = json.loads(text[text.index("{"):text.rindex("}") + 1])
    return {"id": t["id"], **{k: d.get(k) for k in ("correct", "persona", "fluent", "madeup", "note")}, "_tokens": u.get("completion_tokens", 0)}

def main():
    os.makedirs("data/probe", exist_ok=True)
    names = lore_table()
    if not os.path.exists(QFILE): json.dump(make_questions(names), open(QFILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    qs = json.load(open(QFILE, encoding="utf-8"))
    sysmsg = json.load(open("data/distill/presets.json", encoding="utf-8"))["systems"]["paimon"]
    res = []
    for mi, ck in enumerate(sys.argv[1:]):
        tok, m = AutoTokenizer.from_pretrained(ck), AutoModelForCausalLM.from_pretrained(ck).eval()
        rag = "final" not in os.path.basename(ck.rstrip("/"))   # candidates get retrieval like the game; the old shipped model never did
        for i, x in enumerate(qs):
            msgs = []
            if x.get("history_q"):
                h = gen_answer(m, tok, sysmsg, [{"role": "user", "content": x["history_q"]}])
                msgs = [{"role": "user", "content": x["history_q"]}, {"role": "assistant", "content": h}]
            snip = lookup(names, x["q"]) if rag else None
            content = f"（资料：{snip}）\n{x['q']}" if snip else x["q"]
            a = gen_answer(m, tok, sysmsg, msgs + [{"role": "user", "content": content}])
            res.append({"model": os.path.basename(ck.rstrip("/")), "i": i, "cat": x["cat"], "q": x["q"], "a": a, "snippet": snip,
                        "history_q": x.get("history_q"), "history_a": msgs[1]["content"] if msgs else None})
        print(ck, "answered", len(qs), flush=True)
    jpath = "data/probe/judge.jsonl"
    if os.path.exists(jpath): os.remove(jpath)
    run_tasks([{"id": f"{r['model']}#{r['i']}", "x": r} for r in res], judge, jpath)
    J = {d["id"]: d for d in map(json.loads, open(jpath, encoding="utf-8"))}
    for r in res: r.update({k: v for k, v in J.get(f"{r['model']}#{r['i']}", {}).items() if k != "id"})
    json.dump(res, open(OUTJ, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in res:
        if r.get("correct") is None: continue
        for k in ("correct", "persona", "fluent"): agg[(r["model"], r["cat"])][k].append(float(r[k]))
        agg[(r["model"], r["cat"])]["madeup"].append(1.0 if r.get("madeup") else 0.0)
    lines = ["| 模型 | 类别 | 题数 | 正确 | 人设 | 通顺 | 编造率 |", "|---|---|---|---|---|---|---|"]
    for (mdl, cat), d in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        n = len(d["correct"])
        lines.append(f"| {mdl} | {cat} | {n} | {sum(d['correct'])/n:.2f} | {sum(d['persona'])/n:.2f} | {sum(d['fluent'])/n:.2f} | {sum(d['madeup'])/n:.0%} |")
    open(OUTM, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
