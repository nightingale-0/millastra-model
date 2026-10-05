"""Teacher outputs (data/distill/*.jsonl) -> SFT set data/distill/{train,val}.jsonl + held-out eval questions.

    python distill/build_sft.py

Mix (all Paimon samples carry the Paimon system prompt):
  synthesized single-turn Q&A, multi-turn chats, general Q&A rewritten in Paimon's voice,
  a few short game-script exchanges where the player talks to Paimon directly,
  and plain general Q&A without a system prompt (keeps the base model's knowledge).
"""
import collections, json, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_prep"))
from paimon import SYSTEM

D = "data/distill"
# R7=1: the round-7 data exactly (no answer clean-up, no round-8 sets) -> train_r7.jsonl / val_r7.jsonl,
# for ablations against the shipped model (DAPT: base -> continued pretraining -> this SFT set)
R7 = os.environ.get("R7") == "1"
# RAG=1: add retrieval samples (distill/gen_lore_rag.py): the user turn carries "（资料：名字：snippet）"
# in front of the question, as game/chat.lua sends it when a lore name matches
RAG = os.environ.get("RAG") == "1"
# BASICS=1 (with R7=1): also the world-basics / bare-name set (gen_basics.py), without the round-8 clean-up
BASICS = os.environ.get("BASICS") == "1"
# round 9 (with R7=1): SET_A=1 adds sets that teach right answers (school facts, English questions, a
# snippet attached to a question that only mentions the name); SET_B=1 adds the honest-"don't know" sets
# (beyond-scope, exact numbers, made-up names). Kept apart to see whether B makes Paimon hedge elsewhere.
SET_A, SET_B = os.environ.get("SET_A") == "1", os.environ.get("SET_B") == "1"
# MULTI=1 (round 10): two snippets in one question, and yes/no questions about one entry (gen_multi_rag.py)
MULTI = os.environ.get("MULTI") == "1"
SUFFIX = ("_r7" if R7 else "") + ("_rag" if RAG else "") + ("_basics" if R7 and BASICS else "") + ("_A" if SET_A else "") + ("_B" if SET_B else "") + ("_M" if MULTI else "")
N_EVAL = 60
GAME_MAX = 3000
GENERAL_SHARE = 0.2          # of the final set

BAD = re.compile(r"AI|人工智能|语言模型|作为一个|作为派蒙|助手|抱歉，?我|我无法|\*\*|^\s*\d+\.|#|https?://|[\U0001F300-\U0001FAFF]")

def ok_answer(a):
    a = a.strip()
    return 4 <= len(a) <= 120 and re.search(r"[一-鿿]", a) and not BAD.search(a)

def ok_question(q):
    q = q.strip()
    return 1 <= len(q) <= 120 and not BAD.search(q)

def read(name):
    p = os.path.join(D, name)
    if not os.path.exists(p): return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# Round 8 answer clean-up: patterns the student over-learned from the teacher data
#   - an interjection opener on almost every answer (欸？/诶？/欸嘿/呜哇/嗯…: 38% of eval answers)
#   - bragging tails after a fact ("派蒙记得可清楚了"), which the student then attaches to wrong facts
#   - food in unrelated answers (甜甜花酿鸡 in 14% of eval answers, "我想休息一下" -> let's eat)
#   - character traits leaking to other characters ("迪卢克老是忘带摩拉": 钟离's)
# only the surprised "欸？/诶？" opener is thinned (it reads odd when nothing is surprising); 欸嘿 /
# 嗯… / 呜哇 / 哼 are Paimon's flavour and stay (round 8 thinned them all: persona 4.43 -> 4.19)
OPENER = re.compile(r"^(欸|诶)[？?]+\s*")
OPENER_KEEP = 0.3             # share of those openers kept
FOOD_DROP = 0.5               # share of unrelated food mentions removed
BRAG = re.compile(r"派蒙(记得|知道得?)?(可|超|最)?清楚|清清楚楚|不会记错|才不会数错|滚瓜烂熟|派蒙说得没错吧|派蒙可是(最棒的向导|很清楚)")
FOOD = re.compile(r"甜甜花酿鸡|烤吃虎鱼|蜜酱胡萝卜煎肉|甜甜花")
FOOD_Q = re.compile(r"吃|饿|食|菜|饭|料理|好吃|厨|鸡|鱼|请客|零食|点心|甜|喝|味|饱|馋|做饭|餐")
SENT = re.compile(r"[^。！？!?]*[。！？!?]+|[^。！？!?]+$")
clean_stats = {"opener": 0, "brag": 0, "food": 0, "trait": 0}

def clean_answer(q, a):
    """None = drop the sample."""
    if R7: return a
    rnd = random.Random(q + a)
    if re.search(r"忘带(摩拉|钱)|老忘带|总是忘记带钱", a) and "钟离" not in q + a:
        clean_stats["trait"] += 1; return None
    sents = SENT.findall(a)
    if len(sents) > 1:
        keep = [s for s in sents if not BRAG.search(s)]
        if len(keep) < len(sents) and "".join(keep).strip(): clean_stats["brag"] += 1; sents = keep
        if FOOD.search(a) and not FOOD_Q.search(q) and rnd.random() < FOOD_DROP:
            keep = [s for s in sents if not FOOD.search(s)]
            if len("".join(keep)) >= 8: clean_stats["food"] += 1; sents = keep
            # too little left without it: keep the sample as it is
    a = "".join(sents).strip()
    m = OPENER.match(a)
    if m and len(a) - m.end() >= 6 and rnd.random() > OPENER_KEEP:
        clean_stats["opener"] += 1; a = a[m.end():]
    return a

def sys_conv(turns):
    conv = [{"role": "system", "content": SYSTEM}]
    for q, a in turns:
        a = clean_answer(q.strip(), a.strip())
        if a is None: return None
        conv += [{"role": "user", "content": q.strip()}, {"role": "assistant", "content": a}]
    return {"conversations": conv}

# the game's suggestion menu (game/build.mjs SUGGEST): cached with first-step logits, so picking one
# answers without any prefill
SUGGEST = ["你是谁？", "派蒙，我们今天去哪儿？", "你饿了吗？", "给我讲个笑话吧。", "璃月的神是谁？", "七种元素都有哪些？"]

def main():
    random.seed(0)
    seen, single, stats = set(), [], {}
    for r in read("qa.jsonl"):
        for x in r["items"]:
            q, a = str(x["q"]), str(x["a"])
            if not (ok_question(q) and ok_answer(a)) or q in seen: continue
            seen.add(q); single.append((q, a, r.get("topic", "")))
    random.shuffle(single)
    eval_q = [{"q": q, "topic": t, "ref": a} for q, a, t in single[:N_EVAL]]
    qa = [sys_conv([(q, a)]) for q, a, _ in single[N_EVAL:]]
    stats["qa"] = len(qa)

    multi = []
    for r in read("multi.jsonl"):
        turns = [(str(x["q"]), str(x["a"])) for x in r["items"]]
        if len(turns) >= 2 and all(ok_question(q) and ok_answer(a) for q, a in turns):
            multi.append(sys_conv(turns))
    stats["multi"] = len(multi)

    rewrite = []
    for r in read("rewrite.jsonl"):
        for x in r["items"]:
            q, a = str(x["q"]), str(x["a"])
            if ok_question(q) and ok_answer(a) and q not in seen:
                seen.add(q); rewrite.append(sys_conv([(q, a)]))
    stats["rewrite"] = len(rewrite)

    # round 3: context-dependent chats + teacher's ideal answers to the probe questions
    held = {q["q"] for q in eval_q}
    if os.path.exists(os.path.join(D, "eval_multi.json")):
        held |= {q["q"] for q in json.load(open(os.path.join(D, "eval_multi.json"), encoding="utf-8"))}
    followup = []
    for r in read("followup.jsonl"):
        turns = [(str(x["q"]), str(x["a"])) for x in r["items"]]
        if len(turns) >= 2 and all(ok_question(q) and ok_answer(a) for q, a in turns) and turns[-1][0] not in held:
            followup.append(sys_conv(turns))
    stats["followup"] = len(followup)
    corr, weak = [], 0
    import glob
    corr_files = ["corrections.jsonl"] + sorted(os.path.relpath(f, D) for f in glob.glob(os.path.join(D, "r*", "corrections.jsonl")))
    for r in (row for cf in corr_files for row in read(cf)):
        for x in r["items"]:
            q, a = str(x["q"]), str(x["better"])
            if not (ok_question(q) and ok_answer(a)) or q in held or q in seen: continue
            seen.add(q); corr.append(sys_conv([(q, a)]))
            if min(x["relevance"], x["correct"], x["persona"]) <= 3:         # the student failed: count twice
                corr.append(sys_conv([(q, a)])); weak += 1
    stats["corrections"] = len(corr); stats["corrections_weak"] = weak; stats["correction_files"] = len(corr_files)

    # knowledge drilling (gen_facts.py): several phrasings per fact, plus exact local arithmetic
    facts = []
    for name in ("facts.jsonl", "facts_world.jsonl") + (() if R7 else ("school.jsonl",)):
        for r in read(name):
            for x in r["items"]:
                q, a = str(x["q"]), str(x["a"])
                if ok_question(q) and ok_answer(a) and q not in held and q not in seen:
                    seen.add(q); facts.append(sys_conv([(q, a)]))
    # world basics and bare-name inputs (gen_basics.py), twice: few pairs, and the most asked
    for r in ([] if R7 and not BASICS else read("basics.jsonl")):
        for x in r["items"]:
            q, a = str(x["q"]), str(x["a"])
            if any(w in a for w in ("不是同一个地方", "说不准", "不知道", "不清楚")): continue   # 天空岛 answers contradict; unsure ones out
            if ok_answer(a) and q not in held: facts += [sys_conv([(q, a)])] * 2
    if RAG:
        n_rag = 0
        for r in read("lore_rag.jsonl"):
            ctx = f"（资料：{r['name'].strip('「」')}：{r['snippet']}）\n"
            for x in r["items"]:
                q, a = str(x["q"]).strip(), str(x["a"]).strip()
                if re.match(r"^「?无」?", q) or "资料" in a or not ok_answer(a): continue
                facts.append(sys_conv([(ctx + q, a)])); n_rag += 1
        stats["rag"] = n_rag
    r9 = collections.Counter()
    def add(name, rows, prefix=False):
        for x in rows:
            q, a = str(x["q"]).strip(), str(x["a"]).strip()
            if prefix and x.get("snippet"): q = f"（资料：{x['snippet']}）\n" + q
            if ok_answer(a) and q not in held: facts.append(sys_conv([(q, a)])); r9[name] += 1
    if R7 and SET_A:
        add("school", [x for r in read("school.jsonl") if r["kind"] == "school" for x in r["items"]])
        add("english", [x for r in read("r9_english.jsonl") for x in r["items"]])
        add("ragneg", [x for r in read("r9_ragneg.jsonl") for x in r["items"]], prefix=True)
    if R7 and SET_B:
        add("hard", [x for r in read("school.jsonl") if r["kind"] == "hard" for x in r["items"]])
        add("trick", [x for r in read("r9_trick.jsonl") for x in r["items"]])
        add("unknown", [x for r in read("r9_unknown.jsonl") for x in r["items"]])
    if R7 and MULTI:
        add("multirag", [x for r in read("r10_multirag.jsonl") for x in r["items"]], prefix=True)
    if r9: stats["r9"] = dict(r9)
    p = os.path.join(D, "math.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            if x["q"] not in held: facts.append(sys_conv([(x["q"], x["a"])]))
    stats["facts"] = len(facts)
    corr += facts

    game = []
    with open("data/paimon/train.jsonl", encoding="utf-8") as f:
        for line in f:
            conv = json.loads(line)["conversations"]
            if conv[0]["role"] != "system" or len(conv) != 3: continue
            q, a = conv[1]["content"], conv[2]["content"]
            if "：" in q or "\n" in q or "\n" in a or len(a) > 40 or len(q) > 40: continue   # player -> Paimon only, short
            game.append(sys_conv([(q, a)]))
    random.shuffle(game)
    game = game[:GAME_MAX]
    stats["game"] = len(game)

    paimon = [x for x in qa + multi + rewrite + game + followup + corr if x]   # clean_answer may drop samples
    stats['cleaned'] = dict(clean_stats)
    n_general = int(len(paimon) * GENERAL_SHARE / (1 - GENERAL_SHARE))
    general = []
    with open("data/sft_t2t_mini.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 200000: continue                      # away from the rewrite source lines
            d = json.loads(line)["conversations"]
            if len(d) == 2 and len(d[1]["content"]) <= 300:
                general.append({"conversations": [{"role": m["role"], "content": m["content"]} for m in d]})
            if len(general) >= n_general: break
    stats["general"] = len(general)

    data = paimon + general
    random.shuffle(data)
    nval = max(300, len(data) // 50)
    with open(os.path.join(D, f"val{SUFFIX}.jsonl"), "w", encoding="utf-8") as f:
        for s in data[:nval]: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(D, f"train{SUFFIX}.jsonl"), "w", encoding="utf-8") as f:
        for s in data[nval:]: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(D, "eval_questions.json"), "w", encoding="utf-8") as f:
        json.dump(eval_q, f, ensure_ascii=False, indent=1)
    # Lua prefix-cache presets with the same system prompt as training
    presets = {"systems": {"default": "You are a helpful assistant", "paimon": SYSTEM},
               "prompts": [{"system": "default", "user": "你好，介绍一下你自己"}] +
                          [{"system": "paimon", "user": q} for q in SUGGEST]}
    with open(os.path.join(D, "presets.json"), "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=1)
    print(stats, f"train {len(data) - nval} val {nval} eval questions {len(eval_q)}")
    for s in random.sample(paimon, min(6, len(paimon))):
        print("  ", " | ".join(f"{m['role'][0]}: {m['content']}" for m in s["conversations"][1:]))

if __name__ == "__main__":
    main()
