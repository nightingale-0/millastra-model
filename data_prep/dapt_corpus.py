"""Domain corpus for continued pretraining (DAPT) -> data/dapt/game.jsonl ({"text": ...} per document)

    python data_prep/dapt_corpus.py

The student learned lore only from Q&A pairs, so it mixes names and facts up. This is the game's own
text, so names appear in context:
  - dialogue: chains through nextDialogs, one document per chain, "说话人：台词" lines
  - other text: books, item / place / character descriptions, readables (TextMap strings not used in
    dialogue), stat descriptions dropped
  - wiki entries (data/lore/entries.jsonl: geography sections, character profiles)
"""
import collections, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from paimon import AGD, PAIMON, clean, load_textmap

OUT = "data/dapt/game.jsonl"
MIX, VAL = "data/dapt/train.jsonl", "data/dapt/val.jsonl"
GENERAL = 60000          # general chats mixed in against forgetting (data/sft_t2t_mini.jsonl, lines >= 600k)
STAT = re.compile(r"\d+(\.\d+)?%|攻击力|生命值上限|暴击|元素精通|冷却时间|持续\d|秒内|层数|伤害提升|获得\d|×\d|\d+点")

def main():
    random.seed(0)
    tm = load_textmap()
    look = lambda h: None if h is None else (tm.get(str(h)) or tm.get(str(int(h) ^ 512)))
    dlg = json.load(open(os.path.join(AGD, "DialogExcelConfigData.json"), encoding="utf-8"))
    npc_name = {str(n["id"]): look(n.get("nameTextMapHash")) or ""
                for n in json.load(open(os.path.join(AGD, "NpcExcelConfigData.json"), encoding="utf-8"))}
    idkey = next(k for k, v in dlg[0].items() if isinstance(v, int) and k not in ("talkContentTextMapHash", "talkTitleTextMapHash", "talkRoleNameTextMapHash"))
    nodes, prev, used = {}, collections.defaultdict(list), set()
    for d in dlg:
        i, role = d[idkey], d.get("talkRole") or {}
        raw = look(d.get("talkContentTextMapHash"))
        if raw: used.add(raw)
        rtype, rid = role.get("type", ""), str(role.get("id", ""))
        if rtype == "TALK_ROLE_PLAYER": name = "旅行者"
        elif rtype == "TALK_ROLE_NPC" and rid == PAIMON: name = "派蒙"
        else: name = clean(look(d.get("talkRoleNameTextMapHash")) or npc_name.get(rid, "")) or ""
        nodes[i] = {"text": clean(raw), "name": name, "next": d.get("nextDialogs") or []}
        for n in nodes[i]["next"]: prev[n].append(i)
    docs, seen = [], set()
    for i in nodes:                                   # chain starts: no predecessor
        if prev.get(i) or i in seen: continue
        lines, cur = [], i
        while cur in nodes and cur not in seen and len(lines) < 80:
            seen.add(cur); n = nodes[cur]
            if n["text"]: lines.append(f"{n['name']}：{n['text']}" if n["name"] else n["text"])
            cur = n["next"][0] if n["next"] else None
        if len(lines) >= 2: docs.append({"text": "\n".join(lines), "src": "dialog"})
    n_dialog = len(docs)
    other = set()
    for v in tm.values():
        if not isinstance(v, str) or v in used: continue
        s = clean(v)
        if not s or len(s) < 12 or STAT.search(s): continue
        if len(re.findall(r"[一-鿿]", s)) < len(s) * 0.5: continue
        other.add(s)
    other = sorted(other); random.shuffle(other)
    buf = []                                          # short strings packed into ~600-character documents
    for s in other:
        if len(s) >= 300: docs.append({"text": s, "src": "text"}); continue
        buf.append(s)
        if sum(map(len, buf)) >= 600: docs.append({"text": "\n".join(buf), "src": "text"}); buf = []
    if buf: docs.append({"text": "\n".join(buf), "src": "text"})
    lore = "data/lore/entries.jsonl"
    if os.path.exists(lore):
        for l in open(lore, encoding="utf-8"):
            r = json.loads(l)
            docs.append({"text": f"{r['name']}（{r['where']}）：{r['text']}" if r["where"] else f"{r['name']}：{r['text']}", "src": "wiki"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for d in docs: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    by = collections.Counter(d["src"] for d in docs); chars = collections.Counter()
    for d in docs: chars[d["src"]] += len(d["text"])
    print({k: f"{by[k]} docs / {chars[k] / 1e6:.1f}M chars" for k in by}, "->", OUT)
    # training mix: game documents (2% held out for validation) + general chats
    random.shuffle(docs)
    nval = len(docs) // 50
    gen = []
    with open("data/sft_t2t_mini.jsonl", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i < 600000: continue
            d = json.loads(l)["conversations"]
            if len(d) == 2 and len(d[1]["content"]) <= 400: gen.append({"conversations": d})
            if len(gen) >= GENERAL: break
    mix = [{"text": d["text"]} for d in docs[nval:]] + gen
    random.shuffle(mix)
    with open(MIX, "w", encoding="utf-8") as f:
        for x in mix: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(VAL, "w", encoding="utf-8") as f:
        for x in docs[:nval]: f.write(json.dumps({"text": x["text"]}, ensure_ascii=False) + "\n")
    print(f"{len(docs) - nval} game docs + {len(gen)} general chats -> {MIX}; {nval} -> {VAL}")

if __name__ == "__main__":
    main()
