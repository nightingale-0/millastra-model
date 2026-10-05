"""Build a Paimon chat SFT set from the game's dialogue tables (AnimeGameData).

    python data_prep/paimon.py            # -> data/paimon/{train,val}.jsonl + stats

Source: data/agd/DialogExcelConfigData.json (every dialogue line: id, nextDialogs,
talkRole, text hash) + TextMap(_Medium)CHS.json. Paimon = NPC 1005. About half of
the dialogue text hashes only match the TextMap with bit 9 flipped (see look());
the remaining gaps are filled from the genshin-voice transcriptions
(data/genshin/result.json, keyed by dialog id).
Dialogue chains are followed through nextDialogs; Paimon's lines become the
assistant, the player's lines the user, other NPCs' lines user text prefixed with
their name. Consecutive lines of the same side are merged. One sample per Paimon
reply, with up to HISTORY earlier exchanges as context. A share of general
chat data (minimind sft) is mixed in to keep general ability.
"""
import json, os, random, re, collections

AGD = "data/agd"
OUT = "data/paimon"
PAIMON = "1005"
HISTORY = 2                 # earlier (user, assistant) exchanges kept as context
GENERAL_RATIO = 0.35        # general samples per Paimon sample
SYSTEM = "你是派蒙，旅行者最好的伙伴和向导。你说话活泼可爱，喜欢用第三人称“派蒙”称呼自己，爱吃好吃的，偶尔有点小脾气。回答简短口语化。"

random.seed(0)

def load_textmap():
    tm = {}
    for f in ("TextMapCHS.json", "TextMap_MediumCHS.json"):
        with open(os.path.join(AGD, f), encoding="utf-8") as fd:
            tm.update(json.load(fd))
    return tm

GENDER = [(re.compile(r"\{M#([^}]*)\}\{F#[^}]*\}"), r"\1"), (re.compile(r"\{F#[^}]*\}\{M#([^}]*)\}"), r"\1"),
          (re.compile(r"\{PLAYERAVATAR#SEXPRO\[([A-Z_]+)\|[A-Z_]+\]\}"), r"<\1>"),
          (re.compile(r"\{MATEAVATAR#SEXPRO\[[A-Z_]+\|([A-Z_]+)\]\}"), r"<\1>")]   # traveler = Aether, sibling = Lumine
PRONOUN = {"INFO_MALE_PRONOUN_HE": "他", "INFO_FEMALE_PRONOUN_SHE": "她", "INFO_MALE_PRONOUN_BROTHER": "哥哥",
           "INFO_FEMALE_PRONOUN_SISTER": "妹妹", "INFO_MALE_PRONOUN_BOY": "男孩", "INFO_FEMALE_PRONOUN_GIRL": "女孩"}

def clean(s):
    if not s: return None
    s = s.replace("\\n", "\n")
    if s.startswith("#"): s = s[1:]
    for rx, rep in GENDER: s = rx.sub(rep, s)
    s = re.sub(r"<(INFO_[A-Z_]+)>", lambda m: PRONOUN.get(m.group(1), "你"), s)
    s = s.replace("{NICKNAME}", "旅行者").replace("$UNRELEASED", "")
    s = re.sub(r"\{REALNAME\[ID\(1\)(\|HOSTONLY\(true\))?\]\}", "流浪者", s)
    s = re.sub(r"</?color[^>]*>|</?i>|</?b>", "", s)
    if "{" in s or "<" in s or "$" in s: return None          # unresolved placeholder
    s = re.sub(r"[ \t]+", " ", s).strip()
    s = re.sub(r"\n+", "\n", s)
    if len(s) < 2 or not re.search(r"[一-鿿]", s): return None
    return s

def main():
    os.makedirs(OUT, exist_ok=True)
    tm = load_textmap()
    dlg = json.load(open(os.path.join(AGD, "DialogExcelConfigData.json"), encoding="utf-8"))
    def look(h):
        # about half of the dialogue hashes in this dump have bit 9 flipped vs the TextMap key
        # (checked against the voice transcriptions: 28464 / 28481 identical)
        if h is None: return None
        return tm.get(str(h)) or tm.get(str(int(h) ^ 512))
    npc_name = {str(n["id"]): look(n.get("nameTextMapHash")) or ""
                for n in json.load(open(os.path.join(AGD, "NpcExcelConfigData.json"), encoding="utf-8"))}
    voice = {}
    for v in json.load(open("data/genshin/result.json", encoding="utf-8")).values():
        if v.get("language") != "Chinese" or not v.get("transcription"): continue
        for vc in v.get("voiceConfigs") or []:
            if vc.get("gameTrigger") == "Dialog": voice[int(vc["gameTriggerArgs"])] = v["transcription"]
    idkey = next(k for k, v in dlg[0].items() if isinstance(v, int) and k not in ("talkContentTextMapHash", "talkTitleTextMapHash", "talkRoleNameTextMapHash"))
    nodes, prev = {}, collections.defaultdict(list)
    names = collections.Counter()
    for d in dlg:
        i = d[idkey]
        role = d.get("talkRole") or {}
        text = clean(look(d.get("talkContentTextMapHash")) or voice.get(i))
        name = look(d.get("talkRoleNameTextMapHash")) or npc_name.get(str(role.get("id", "")), "")
        rtype, rid = role.get("type", ""), str(role.get("id", ""))
        if rtype == "TALK_ROLE_NPC" and rid == PAIMON: side = "assistant"
        elif rtype == "TALK_ROLE_PLAYER": side = "player"
        else: side = "npc"
        nodes[i] = {"text": text, "side": side, "name": clean(name) or "", "next": d.get("nextDialogs") or []}
        for n in nodes[i]["next"]: prev[n].append(i)
        if side == "assistant": names[name] += 1
    print(f"dialog lines {len(nodes)}, Paimon lines {sum(1 for n in nodes.values() if n['side'] == 'assistant')} (names {names.most_common(3)})")

    def back(i, limit=12):
        """linear chain of lines ending at i (follow the first predecessor)."""
        chain = [i]
        seen = {i}
        while len(chain) < limit and prev.get(chain[-1]):
            p = prev[chain[-1]][0]
            if p in seen: break
            seen.add(p); chain.append(p)
        return chain[::-1]

    def fwd(i, limit=4):
        """Paimon lines directly following i (to merge a multi-line reply)."""
        out, cur = [], i
        while len(out) < limit:
            nx = nodes[cur]["next"]
            if len(nx) != 1 or nx[0] not in nodes or nodes[nx[0]]["side"] != "assistant": break
            cur = nx[0]; out.append(cur)
        return out

    samples, seen_keys = [], set()
    for i, n in nodes.items():
        if n["side"] != "assistant" or not n["text"]: continue
        ps = prev.get(i, [])
        if ps and all(nodes[p]["side"] == "assistant" for p in ps if p in nodes): continue   # middle of a Paimon run
        chain = [c for c in back(i) if c in nodes] + fwd(i)
        # to (role, text) turns, merging consecutive lines of the same side
        turns = []
        for c in chain:
            nd = nodes[c]
            if not nd["text"]: continue
            role = "assistant" if nd["side"] == "assistant" else "user"
            text = nd["text"] if nd["side"] != "npc" or not nd["name"] else f"{nd['name']}：{nd['text']}"
            if turns and turns[-1][0] == role: turns[-1][1] += "\n" + text
            else: turns.append([role, text])
        while turns and turns[-1][0] != "assistant": turns.pop()
        while turns and turns[0][0] != "user": turns.pop(0)
        if len(turns) < 2: continue
        turns = turns[-(2 * HISTORY + 2):]
        if turns[0][0] != "user": turns = turns[1:]
        reply = turns[-1][1]
        if len(reply) > 200 or len(turns[-2][1]) > 300: continue
        key = (turns[-2][1], reply)
        if key in seen_keys: continue
        seen_keys.add(key)
        samples.append({"conversations": [{"role": "system", "content": SYSTEM}] +
                        [{"role": r, "content": t} for r, t in turns]})
    print(f"Paimon samples {len(samples)}; reply chars {sum(len(s['conversations'][-1]['content']) for s in samples)}")

    # general chat mix (short answers preferred)
    general, want = [], int(len(samples) * GENERAL_RATIO)
    with open("data/sft_t2t_mini.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            conv = [{"role": m["role"], "content": m["content"]} for m in d["conversations"]]
            if len(conv) != 2 or len(conv[1]["content"]) > 300: continue
            general.append({"conversations": conv})
            if len(general) >= want * 4: break
    general = random.sample(general, min(want, len(general)))
    data = samples + general
    random.shuffle(data)
    nval = max(200, len(data) // 50)
    with open(os.path.join(OUT, "val.jsonl"), "w", encoding="utf-8") as f:
        for s in data[:nval]: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for s in data[nval:]: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"general {len(general)}; train {len(data) - nval}, val {nval}")
    for s in random.sample(samples, 5):
        print("----")
        for m in s["conversations"][1:]: print(f"[{m['role']}] {m['content']}")

if __name__ == "__main__":
    main()
