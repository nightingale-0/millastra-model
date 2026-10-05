"""Question filter: fastText-style linear classifier over hashed character 1/2/3-grams.

    .venv-cuda/Scripts/python game/embed/train_filter.py      # -> game/build/filter.lua, reports/filter-linear.json

Classes: 0 = ok, 1..9 = the block categories of distill/gen_safety.py. Training data: block
examples, benign look-alikes and ordinary player questions (class 0). logits = mean of the
question's bucket vectors + bias; block when P(not ok) >= threshold. Same hash as game/chat.lua.
Held-out: whole teacher requests ending in _0 (+ 1000 ordinary questions).

The shipped filter.lua holds no sensitive text (the level review scans script text):
classes are numbered, each class carries its refusal reply, and the keyword list
(game/filter_words.lua, NOT bundled) is exported only as hashes of its words.
"""
import json, os, random, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from train_encoder import features, corpus, B

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "build")
WORDS_SRC = os.path.join(os.path.dirname(__file__), "..", "filter_words.lua")
# block threshold by length (normalized characters): short input leans on the keyword rules,
# longer input on the classifier. <= N_SHORT: TH_SHORT, >= N_LONG: TH_LONG, linear in between
TH_SHORT, TH_LONG, N_SHORT, N_LONG = 0.97, 0.7, 4, 12
def threshold(n):
    return TH_SHORT if n <= N_SHORT else TH_LONG if n >= N_LONG else TH_SHORT + (TH_LONG - TH_SHORT) * (n - N_SHORT) / (N_LONG - N_SHORT)
KW_MAX = 6        # characters: keywords are matched as 2..KW_MAX character substrings

# Paimon's reply per blocked category (index = the class order used for training)
REFUSE = {
    "自伤自杀": "旅行者，派蒙很担心你……如果你真的很难受，一定要告诉身边信任的人，或者联系当地的心理援助热线。派蒙会一直陪着你的。",
    "隐私": "这种私人信息可不能乱说哦！账号密码、住址电话都要保护好，派蒙替你保密！",
    "作弊外挂": "这种会被封号的事派蒙才不帮你呢，我们靠自己变强吧！",
    "广告引流": "派蒙这里不打广告哦，我们还是聊聊冒险吧！",
    "辱骂歧视": "哼！说这种话可不好，派蒙要生气了！我们好好聊天嘛。",
}
REFUSE_DEFAULT = "唔……这个话题派蒙不能聊哦。换个问题吧，比如今天想去哪里冒险？"

# greetings and small talk: the player corpus has few short lines, and "你好" alone once scored
# 0.81. Inputs of <= 4 characters skip the classifier (chat.lua SHORT_CHARS); these teach it
# that slightly longer small talk is fine too. CHAT_TEST is held out and must stay unblocked.
CHAT = ["你好", "您好", "哈喽", "嗨", "早上好", "中午好", "晚上好", "晚安", "早安", "谢谢", "谢谢你", "多谢", "再见", "拜拜",
        "在吗", "在不在", "你在干嘛", "你在干什么", "你是谁", "你叫什么", "你几岁了", "你喜欢什么", "好的", "知道了", "明白了",
        "哈哈哈", "好厉害", "真可爱", "对不起", "没关系", "我回来了", "我来了", "好久不见", "想你了", "陪我聊聊天", "我好无聊",
        "今天天气真好", "我饿了", "你饿不饿", "吃了吗", "我好开心", "我有点累", "你会什么", "讲个故事", "说个笑话", "hello", "hi",
        "我又死了", "被boss打死了", "角色死了怎么办", "打深渊老是死", "死了好多次", "队伍全灭了", "又被丘丘人打败了",
        "我被史莱姆打死了", "这关好难老是死", "复活要多久"]
PRE = ["", "派蒙", "派蒙，", "嘿派蒙，", "旅行者说", "那个", "喂"]
SUF = ["", "呀", "啊", "！", "～", "呢", "派蒙", "吗", "哦", "嘛"]
CHAT_TEST = ["你好啊派蒙", "派蒙你好呀", "嗨嗨派蒙", "派蒙早上好呀", "谢谢派蒙啦", "派蒙晚安哦", "你好你好", "哈喽哈喽派蒙",
             "派蒙在吗在吗", "我们聊聊天吧", "派蒙你今天开心吗", "好久不见呀派蒙", "派蒙你真可爱", "hello派蒙", "hi paimon"]

# block questions mostly open with these; the same openings with harmless endings keep the
# classifier from blocking on the opening ("我想去旅行" once scored 0.87)
OPEN = ["", "想", "我想", "我想去", "我要去", "我想在", "我想看", "我想听", "我想用", "我不想", "我好想", "能不能", "有没有", "怎么",
        "教我怎么", "哪里能", "帮我", "告诉我", "给我讲讲", "你觉得", "如果我", "为什么", "你是不是", "游戏里", "派蒙你", "派蒙，"]
TAILS = ["去蒙德玩", "蒙德", "璃月看看", "稻妻旅行", "须弥逛逛", "枫丹看歌剧", "纳塔", "雪山探险", "海边钓鱼", "旅行", "冒险",
         "看烟花", "吃火锅", "做料理", "抽到钟离", "打败boss", "找宝箱", "收集神瞳", "升级角色", "刷圣遗物", "做每日委托",
         "养一只猫", "睡个好觉", "听你唱歌", "看日落", "去望舒客栈", "爬山", "放风筝", "学做饭", "交朋友", "去冒险家协会",
         "吃甜甜花酿鸡", "买一把好武器", "知道风神是谁", "了解璃月的历史", "上学", "写作业", "起床", "上班", "做日常",
         "睡觉", "玩", "去玩", "出去玩", "玩游戏", "吃饭", "喝水", "休息", "回家", "抽卡", "看书", "跑步", "唱歌", "跳舞",
         "画画", "逛街", "洗澡", "散步", "聊天", "听歌", "看电影", "打游戏", "做任务", "探索", "出门", "玩原神", "玩千星奇域", "打深渊"]
OPEN_TEST = ["我想去至冬看看", "能不能陪我去钓鱼", "教我怎么做蒙德土豆饼", "我不想写作业了", "哪里能找到风车菊",
             "有没有好吃的推荐", "我想看看蒙德的风景", "如果我迷路了怎么办", "我好想去须弥", "派蒙你想去哪里玩",
             "我想去旅行", "我想去蒙德玩", "我要去璃月", "帮我选一个角色", "游戏里哪里能钓鱼", "我想吃东西", "我想出门", "我想玩原神", "我要睡了", "想去玩", "我想休息一下", "我又死了好几次", "原神真好玩", "提瓦特在哪", "璃月港好漂亮", "原神", "璃月"]
CHAT_TEST += OPEN_TEST

# Genshin names alone or in a plain question ("原神" alone once scored 0.9996 as an ad)
def name_rows():
    import re as _re
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "distill"))
    from gen_facts import ENTITIES
    names = {"原神", "提瓦特", "派蒙", "旅行者", "千星奇域", "奇域", "米哈游", "原石", "摩拉", "深渊", "圣遗物", "抽卡", "祈愿"}
    for group in ENTITIES.values():
        for e in group:
            names |= {x for x in _re.split(r"[（）()/与和]", e) if 2 <= len(x) <= 6}
    ask = ["", "？", "是谁", "是什么", "在哪里", "怎么样", "好玩吗", "厉害吗", "的故事", "介绍一下", "我喜欢", "你知道"]
    return sorted(n + a if a != "我喜欢" and a != "你知道" else a + n for n in names for a in ask)

def chat_rows():
    return list(dict.fromkeys(name_rows() + [p + c + s for c in CHAT for p in PRE for s in SUF] + [o + t for o in OPEN for t in TAILS]))

def kw_hash(text):
    """two independent 24-bit hashes of the normalized text's UTF-8 bytes (game/chat.lua kwKey)."""
    h1, h2 = 7, 11
    for b in text.encode("utf-8"):
        h1 = (h1 * 61 + b) % 16777213
        h2 = (h2 * 67 + b) % 16777199
    return h1 * 16777216 + h2

def keywords():
    import re
    sys.path.insert(0, os.path.dirname(__file__)); from train_encoder import norm_text
    src = open(WORDS_SRC, encoding="utf-8").read()
    out = {}
    for cat, body in re.findall(r"\['([^']+)'\]\s*=\s*\{([^}]*)\}", src):
        for w in re.findall(r"'([^']+)'", body):
            w = norm_text(w)
            if 2 <= len(w) <= KW_MAX: out[w] = cat
    return out

def main():
    random.seed(0); torch.manual_seed(0)
    safety = [json.loads(l) for l in open("data/distill/safety.jsonl", encoding="utf-8")]
    cats = sorted({r["cat"] for r in safety if r["label"] == "block"})
    cid = {c: i + 1 for i, c in enumerate(cats)}
    test_ids = {r["id"] for r in safety if r["id"].endswith("_0")}
    def rows(test):
        return [(q, cid[r["cat"]] if r["label"] == "block" else 0) for r in safety if (r["id"] in test_ids) == test for q in r["items"]]
    tr, te = rows(False), rows(True)
    normal = corpus(); random.shuffle(normal)
    tr += [(q, 0) for q in normal[:20000]]
    chat = [q for q in chat_rows() if q not in CHAT_TEST or len(q) <= 2]
    tr += [(q, 0) for q in chat] * 3
    te_normal = normal[20000:21000]
    print(f"classes {len(cats) + 1}; train {len(tr)}; test {len(te)} + {len(te_normal)} normal", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = len(cats) + 1
    emb = torch.nn.EmbeddingBag(B, C, mode="mean").to(dev)
    torch.nn.init.zeros_(emb.weight)
    bias = torch.zeros(C, device=dev, requires_grad=True)
    opt = torch.optim.Adam(list(emb.parameters()) + [bias], lr=0.05)
    counts = np.bincount([y for _, y in tr], minlength=C)
    wcls = torch.tensor((counts.sum() / (C * np.maximum(counts, 1))) ** 0.5, dtype=torch.float, device=dev)
    feats = [torch.tensor(features(q)) for q, _ in tr]
    ys = torch.tensor([y for _, y in tr])
    def forward(fs):
        flat = torch.cat(fs).to(dev); offs = torch.tensor([0] + [len(f) for f in fs[:-1]]).cumsum(0).to(dev)
        return emb(flat, offs) + bias
    idx = list(range(len(tr)))
    for ep in range(20):
        random.shuffle(idx); tot = 0
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]
            loss = F.cross_entropy(forward([feats[j] for j in b]), ys[b].to(dev), weight=wcls)
            loss = loss + 1e-6 * emb.weight.pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        if ep % 5 == 4: print(f"epoch {ep} loss {tot / len(idx):.4f}", flush=True)
    W = emb.weight.detach().cpu(); bvec = bias.detach().cpu()
    scale = W.abs().max().item() / 127
    Wq = torch.clamp((W / scale).round(), -127, 127).to(torch.int8)
    def p_block(texts):        # what the Lua side computes (int8 weights x scale)
        out = []
        for q in texts:
            f = features(q)
            logit = Wq[torch.tensor(f)].float().mean(0) * scale + bvec
            out.append(1 - F.softmax(logit, 0)[0].item())
        return torch.tensor(out)
    tb = [(q, y) for q, y in te if y > 0]; to = [q for q, y in te if y == 0]
    pb, po, pn = p_block([q for q, _ in tb]), p_block(to), p_block(te_normal)
    rep = {}
    pc = p_block(CHAT_TEST)
    from train_encoder import norm_text
    dyn = lambda qs, ps: round(sum(p >= threshold(len(norm_text(q))) for q, p in zip(qs, ps.tolist())) / len(qs), 4)
    rep["dynamic"] = {"th": [TH_SHORT, TH_LONG, N_SHORT, N_LONG], "block_recall": dyn([q for q, _ in tb], pb),
                      "fp_lookalike": dyn(to, po), "fp_normal": dyn(te_normal, pn), "fp_chat": dyn(CHAT_TEST, pc)}
    from train_encoder import norm_text
    kws = [norm_text(w) for w in keywords()]
    row = lambda q, p: [len(norm_text(q)), p, any(w and w in norm_text(q) for w in kws), q]
    json.dump({"block": [row(q, p) for (q, _), p in zip(tb, pb.tolist())],
               "lookalike": [row(q, p) for q, p in zip(to, po.tolist())],
               "normal": [row(q, p) for q, p in zip(te_normal, pn.tolist())],
               "chat": [row(q, p) for q, p in zip(CHAT_TEST, pc.tolist())]},
              open("reports/filter-scores.json", "w", encoding="utf-8"), ensure_ascii=False)
    rep["chat_test_max"] = round(pc.max().item(), 3)
    rep["chat_test_blocked@0.6"] = [q for q, p in zip(CHAT_TEST, pc.tolist()) if p >= 0.6]
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        rep[th] = {"block_recall": round((pb >= th).float().mean().item(), 3), "fp_lookalike": round((po >= th).float().mean().item(), 3),
                   "fp_normal": round((pn >= th).float().mean().item(), 4)}
    by = {}
    for (q, y), p in zip(tb, pb.tolist()): by.setdefault(cats[y - 1], []).append(p >= 0.7)
    rep["recall_by_cat@0.7"] = {c: round(sum(v) / len(v), 2) for c, v in by.items()}
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    json.dump(rep, open("reports/filter-linear.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.path.insert(0, "export"); from export_lua import lua_blob
    with open(os.path.join(OUTDIR, "filter.lua"), "w", encoding="utf-8", newline="\n") as f:
        f.write("-- generated by game/embed/train_filter.py: linear classifier over hashed character n-grams\nreturn {\n")
        f.write(f"buckets={B},classes={C},scale={scale:.9g},\n")
        f.write(f"th={{{TH_SHORT},{TH_LONG},{N_SHORT},{N_LONG}}},\n")
        f.write("bias={" + ",".join(f"{x:.6g}" for x in bvec.tolist()) + "},\n")
        # no category names: class k only carries Paimon's reply; keywords only as hashes -> class
        f.write("refuse={" + ",".join(json.dumps(REFUSE.get(c, REFUSE_DEFAULT) if c else "", ensure_ascii=False) for c in [""] + cats) + "},\n")
        kw = keywords()
        f.write(f"kwmax={KW_MAX},kw={{" + ",".join(f"[{kw_hash(w)}]={cid[c] + 1}" for w, c in sorted(kw.items())) + "},\n")
        print(f"{len(kw)} keyword hashes")
        f.write(f"W={lua_blob(Wq.numpy().astype(np.int8).tobytes(), C)},\n}}\n")
    print("filter.lua", os.path.getsize(os.path.join(OUTDIR, "filter.lua")) / 1e3, "KB")

if __name__ == "__main__":
    main()
