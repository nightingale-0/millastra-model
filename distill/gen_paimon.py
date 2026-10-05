"""Synthesize Paimon-style chat data with the local teacher (sequence-level distillation).

    python distill/gen_paimon.py --qa 1500 --multi 400 --rewrite 1200      # resumable

Outputs (one JSON line per teacher request) in data/distill/:
  qa.jsonl       single-turn pairs: player question -> Paimon answer, by topic
  multi.jsonl    3-5 turn conversations
  rewrite.jsonl  general Q&A (minimind sft) with the answer rewritten in Paimon's voice, facts kept
Real Paimon lines from the game (data/paimon/train.jsonl) are shown as style examples.
"""
import argparse, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks, wait_ready

OUT = "data/distill"

PERSONA = """派蒙是《原神》里旅行者的向导和最好的伙伴，一个会飞的白色小不点。
- 说话活泼、直接、口语化，常用第三人称“派蒙”称呼自己（如“派蒙觉得…”“交给派蒙吧！”），也会说“我”。
- 口头禅和语气词：“欸嘿”“呜哇”“诶？”“哼！”“好耶！”“嗯嗯”“喂！”。
- 超级爱吃（甜甜花酿鸡、蜜酱胡萝卜煎肉、烤吃虎鱼、杏仁豆腐、各国美食），喜欢摩拉和宝箱。
- 最讨厌被叫“应急食品”，生气时会给别人起难听的绰号。胆子有点小，但很讲义气，总是给旅行者加油。
- 称呼玩家为“旅行者”。和旅行者一起走过蒙德、璃月、稻妻、须弥、枫丹、纳塔等国家。
- 提瓦特常识：蒙德-风神巴巴托斯（温迪），璃月-岩神摩拉克斯（钟离），稻妻-雷神（雷电将军/影），须弥-草神纳西妲（小吉祥草王），
  枫丹-水神芙宁娜，纳塔-火神玛薇卡，至冬-冰神（冰之女皇）。七种元素：风岩雷草水火冰。
- 千星奇域是旅行者可以游玩、也可以亲手创造各种奇妙关卡的地方，派蒙在这里陪旅行者冒险。"""

RULES = """要求：
1. 回答必须符合派蒙的性格和说话方式，但内容要切题、真的回答问题，不要扯到无关的剧情或人物。
2. 每条回答 10~60 个汉字，最多两三句话，口语化，不用列表、不用 Markdown、不用表情符号。
3. 现实常识（地理、科学、生活）要回答正确；派蒙不知道或不确定的事要坦白说不知道，不许编造。
4. 不要说自己是 AI、模型或助手；不要出现“作为”“我无法”这类客服腔。
5. 只输出 JSON，不要任何解释。"""

TOPICS = [
    ("自我介绍与身份", ["你是谁", "你叫什么", "你几岁了", "你从哪里来", "你是什么生物", "你会一直陪着我吗"]),
    ("和旅行者闲聊", ["打招呼", "早安晚安", "夸奖派蒙", "调侃派蒙", "问派蒙心情", "谢谢派蒙", "道别"]),
    ("情绪陪伴", ["旅行者累了", "旅行者难过", "旅行者开心", "旅行者无聊", "旅行者害怕", "旅行者生气"]),
    ("美食", ["派蒙最爱吃什么", "蒙德美食", "璃月美食", "稻妻美食", "做饭", "饿了", "请客", "零食"]),
    ("应急食品梗", ["叫派蒙应急食品", "开玩笑要吃派蒙", "派蒙好不好吃"]),
    ("冒险与带路", ["今天去哪", "迷路了", "找宝箱", "爬山", "下雨了", "天黑了", "赶路", "休息"]),
    ("提瓦特国家", ["蒙德", "璃月", "稻妻", "须弥", "枫丹", "纳塔", "至冬"]),
    ("神明与角色", ["温迪", "钟离", "雷电将军", "纳西妲", "芙宁娜", "可莉", "胡桃", "迪卢克", "安柏", "凯亚", "琴"]),
    ("元素与战斗", ["七种元素", "元素反应", "打怪物", "丘丘人", "队伍搭配", "受伤了"]),
    ("摩拉与宝箱", ["摩拉不够", "开宝箱", "派蒙想要摩拉", "省钱"]),
    ("千星奇域", ["千星奇域是什么", "在千星奇域玩什么", "自己做关卡", "邀请朋友一起玩", "关卡太难了"]),
    ("给别人起绰号", ["派蒙生气了", "被人欺负", "被嘲笑个子小"]),
    ("现实常识问答", ["地理首都", "天文", "动物", "植物", "数学计算", "物理常识", "历史人物", "节日"]),
    ("生活建议", ["睡不着", "学习", "运动", "交朋友", "天气冷", "感冒了"]),
    ("讲笑话和故事", ["讲个笑话", "讲个短故事", "猜谜语", "绕口令"]),
    ("不知道的问题", ["很专业的问题", "派蒙没去过的地方", "未来的事", "旅行者的隐私"]),
    ("游戏玩法", ["怎么变强", "抽卡", "每日委托", "树脂不够", "体力", "跳过剧情"]),
]

def style_lines(pool, k=8):
    return "\n".join("派蒙：" + s for s in random.sample(pool, k))

def load_style_pool():
    pool = []
    with open("data/paimon/train.jsonl", encoding="utf-8") as f:
        for line in f:
            conv = json.loads(line)["conversations"]
            if conv[0]["role"] != "system": continue
            for m in conv:
                if m["role"] == "assistant":
                    for s in m["content"].split("\n"):
                        if 6 <= len(s) <= 45: pool.append(s)
    return list(set(pool))

def task_qa(t, pool):
    prompt = f"""{PERSONA}

下面是派蒙在游戏里的真实台词，用来参考语气：
{style_lines(pool)}

话题：{t['topic']}（侧重：{t['seed']}）
请写 {t['n']} 组“旅行者对派蒙说的话 → 派蒙的回答”。旅行者的话要多样：提问、闲聊、请求、吐槽都可以，长短不一，像真实玩家在游戏里打字。
{RULES}
输出格式：[{{"q": "旅行者的话", "a": "派蒙的回答"}}, ...]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=1600, temperature=0.9, top_p=0.95)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    items = [x for x in items if isinstance(x, dict) and x.get("q") and x.get("a")]
    return {"id": t["id"], "topic": t["topic"], "items": items, "_tokens": usage.get("completion_tokens", 0)}

def task_multi(t, pool):
    prompt = f"""{PERSONA}

下面是派蒙在游戏里的真实台词，用来参考语气：
{style_lines(pool)}

话题：{t['topic']}（侧重：{t['seed']}）
请写一段旅行者和派蒙的连续对话，共 {t['turns']} 轮（旅行者说一句、派蒙回一句为一轮），话题自然推进，派蒙要接住旅行者上一句话。
{RULES}
输出格式：[{{"q": "旅行者", "a": "派蒙"}}, ...]（按顺序）"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=1200, temperature=0.9, top_p=0.95)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    items = [x for x in items if isinstance(x, dict) and x.get("q") and x.get("a")]
    return {"id": t["id"], "topic": t["topic"], "items": items, "_tokens": usage.get("completion_tokens", 0)}

def task_rewrite(t, pool):
    qa = "\n".join(json.dumps({"i": i, "q": q, "ref": a}, ensure_ascii=False) for i, (q, a) in enumerate(t["pairs"]))
    prompt = f"""{PERSONA}

下面是派蒙在游戏里的真实台词，用来参考语气：
{style_lines(pool, 6)}

下面每行是一个问题 q 和参考答案 ref。请让派蒙来回答这些问题：保留参考答案里正确的关键信息，但改成派蒙的口吻，压缩到 15~70 个汉字。
如果问题很长、很专业，派蒙可以只讲最关键的一两点。
{qa}
{RULES}
输出格式：[{{"i": 序号, "a": "派蒙的回答"}}, ...]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=1600, temperature=0.7, top_p=0.8)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    out = []
    for x in items:
        if isinstance(x, dict) and isinstance(x.get("i"), int) and 0 <= x["i"] < len(t["pairs"]) and x.get("a"):
            out.append({"q": t["pairs"][x["i"]][0], "a": x["a"]})
    return {"id": t["id"], "items": out, "_tokens": usage.get("completion_tokens", 0)}

def general_pairs(n, seed=1):
    """short single-turn Q&A from minimind sft (skipping the first 50k lines used by the earlier mix)."""
    # sampled over the whole file (it is grouped by topic) and kept to short, concrete questions
    rnd, out = random.Random(seed), []
    skip = ("AI", "人工智能", "量子", "伦理", "元宇宙", "区块链", "如何看待", "边界", "本质")
    with open("data/sft_t2t_mini.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 50000 or i % 7: continue
            d = json.loads(line)["conversations"]
            if len(d) != 2 or not (4 <= len(d[0]["content"]) <= 40) or len(d[1]["content"]) > 400: continue
            q = d[0]["content"]
            if any(s in q for s in skip + ("以下", "下面", "这段", "这个句子", "给定", "根据", "文章", "翻译", "生成")): continue
            if not re.search(r"[？?]$|什么|为什么|怎么|如何|吗|哪|多少|几", q) or len(re.findall(r"[a-zA-Z]", q)) > 5: continue
            out.append((d[0]["content"], d[1]["content"]))
    rnd.shuffle(out)
    return out[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=int, default=1500, help="requests of 10 pairs")
    ap.add_argument("--multi", type=int, default=400)
    ap.add_argument("--rewrite", type=int, default=1200, help="requests of 6 rewritten pairs")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    random.seed(0)
    pool = load_style_pool()
    print(f"style pool {len(pool)} lines", flush=True)
    wait_ready()
    rnd = random.Random(42)
    qa = []
    for i in range(a.qa):
        topic, seeds = rnd.choice(TOPICS)
        qa.append({"id": f"qa{i}", "topic": topic, "seed": rnd.choice(seeds), "n": 10})
    multi = []
    for i in range(a.multi):
        topic, seeds = rnd.choice(TOPICS)
        multi.append({"id": f"mt{i}", "topic": topic, "seed": rnd.choice(seeds), "turns": rnd.choice([3, 4, 5])})
    pairs = general_pairs(a.rewrite * 6)
    rw = [{"id": f"rw{i}", "pairs": pairs[i * 6:(i + 1) * 6]} for i in range(a.rewrite) if pairs[i * 6:(i + 1) * 6]]
    run_tasks(qa, lambda t: task_qa(t, pool), os.path.join(OUT, "qa.jsonl"), a.workers)
    run_tasks(multi, lambda t: task_multi(t, pool), os.path.join(OUT, "multi.jsonl"), a.workers)
    run_tasks(rw, lambda t: task_rewrite(t, pool), os.path.join(OUT, "rewrite.jsonl"), a.workers)

if __name__ == "__main__":
    main()
