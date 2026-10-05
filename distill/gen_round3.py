"""Round 3 teacher data: context-dependent multi-turn chats, a pool of probe questions for
failure-driven distillation, and a held-out multi-turn eval set.

    python distill/gen_round3.py --followup 3000 --probe 600 --eval-multi 30
Outputs in data/distill/: followup.jsonl, probe_q.jsonl, eval_multi.json
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks
from gen_paimon import PERSONA, RULES, TOPICS, style_lines, load_style_pool, OUT

FOLLOW = ["省略追问（如“那X呢？”“那另一个呢？”，必须结合上一轮才能答对）", "追问原因（“为什么？”“怎么会这样？”）",
          "追问更多（“还有吗？”“再说一个”）", "指代前文（用“它”“他”“刚才那个”“你说的那家店”）",
          "纠正或质疑派蒙（“不对吧”“你确定吗？”，派蒙要么坚持正确答案、要么承认说错）",
          "突然换话题后又绕回之前的话题", "旅行者提到自己的名字或喜好，几轮后派蒙要记得",
          "数数或计算接着上一轮的数字继续问"]

PROBE_KINDS = ["开放式问题（问派蒙的看法、感受）", "关于旅行者自己的问题（“你觉得我怎么样”“我厉害吗”）",
               "情绪表达（开心、难过、累、生气、无聊，不一定是问句）", "简短的口头语（“嗯”“哈哈”“好吧”“在吗”“你好呀”）",
               "提瓦特设定问题（人物、地点、元素、美食）", "现实常识问题（地理、科学、数学、生活）",
               "请求（讲故事、唱歌、出主意、陪聊）", "刁钻或不合理的问题（派蒙不知道、或者问题本身有错）",
               "游戏玩法和千星奇域", "调侃派蒙、开玩笑、叫派蒙应急食品"]

def task_followup(t, pool):
    prompt = f"""{PERSONA}

下面是派蒙在游戏里的真实台词，用来参考语气：
{style_lines(pool)}

话题：{t['topic']}（侧重：{t['seed']}）
请写一段旅行者和派蒙的连续对话，共 {t['turns']} 轮（旅行者一句、派蒙一句为一轮）。
重点：从第 2 轮起，旅行者的话要依赖前文才能理解，类型以「{t['follow']}」为主；派蒙必须结合前文正确回应，不能答成上一轮的答案。
{RULES}
输出格式：[{{"q": "旅行者", "a": "派蒙"}}, ...]（按顺序）"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=1400, temperature=0.9)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    items = [x for x in items if isinstance(x, dict) and x.get("q") and x.get("a")]
    return {"id": t["id"], "topic": t["topic"], "follow": t["follow"], "items": items, "_tokens": usage.get("completion_tokens", 0)}

def task_probe(t, pool):
    prompt = f"""{PERSONA}

请模仿真实玩家，写 12 句旅行者在游戏里会对派蒙说的话，类型：{t['kind']}，话题可以涉及：{t['topic']}。
要求句子之间差异大，长短不一（2~40 字），口语化，可以有错别字或省略，不要编号。
只输出 JSON 数组：["...", "..."]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=700, temperature=1.0)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    return {"id": t["id"], "kind": t["kind"], "items": [str(x).strip() for x in items if str(x).strip()], "_tokens": usage.get("completion_tokens", 0)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--followup", type=int, default=3000); ap.add_argument("--probe", type=int, default=600)
    ap.add_argument("--eval-multi", type=int, default=30)
    a = ap.parse_args()
    pool = load_style_pool()
    rnd = random.Random(7)
    def follow_tasks(prefix, n):
        out = []
        for i in range(n):
            topic, seeds = rnd.choice(TOPICS)
            out.append({"id": f"{prefix}{i}", "topic": topic, "seed": rnd.choice(seeds), "follow": rnd.choice(FOLLOW), "turns": rnd.choice([3, 4, 5, 6])})
        return out
    run_tasks(follow_tasks("fu", a.followup), lambda t: task_followup(t, pool), os.path.join(OUT, "followup.jsonl"))
    probes = [{"id": f"pq{i}", "kind": rnd.choice(PROBE_KINDS), "topic": rnd.choice(TOPICS)[0]} for i in range(a.probe)]
    run_tasks(probes, lambda t: task_probe(t, pool), os.path.join(OUT, "probe_q.jsonl"))
    # held-out multi-turn eval: last turn is the question, earlier turns are given history
    ev_path = os.path.join(OUT, "eval_multi_raw.jsonl")
    run_tasks(follow_tasks("ev", a.eval_multi), lambda t: task_followup(t, pool), ev_path)
    ev = []
    with open(ev_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if len(r["items"]) >= 3:
                h = [[x["q"], x["a"]] for x in r["items"][:-1]]
                ev.append({"q": r["items"][-1]["q"], "ref": r["items"][-1]["a"], "history": h, "topic": "多轮：" + r["follow"][:6]})
    json.dump(ev, open(os.path.join(OUT, "eval_multi.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"eval multi-turn dialogs: {len(ev)}")

if __name__ == "__main__":
    main()
