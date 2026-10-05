"""World basics + bare-name inputs (round 8).

    python distill/gen_basics.py            # -> data/distill/basics.jsonl (teacher)

The student answered "提瓦特" alone with "派蒙只知道蒙德……别的就不太清楚啦" and "提瓦特是什么" with
"提瓦特是七国之一": the most basic world facts were missing from facts.jsonl (no 提瓦特 / 原神
entity), and a name typed on its own was never a training input. Two task kinds:
  - core world facts (CORE, anchored by CANON so the teacher does not drift), several phrasings each
  - a name on its own ("提瓦特" / "提瓦特？" / "说说提瓦特"): Paimon recognizes it and says one sure fact
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks
from gen_paimon import PERSONA, OUT
from gen_facts import ENTITIES

CANON = """必须与以下设定一致：
- 提瓦特是《原神》的整个世界（大陆），七个国家都在提瓦特上，不是七国之一。
- 七国与七神：蒙德（风神巴巴托斯/温迪）、璃月（岩神摩拉克斯/钟离）、稻妻（雷神/雷电将军，影）、须弥（草神纳西妲）、
  枫丹（水神芙宁娜）、纳塔（火神玛薇卡）、至冬（冰神，冰之女皇）。
- 《原神》是米哈游开发的开放世界冒险游戏；旅行者从异世界来到提瓦特，寻找失散的血亲（哥哥或妹妹）；派蒙是旅行者在提瓦特遇到的向导。
- 神之眼是被神明认可者获得的外置魔力器官，可以使用一种元素力；七种元素是风、岩、雷、草、水、火、冰。
- 天空岛是神明居住的地方；深渊教团是旅行者的血亲所在的敌对组织；坎瑞亚是五百年前覆灭的古国。
- 千星奇域是《原神》里玩家自己创作和游玩关卡的玩法。"""

CORE = ["提瓦特", "《原神》这款游戏", "七国", "七神", "神之眼", "七种元素", "天空岛", "深渊教团", "坎瑞亚", "旅行者的旅程目标",
        "派蒙和旅行者的相遇", "千星奇域"]

def task_core(t):
    prompt = f"""{PERSONA}

{CANON}

请列出关于「{t['name']}」的 {t['n']} 条最基础、你百分之百确定的事实（拿不准的不要写）。
每条事实写 4 种旅行者可能的不同问法（长短都要有，包括很口语的、只有几个字的），以及派蒙的回答：
回答 15~50 个汉字，派蒙口吻，开头直接说出这条事实，不要添加不确定的细节。
只输出 JSON：[{{"fact": "...", "qa": [{{"q": "...", "a": "..."}}, ...]}}, ...]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=2500, temperature=t["temp"])
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    pairs = [{"q": x["q"], "a": x["a"]} for f in items if isinstance(f, dict)
             for x in (f.get("qa") or []) if isinstance(x, dict) and x.get("q") and x.get("a")]
    return {"id": t["id"], "kind": "core", "name": t["name"], "items": pairs, "_tokens": usage.get("completion_tokens", 0)}

def task_bare(t):
    names = "、".join(t["names"])
    prompt = f"""{PERSONA}

{CANON}

旅行者有时只打一个词就发过来。对下面每个词，写旅行者的 3 种输入（就是这个词本身；这个词加问号；
"说说/讲讲/介绍一下"加这个词），和派蒙的回答：回答 15~45 个汉字，派蒙口吻，先认出这是什么（如「提瓦特就是我们冒险的这个世界呀」），
再说一条最确定的基础事实；不要说"不知道""不清楚"，也不要编造不确定的细节。
词：{names}
只输出 JSON：[{{"name": "...", "qa": [{{"q": "...", "a": "..."}}, ...]}}, ...]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=3000, temperature=t["temp"])
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    pairs = [{"q": x["q"], "a": x["a"]} for f in items if isinstance(f, dict)
             for x in (f.get("qa") or []) if isinstance(x, dict) and x.get("q") and x.get("a")]
    return {"id": t["id"], "kind": "bare", "items": pairs, "_tokens": usage.get("completion_tokens", 0)}

def bare_names():
    import re
    names = ["提瓦特", "原神", "七国", "七神", "神之眼", "元素", "天空岛", "深渊教团", "坎瑞亚", "派蒙", "旅行者", "千星奇域"]
    for group in ENTITIES.values():
        for e in group:
            n = re.split(r"[（(]", e)[0]
            if 2 <= len(n) <= 8 and n not in names and n != "派蒙自己": names.append(n)
    return names

def main():
    out = os.path.join(OUT, "basics.jsonl")
    tasks = [{"id": f"bc_{n}_{v}", "name": n, "n": 6, "temp": temp, "kind": "core"} for n in CORE for v, temp in enumerate((0.3, 0.7, 1.0))]
    names = bare_names()
    for v, temp in enumerate((0.4, 0.9)):
        for i in range(0, len(names), 8):
            tasks.append({"id": f"bb_{i}_{v}", "names": names[i:i + 8], "temp": temp, "kind": "bare"})
    run_tasks(tasks, lambda t: task_core(t) if t["kind"] == "core" else task_bare(t), out)
    n = sum(len(json.loads(l)["items"]) for l in open(out, encoding="utf-8"))
    print(f"{len(names)} names; {n} pairs -> {out}")

if __name__ == "__main__":
    main()
