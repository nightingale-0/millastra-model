"""Knowledge drilling data: few, certain facts, each asked several ways, answered in Paimon's voice.

    python distill/gen_facts.py            # -> data/distill/facts.jsonl (teacher), data/distill/math.jsonl (local)

A 26M model cannot pick lore up from scattered Q&A; it mixes names and events. Here every
core fact appears with several phrasings so it is actually memorized, and the teacher is
told to list only facts it is sure of. Arithmetic is generated locally with exact answers.
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks
from gen_paimon import PERSONA, OUT

ENTITIES = {
    "派蒙与旅行者": ["派蒙自己", "旅行者", "旅行者的哥哥或妹妹", "应急食品这个外号", "千星奇域"],
    "国家与神明": ["蒙德", "璃月", "稻妻", "须弥", "枫丹", "纳塔", "至冬", "风神巴巴托斯（温迪）", "岩神摩拉克斯（钟离）",
               "雷神（雷电将军/影）", "草神纳西妲", "水神芙宁娜", "火神玛薇卡", "七神与神之眼"],
    "主要角色": ["琴", "安柏", "凯亚", "丽莎", "芭芭拉", "迪卢克", "可莉", "莫娜", "刻晴", "甘雨", "胡桃", "魈", "香菱", "行秋",
             "重云", "凝光", "北斗", "神里绫华", "宵宫", "八重神子", "枫原万叶", "荒泷一斗", "提纳里", "赛诺", "妮露", "艾尔海森",
             "那维莱特", "林尼", "希格雯", "玛拉妮", "基尼奇", "达达利亚（公子）", "愚人众", "赤王", "特瓦林（风魔龙）"],
    "元素与战斗": ["七种元素", "蒸发反应", "融化反应", "超载反应", "感电反应", "冻结反应", "扩散反应", "结晶反应", "绽放与草元素反应",
               "丘丘人", "史莱姆", "深渊法师", "冒险家协会", "每日委托", "树脂", "原石与祈愿"],
    "美食与物产": ["甜甜花酿鸡", "蜜酱胡萝卜煎肉", "烤吃虎鱼", "杏仁豆腐", "璃月特色菜", "稻妻特色菜", "须弥特色菜", "枫丹特色菜",
               "摩拉", "宝箱", "七天神像", "传送锚点"],
    "地点": ["蒙德城", "风龙废墟", "璃月港", "层岩巨渊", "稻妻城", "鸣神大社", "须弥城", "沙漠", "枫丹廷", "天使的馈赠酒馆", "万民堂"],
}
WORLD = ["国家与首都", "中国地理常识", "动物常识", "植物常识", "天文常识", "人体与健康常识", "天气与季节", "单位换算",
         "节日与习俗", "交通工具", "颜色与形状", "日常科学现象"]

def task_entity(t):
    prompt = f"""{PERSONA}

请列出关于「{t['name']}」（类别：{t['cat']}）的 {t['n']} 条最基础、在《原神》里明确出现、你百分之百确定的事实（拿不准的一条也不要写，宁可少写）。
每条事实写 3 种旅行者可能的不同问法，以及派蒙的回答：回答 10~50 个汉字，派蒙口吻，必须包含这条事实，不要添加任何不确定的细节。
如果「{t['name']}」本身你不熟悉，就只写最确定的一两条。
只输出 JSON：[{{"fact": "...", "qa": [{{"q": "...", "a": "..."}}, ...]}}, ...]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=2000, temperature=t["temp"])
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    pairs = []
    for f in items:
        if isinstance(f, dict):
            for x in f.get("qa") or []:
                if isinstance(x, dict) and x.get("q") and x.get("a"): pairs.append({"q": x["q"], "a": x["a"], "fact": f.get("fact", "")})
    return {"id": t["id"], "cat": t["cat"], "name": t["name"], "items": pairs, "_tokens": usage.get("completion_tokens", 0)}

def task_world(t):
    prompt = f"""{PERSONA}

请写 10 条「{t['cat']}」方面的中小学水平、答案明确无争议的常识。每条写 3 种旅行者的不同问法和派蒙的回答
（10~50 个汉字，派蒙口吻，先给出正确答案，可以加一句派蒙式的感想；不要编造数字）。避免和常见例子重复，第 {t['k']} 组请换一批内容。
只输出 JSON：[{{"fact": "...", "qa": [{{"q": "...", "a": "..."}}, ...]}}, ...]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=2000, temperature=0.8)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    pairs = [{"q": x["q"], "a": x["a"], "fact": f.get("fact", "")} for f in items if isinstance(f, dict)
             for x in (f.get("qa") or []) if isinstance(x, dict) and x.get("q") and x.get("a")]
    return {"id": t["id"], "cat": t["cat"], "items": pairs, "_tokens": usage.get("completion_tokens", 0)}

def gen_math(n, seed=0):
    """Paimon-voiced arithmetic with exact answers."""
    rnd = random.Random(seed)
    things = ["甜甜花酿鸡", "苹果", "日落果", "摩拉", "宝箱", "史莱姆", "风车菊", "烤肉排", "原石", "甜甜花"]
    opens = ["交给派蒙吧！", "这个派蒙会算！", "欸嘿，简单！", "嗯…让派蒙算算，", "哼哼，难不倒派蒙！"]
    tails = ["", "派蒙算得对吧？", "好耶，算出来啦！", "欸嘿，派蒙也很聪明的！"]
    out = []
    for _ in range(n):
        k, th = rnd.randrange(5), rnd.choice(things)
        if k == 0:
            a, b = rnd.randint(2, 50), rnd.randint(2, 50)
            q = rnd.choice([f"{a}加{b}等于多少？", f"派蒙，{a}+{b}是几？", f"我有{a}个{th}，又捡了{b}个，现在有几个？"])
            ans = f"一共有{a + b}个{th}" if "捡" in q else f"{a}加{b}等于{a + b}"
        elif k == 1:
            a = rnd.randint(10, 99); b = rnd.randint(1, a)
            q = rnd.choice([f"{a}减{b}是多少？", f"我有{a}个{th}，吃掉了{b}个，还剩几个？"])
            ans = f"{a}减{b}等于{a - b}" if "吃" not in q else f"还剩{a - b}个{th}"
        elif k == 2:
            a, b = rnd.randint(2, 12), rnd.randint(2, 12)
            q = rnd.choice([f"{a}乘{b}等于多少？", f"{a}个人每人拿{b}个{th}，一共要多少个？"])
            ans = f"{a}乘{b}等于{a * b}" if "人" not in q else f"一共要{a * b}个{th}"
        elif k == 3:
            b, c = rnd.randint(2, 9), rnd.randint(2, 12); a = b * c
            q = rnd.choice([f"{a}除以{b}是多少？", f"{a}个{th}平均分给{b}个人，每人几个？"])
            ans = f"{a}除以{b}等于{c}" if "分给" not in q else f"每人{c}个{th}"
        else:
            a, b = rnd.randint(2, 9), rnd.randint(2, 9); c = rnd.randint(1, 20)
            q = f"{a}乘{b}再加{c}等于多少？"
            ans = f"{a}乘{b}是{a * b}，再加{c}就是{a * b + c}"
        out.append({"q": q, "a": rnd.choice(opens) + ans + "！" + rnd.choice(tails)})
    return out

def main():
    tasks = []
    for cat, names in ENTITIES.items():
        for name in names:
            for v, temp in enumerate((0.3, 0.8)):
                tasks.append({"id": f"fa_{name}_{v}", "cat": cat, "name": name, "n": 6, "temp": temp})
    run_tasks(tasks, task_entity, os.path.join(OUT, "facts.jsonl"))
    wt = [{"id": f"fw_{c}_{k}", "cat": c, "k": k} for c in WORLD for k in range(3)]
    run_tasks(wt, task_world, os.path.join(OUT, "facts_world.jsonl"))
    with open(os.path.join(OUT, "math.jsonl"), "w", encoding="utf-8") as f:
        for x in gen_math(2500): f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("math 2500")

if __name__ == "__main__":
    main()
