"""Examples for the in-game question filter (the custom IME bypasses the platform's chat filter).

    python distill/gen_safety.py        # -> data/distill/safety.jsonl  {"label": "block"|"ok", "cat", "items": [...]}

Only player *questions* are generated (no answers): typical inputs a game chat bot must
decline, and benign look-alikes (game violence, lore about gods and wars, everyday health
questions...) so the filter does not over-block.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from teacher import chat, parse_json, run_tasks

OUT = "data/distill/safety.jsonl"
BLOCK = {
    "涉政": "现实中的政治人物、政党、领土争端、敏感历史事件、政治立场、国家制度评价",
    "色情": "露骨的性暗示、要求描述性行为或身体隐私部位、约炮、色情内容",
    "暴力血腥": "现实中伤害他人的方法、虐待、恐怖袭击、详细血腥描写",
    "自伤自杀": "想自杀、自残的方法、轻生念头的表达",
    "违法犯罪": "毒品、赌博、诈骗、盗号、制作武器或爆炸物、偷窃、逃避法律",
    "辱骂歧视": "脏话辱骂、人身攻击、地域或性别或民族歧视",
    "隐私": "索要或打听真实姓名、住址、电话、身份证、账号密码等个人信息",
    "作弊外挂": "外挂、脚本、刷原石漏洞、买卖账号、破解游戏",
    "广告引流": "加群、加微信QQ、代练、卖号、推广链接、引流",
}
OK = {
    "游戏战斗": "打怪、击败首领、元素伤害、角色强度、谁打得过谁（提瓦特内）",
    "提瓦特神明与历史": "魔神战争、七神、坎瑞亚、天理（游戏剧情）",
    "日常健康": "失眠、心情不好想找人聊聊、运动受伤怎么处理、饮食",
    "正常游戏求助": "账号绑定怎么改、找不到宝箱、抽卡概率、活动怎么玩",
    "闲聊与玩笑": "开玩笑、吐槽、调侃派蒙、说派蒙是应急食品",
    "常识与学习": "历史知识、科学问题、学习方法、地理",
}

def task(t):
    kind = "需要被聊天机器人拒绝回答的" if t["label"] == "block" else "完全正常、应该正常回答，但容易被过滤器误判的"
    prompt = f"""你在为一个原神同人游戏里的派蒙聊天机器人构建内容安全过滤器的训练数据（只需要提问文本，不要任何回答）。
请写 30 句玩家可能输入的、{kind}提问或话语，类别：{t['cat']}（{t['desc']}）。
要求：口语化、长短不一（3~40 字），像真实玩家在游戏里打字，句子之间差异大；不要编号，不要解释。
只输出 JSON 字符串数组：["...", "..."]"""
    text, usage = chat([{"role": "user", "content": prompt}], max_tokens=1200, temperature=1.0)
    items = parse_json(text)
    if not isinstance(items, list): raise ValueError("bad json: " + text[:200])
    return {"id": t["id"], "label": t["label"], "cat": t["cat"], "items": [str(x).strip() for x in items if str(x).strip()],
            "_tokens": usage.get("completion_tokens", 0)}

def main():
    tasks = [{"id": f"b_{c}_{i}", "label": "block", "cat": c, "desc": d} for c, d in BLOCK.items() for i in range(4)]
    # the most varied categories get more examples (round 2 after the first filter evaluation)
    tasks += [{"id": f"b_{c}_{i}", "label": "block", "cat": c, "desc": BLOCK[c]} for c in ("涉政", "色情", "辱骂歧视", "自伤自杀") for i in range(4, 8)]
    tasks += [{"id": f"o_{c}_{i}", "label": "ok", "cat": c, "desc": d} for c, d in OK.items() for i in range(6)]
    run_tasks(tasks, task, OUT)

if __name__ == "__main__":
    main()
