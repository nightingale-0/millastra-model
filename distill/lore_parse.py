"""data/lore/raw/*.json (wiki text) -> data/lore/entries.jsonl: {name, kind, where, text}

    python distill/lore_parse.py

Geography pages: one entry per section (==区域== / ===地点=== / ====...====), text = the section's
own prose with wiki markup removed. Character pages: the 角色 template (名称/称号/所属/种族/介绍/职业).
"""
import glob, json, os, re

RAW, OUT = "data/lore/raw", "data/lore/entries.jsonl"

def strip(t):
    t = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"\[\[(?:文件|File|Image):[^\]]*\]\]", "", t)
    t = re.sub(r"\{\{黑幕\|([^{}]*)\}\}", r"\1", t)
    t = re.sub(r"\{\{注音\|([^|{}]*)\|[^{}]*\}\}", r"\1", t)
    t = re.sub(r"\{\{#info:[^{}]*\}\}", "", t)
    for _ in range(3): t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t)
    t = re.sub(r"<br\s*/?>", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"'''?", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    return "\n".join(l.strip() for l in t.split("\n") if l.strip() and not l.strip().startswith(("|", "{", "}", "*", "__")))

def geo(title, text):
    where = title.replace("·地理", "").replace("地理志", "")
    out, path = [], []
    for m in re.finditer(r"^(={2,5})\s*([^=\n]+?)\s*\1\s*$\n(.*?)(?=^={2,5}[^=]|\Z)", text, re.M | re.S):
        level, name, body = len(m.group(1)), m.group(2).strip(), strip(m.group(3))
        path = path[:level - 2] + [name]
        if name == "简介" and level == 2 and where and "编年史" not in title:   # the region itself (挪德卡莱, 蒙德…)
            if len(body) >= 8: out.append({"name": where, "kind": "地区", "where": "", "text": body[:600]})
            continue
        if name in ("简介", "参考资料", "注释", "相关", "画廊", "其他"): continue
        if len(body) < 8: continue
        out.append({"name": name, "kind": "地点", "where": where + ("·" + "·".join(path[:-1]) if len(path) > 1 else ""), "text": body[:600]})
    return out

def character(title, text):
    m = re.search(r"\{\{角色\n(.*?)\n\}\}", text, re.S)
    if not m: return []
    f = dict(re.findall(r"^\|([^=\n]+)=(.*)$", m.group(1), re.M))
    g = lambda k: strip(f.get(k, "")).replace("\n", "")
    info = re.search(r"\{\{角色/信息\n(.*?)\n\}\}", text, re.S)
    fi = dict(re.findall(r"^\|([^=\n]+)=(.*)$", info.group(1), re.M)) if info else {}
    parts = [f"称号「{g('称号')}」" if g("称号") else "", f"全名{g('全名')}" if g("全名") and g("全名") != g("名称") else "",
             f"所属{g('所属')}" if g("所属") else "", f"职业{strip(fi.get('职业', ''))}" if fi.get("职业") else "",
             f"{g('元素属性')}元素" if g("元素属性") else "", f"武器{g('武器类型')}" if g("武器类型") else "", g("介绍")]
    return [{"name": g("名称") or title, "kind": "角色", "where": g("所属"), "text": "；".join(p for p in parts if p)}]

def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        if os.path.basename(p).startswith("_"): continue
        d = json.load(open(p, encoding="utf-8"))
        if d.get("missing"): continue
        t, w = d["title"], d["wikitext"]
        rows += geo(t, w) if ("地理" in t or t == "提瓦特编年史") else character(t, w)
    seen, out = set(), []
    for r in rows:
        if r["name"] in seen: continue
        seen.add(r["name"]); out.append(r)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} entries ({sum(r['kind'] == '地点' for r in out)} places, {sum(r['kind'] == '角色' for r in out)} characters) -> {OUT}")

if __name__ == "__main__":
    main()
