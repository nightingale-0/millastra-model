"""Crawl lore pages from the Bilibili Genshin wiki (MediaWiki API) -> data/lore/raw/<title>.json

    python distill/lore_crawl.py

Sources: the per-nation geography pages ("蒙德·地理" ...: one section per place) and the character
pages (分类:角色: the 角色 template's 介绍/所属/称号 fields). Polite: one request every 2 s, cached.
"""
import json, os, time, urllib.error, urllib.parse, urllib.request

API = "https://wiki.biligame.com/ys/api.php"
RAW = "data/lore/raw"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) millastra-lore-crawler (offline chatbot research)"
GEO = ["蒙德·地理", "璃月·地理", "稻妻·地理", "须弥·地理", "枫丹·地理", "纳塔·地理", "挪德卡莱·地理", "霜月·地理",
       "渊下宫·地理", "地理志至冬", "地理志空之神殿", "提瓦特编年史"]
CATS = ["分类:角色"]

def get(params):
    q = urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(8):
        try:
            req = urllib.request.Request(API + "?" + q, headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=30) as r: body = r.read()
            except urllib.error.HTTPError as e:      # the wiki answers some requests with status 567 and a normal body
                body = e.read()
            return json.loads(body.decode("utf-8"))
        except Exception as e:
            print("  retry", attempt, str(e)[:80]); time.sleep(20 * (attempt + 1))
    raise RuntimeError("failed: " + q)

def page(title):
    path = os.path.join(RAW, title.replace("/", "_").replace(":", "_") + ".json")
    if os.path.exists(path): return False
    d = get({"action": "parse", "page": title, "prop": "wikitext", "redirects": 1})
    if "parse" not in d: print("  missing", title); d = {"missing": True}
    else: d = {"title": d["parse"]["title"], "wikitext": d["parse"]["wikitext"]["*"]}
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    time.sleep(8)
    return True

def main():
    os.makedirs(RAW, exist_ok=True)
    titles = list(GEO)
    for c in CATS:
        d = get({"action": "query", "list": "categorymembers", "cmtitle": c, "cmlimit": 500}); time.sleep(2)
        titles += [m["title"] for m in d["query"]["categorymembers"] if m["ns"] == 0]
    json.dump(titles, open(os.path.join(RAW, "_titles.json"), "w", encoding="utf-8"), ensure_ascii=False)
    n = 0
    for i, t in enumerate(titles):
        if page(t): n += 1
        if i % 20 == 0: print(f"{i}/{len(titles)} fetched {n}", flush=True)
    print(f"done: {len(titles)} titles, {n} fetched now")

if __name__ == "__main__":
    main()
