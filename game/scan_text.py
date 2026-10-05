"""Scan a built client script for text the level review could flag ("脚本文件内包含违规文本").

    python game/scan_text.py game/out/scripts/paimon-chat-full.main.lua

Reports: every distinct plain CJK run (should be UI strings and Paimon's replies only),
any keyword-list word or block-category name in plain text, and occurrences of common
English profanity (case-insensitive) — e.g. spelled by chance inside the base85 weight data.
Exit code 1 if anything is found.
"""
import re, sys

ENGLISH = ["fuck", "shit", "porn", "sex", "rape", "nazi", "dick", "cunt", "bitch", "nigg", "slut", "whore", "pussy", "cock", "kill", "drug"]
CATEGORIES = ["涉政", "色情", "暴力血腥", "自伤自杀", "违法犯罪", "辱骂歧视", "作弊外挂", "广告引流"]

def main(path):
    s = open(path, encoding="utf-8-sig").read()
    words = set(re.findall(r"'([^']+)'", open("game/filter_words.lua", encoding="utf-8").read())) - set(CATEGORIES)
    bad = 0
    runs = sorted(set(re.findall(r"[一-鿿][一-鿿，。？！、：；“”‘’（）…—～·\w\s%]*", s)), key=len, reverse=True)
    print(f"{len(runs)} distinct plain CJK runs:")
    for r in runs: print("   ", r.strip()[:80])
    for w in sorted(words | set(CATEGORIES)):
        if len(w) >= 2 and not w.isascii() and w in s: print("  PLAIN BLOCKED TEXT:", w); bad += 1
    low = s.lower()
    for w in ENGLISH:
        n = low.count(w)
        if n: print(f"  english '{w}': {n}"); bad += 1
    print("clean" if not bad else f"{bad} problems")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main(sys.argv[1])
