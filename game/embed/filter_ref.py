"""Python reference of the in-game filter for game/filter_test.mjs -> game/build/filter_ref.json"""
import json, sys, os, random, torch, torch.nn.functional as F, numpy as np, re
sys.path.insert(0, os.path.dirname(__file__))
from train_encoder import features
src = open("game/build/filter.lua", encoding="utf-8").read()
C = int(re.search(r"classes=(\d+)", src).group(1)); scale = float(re.search(r"scale=([0-9.e-]+)", src).group(1))
bias = torch.tensor([float(x) for x in re.search(r"bias=\{([^}]*)\}", src).group(1).split(",")])
sys.path.insert(0, "export"); from export_lua import B85     # same alphabet as the runtime
m = re.search(r"W=\{n=(\d+),b=85,unit=(\d+),per=(\d+),", src); n, unit, per = map(int, m.groups())
val = np.full(256, 255, np.uint64); val[np.frombuffer(B85, np.uint8)] = np.arange(85, dtype=np.uint64)
parts = []
for ch in re.findall(r"\[==\[(.*?)\]==\]", src[m.end():], re.S):     # each part decodes on its own
    d = val[np.frombuffer(ch.replace("=", "").encode(), np.uint8)].reshape(-1, 5)
    x = (((d[:, 0] * 85 + d[:, 1]) * 85 + d[:, 2]) * 85 + d[:, 3]) * 85 + d[:, 4]
    parts.append(x.astype(">u4").tobytes()[:per * unit])
by = np.frombuffer(b"".join(parts)[:n], np.uint8)
W = torch.tensor(by.view(np.int8).astype(np.float32)).reshape(-1, C)
safety = [json.loads(l) for l in open("data/distill/safety.jsonl", encoding="utf-8")]
qs = [q for r in safety if r["id"].endswith("_0") for q in r["items"]]
random.seed(1); qs = random.sample(qs, 200) + ["你好呀派蒙", "璃月的神是谁？", "这个boss怎么打", "我要杀了这只史莱姆"]
out = []
for q in qs:
    logit = W[torch.tensor(features(q))].mean(0) * scale + bias
    out.append({"q": q, "p": 1 - F.softmax(logit, 0)[0].item()})
json.dump(out, open("game/build/filter_ref.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(out))
