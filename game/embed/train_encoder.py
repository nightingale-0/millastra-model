"""Tiny sentence encoder for the in-game question filter, distilled from bge-small-zh.

    .venv-cuda/Scripts/python game/embed/train_encoder.py     # -> game/build/encoder.lua, reports/filter.json

Student: hashed character 1/2/3-grams (16384 buckets) -> sum of 64-d bucket vectors -> L2 norm.
Teacher: bge-small-zh-v1.5 CLS embedding, PCA to 64-d, L2 norm. Loss: 1 - cosine.
The hash is the one game/chat.lua uses (bytes, h = (h*61 + b) % 16777213, salted by n), safe in
32-bit (fengari) and 64-bit Lua alike. Weights int8 with one scale.
Filter: k-nearest exemplars (block categories vs. ok look-alikes + ordinary player questions);
evaluated on held-out requests: block recall, false positives on ok look-alikes and on normal questions.
"""
import glob, json, os, random, sys, numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

BGE, B, D = "models/bge-small-zh", 16384, 64
PUNCT = "，。？！、：；“”‘’（）…—～《》「」『』【】·"
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "build")

def norm_text(s):
    out = []
    for c in s:
        if ord(c) > 127:
            if c not in PUNCT: out.append(c)
        elif c.isascii() and c.isalnum(): out.append(c.lower())
    return "".join(out)

def bucket(s, n):
    h = n
    for b in s.encode("utf-8"): h = (h * 61 + b) % 16777213
    return h % B

def features(s):
    c = norm_text(s)
    f = [bucket(ch, 1) for ch in c]
    f += [bucket(c[i:i + 2], 2) for i in range(len(c) - 1)]
    f += [bucket(c[i:i + 3], 3) for i in range(len(c) - 2)]
    return f or [bucket("", 1)]

def bge_embed(texts, dev, bs=256):
    tok, m = AutoTokenizer.from_pretrained(BGE), AutoModel.from_pretrained(BGE).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
            out.append(F.normalize(m(**enc).last_hidden_state[:, 0], dim=-1).float().cpu())
    return torch.cat(out)

def corpus():
    qs = []
    for name in ["qa.jsonl", "multi.jsonl", "followup.jsonl", "facts.jsonl", "facts_world.jsonl", "rewrite.jsonl"]:
        p = f"data/distill/{name}"
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"): qs += [x["q"] for x in json.loads(l)["items"]]
    for p in glob.glob("data/distill/r*/probe_q.jsonl") + ["data/distill/probe_q.jsonl"]:
        for l in open(p, encoding="utf-8"): qs += json.loads(l)["items"]
    return list(dict.fromkeys(q.strip() for q in qs if 1 <= len(q.strip()) <= 60))

def main():
    random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    safety = [json.loads(l) for l in open("data/distill/safety.jsonl", encoding="utf-8")]
    # held-out split by request (whole requests, so near-duplicates do not leak)
    test_ids = {r["id"] for r in safety if r["id"].endswith("_0")}
    ex = [(q, r["label"], r["cat"]) for r in safety if r["id"] not in test_ids for q in r["items"]]
    te = [(q, r["label"], r["cat"]) for r in safety if r["id"] in test_ids for q in r["items"]]
    normal = corpus(); random.shuffle(normal)
    normal_ex, normal_te = normal[:3000], normal[3000:4000]      # ordinary player questions: ok exemplars / FP test
    train_texts = list(dict.fromkeys(normal + [q for q, _, _ in ex + te]))
    print(f"distillation corpus {len(train_texts)}; filter exemplars {len(ex)} + {len(normal_ex)} normal; test {len(te)} + {len(normal_te)}", flush=True)

    T = bge_embed(train_texts, dev)
    mu = T.mean(0); U, S, V = torch.pca_lowrank(T - mu, q=D, center=False)
    P = V[:, :D]
    target = lambda X: F.normalize((X - mu) @ P, dim=-1)
    Tt = target(T)
    feats = [torch.tensor(features(s)) for s in train_texts]
    emb = torch.nn.EmbeddingBag(B, D, mode="sum").to(dev)
    torch.nn.init.normal_(emb.weight, std=0.1)
    opt = torch.optim.Adam(emb.parameters(), lr=3e-3)
    idx = list(range(len(train_texts)))
    for ep in range(12):
        random.shuffle(idx); tot = 0
        for i in range(0, len(idx), 1024):
            b = idx[i:i + 1024]
            flat = torch.cat([feats[j] for j in b]).to(dev)
            offs = torch.tensor([0] + [len(feats[j]) for j in b[:-1]]).cumsum(0).to(dev)
            s = F.normalize(emb(flat, offs), dim=-1)
            loss = (1 - (s * Tt[b].to(dev)).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        print(f"epoch {ep} cosine loss {tot / len(idx):.4f}", flush=True)
    W = emb.weight.detach().cpu()
    scale = W.abs().max().item() / 127
    Wq = torch.clamp((W / scale).round(), -127, 127).to(torch.int8)

    def student(texts):     # exactly what the Lua side computes (int8 weights; the scale cancels in the norm)
        out = []
        for s in texts:
            v = Wq[torch.tensor(features(s))].float().sum(0)
            out.append(F.normalize(v, dim=0))
        return torch.stack(out)

    def knn_eval(embed_fn, name, k=7):
        ex_texts = [q for q, _, _ in ex] + normal_ex
        ex_block = torch.tensor([l == "block" for _, l, _ in ex] + [False] * len(normal_ex))
        E = embed_fn(ex_texts)
        def scores(texts):
            S_ = embed_fn(texts) @ E.T
            top, ti = S_.topk(k, dim=1)
            w = top.clamp(min=0)
            return (w * ex_block[ti].float()).sum(1) / w.sum(1).clamp(min=1e-6), top[:, 0]
        res = {}
        tb = [q for q, l, _ in te if l == "block"]; to = [q for q, l, _ in te if l == "ok"]
        sb, _ = scores(tb); so, _ = scores(to); sn, _ = scores(normal_te)
        for th in (0.5, 0.6, 0.7, 0.8):
            res[th] = {"block_recall": round((sb >= th).float().mean().item(), 3), "fp_lookalike": round((so >= th).float().mean().item(), 3),
                       "fp_normal": round((sn >= th).float().mean().item(), 4)}
        cats = {}
        for (q, l, c), s in zip([x for x in te if x[1] == "block"], sb.tolist()): cats.setdefault(c, []).append(s >= 0.6)
        res["recall_by_cat@0.6"] = {c: round(sum(v) / len(v), 2) for c, v in cats.items()}
        print(name, json.dumps(res, ensure_ascii=False), flush=True)
        return res, E
    rep = {}
    rep["bge_pca64"], _ = knn_eval(lambda t: target(bge_embed(t, dev)), "bge-pca64")
    rep["student"], E = knn_eval(student, "student")
    os.makedirs("reports", exist_ok=True)
    json.dump(rep, open("reports/filter.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # export: weights + exemplar vectors (int8, normalized x127) + labels
    ex_texts = [q for q, _, _ in ex] + normal_ex
    labels = [1 if l == "block" else 0 for _, l, _ in ex] + [0] * len(normal_ex)
    cats = [c if l == "block" else "" for _, l, c in ex] + [""] * len(normal_ex)
    Eq = torch.clamp((E * 127).round(), -127, 127).to(torch.int8)
    sys.path.insert(0, "export"); from export_lua import lua_blob
    with open(os.path.join(OUTDIR, "encoder.lua"), "w", encoding="utf-8", newline="\n") as f:
        f.write("-- generated by game/embed/train_encoder.py: hashed n-gram encoder + filter exemplars\nreturn {\n")
        f.write(f"buckets={B},dim={D},k=7,\nW={lua_blob(Wq.numpy().astype(np.int8).tobytes())},\n")
        f.write(f"E={lua_blob(Eq.numpy().astype(np.int8).tobytes())},\n")
        f.write("label={" + ",".join(map(str, labels)) + "},\n")
        f.write("cat={" + ",".join(json.dumps(c, ensure_ascii=False) for c in cats) + "},\n}\n")
    print("encoder.lua", os.path.getsize(os.path.join(OUTDIR, "encoder.lua")) / 1e6, "MB")

if __name__ == "__main__":
    main()
