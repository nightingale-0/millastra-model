"""Would speculative decoding pay off in the Lua runtime? Offline estimate on real answers.

    python bench/spec_sim.py

Drafts come from (a) prompt lookup: the last n tokens matched earlier in system prompt + question +
answer so far, (b) optionally an n-gram table of Paimon's training answers. A verify pass of B tokens
costs 0.4 + 0.6 B single-token passes (lua/minimind.lua matvec_batch); greedy output is unchanged.
"""
import collections, json, sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("ckpt/final-26m-q4")
sysmsg = json.load(open("data/distill/presets.json", encoding="utf-8"))["systems"]["paimon"]
ans = json.load(open("reports/answers-r8q4-nr6.json", encoding="utf-8"))
qs, A = ans["questions"], ans["answers"]["final-26m-q4"]

def table_from_train(n, limit=60000):
    t = collections.defaultdict(collections.Counter)
    with open("data/distill/train_r7.jsonl", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= limit: break
            for m in json.loads(l)["conversations"]:
                if m["role"] != "assistant": continue
                ids = tok(m["content"], add_special_tokens=False)["input_ids"]
                for j in range(len(ids) - n):
                    t[tuple(ids[j:j + n])][ids[j + n]] += 1
    return {k: c.most_common(1)[0][0] for k, c in t.items()}

def run(k, n, table=None, tn=3):
    passes = toks = 0
    for q, a in zip(qs, A):
        ctx = tok(sysmsg + q["q"], add_special_tokens=False)["input_ids"]
        out = tok(a, add_special_tokens=False)["input_ids"]
        i = 0
        while i < len(out):
            seq = ctx + out[:i]
            draft = []
            key = seq[-n:]
            for s in range(len(seq) - n - 1, -1, -1):            # prompt lookup, latest match first
                if seq[s:s + n] == key: draft = seq[s + n:s + n + k]; break
            if not draft and table is not None:
                cur = list(seq)
                while len(draft) < k:
                    nx = table.get(tuple(cur[-tn:]))
                    if nx is None: break
                    draft.append(nx); cur.append(nx)
            acc = 0
            while acc < len(draft) and i + acc < len(out) and draft[acc] == out[i + acc]: acc += 1
            B = 1 + len(draft)
            passes += 0.4 + 0.6 * B if B > 1 else 1.0
            i += acc + 1; toks += acc + 1
    return toks / passes                                          # tokens per single-pass cost unit

print(f"{len(A)} answers; baseline 1.00 token per pass-cost")
for n in (2, 3):
    for k in (2, 3, 4):
        print(f"prompt lookup n={n} k={k}: speedup x{run(k, n):.2f}")
for tn in (2, 3):
    tb = table_from_train(tn)
    print(f"n-gram table (n={tn}, {len(tb)} entries) + lookup, k=3: speedup x{run(3, 3, tb, tn):.2f}")
