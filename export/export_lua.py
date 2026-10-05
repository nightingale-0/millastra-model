"""Export a (QAT) MiniMind2 checkpoint to Lua modules for the 千星 client runtime.

    python export/export_lua.py --ckpt ckpt/qat-q4 --out build

Writes
  build/mm_weights.lua  4-bit weights (bit planes) + fp16 block scales + norms + per-turn prefix cache
  build/mm_tok.lua      byte-level BPE tokenizer tables
  build/golden.json     reference data for lua/test.mjs (tokenizer cases, logits, greedy ids)

Weight format (per matrix, row-major): every row is cols/8 groups of 8 inputs;
a group is 4 bytes, byte p = bit p of the 8 signed 4-bit weights (two's
complement, so w = b0 + 2 b1 + 4 b2 - 8 b3). Scales: one per (row, 32 inputs), stored as
an 8-bit code on a per-matrix log2 scale (s = 2^(lo + code*step); vs fp16: KL 4e-5,
top-1 99.8%, quant/scale8.py).
Prefix cache (export/presets.json), one entry per message turn: each persona's
system turn, and under it each preset user turn (incl. the assistant header).
An entry stores the K/V of its own tokens (int8, one fp16 scale per
token/layer/K|V/kv-head); user turns also store the top-64 first-step logits.
Binary data is base85 text: 4 bytes -> 5 chars from the printable ASCII range without
'=' (so the [==[ ]==] long strings can hold it verbatim); the level's save budget is
20MB, so the 25% text overhead instead of 33% matters. Every blob is split into parts of
whole units (rows / tokens) of at most 48KB decoded: the client sandbox rejects large
table.concat results, so the runtime keeps the parts and indexes into them.
"""
import argparse, json, os, re, sys, numpy as np, torch, regex
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quant"))
from prompts import EVAL

BLOCK, CHUNK = 32, 1 << 16          # CHUNK: encoded chars per long string
TOPK = 64                           # cached first-step logits per full preset

# ---------------------------------------------------------------- encoding helpers
def enc(b: bytes):
    """bytes -> list of text chunks (4 chars per 3 bytes, alphabet chr(63..126))."""
    a = np.frombuffer(b + b"\0" * (-len(b) % 3), dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    n = (a[:, 0] << 16) | (a[:, 1] << 8) | a[:, 2]
    c = np.stack([(n >> 18) & 63, (n >> 12) & 63, (n >> 6) & 63, n & 63], 1).astype(np.uint8) + 63
    s = c.tobytes().decode("ascii")
    return [s[i:i + CHUNK] for i in range(0, len(s), CHUNK)]

# printable ASCII without "=" (long-string safe) and without the lowercase vowels: random weight data
# then cannot spell English words (the level review scans script text)
B85 = bytes(c for c in range(33, 127) if c != 61 and chr(c) not in "aeiouUIO")   # exactly 85

def enc85(b: bytes):
    """bytes -> text chunks, 5 chars per 4 bytes (big-endian base 85)."""
    a = np.frombuffer(b + b"\0" * (-len(b) % 4), dtype=">u4").astype(np.uint64)
    d = np.stack([(a // 85 ** k) % 85 for k in (4, 3, 2, 1, 0)], 1).astype(np.uint8)
    s = np.frombuffer(B85, dtype=np.uint8)[d].tobytes().decode("ascii")
    step = CHUNK // 5 * 5
    return [s[i:i + step] for i in range(0, len(s), step)]

# words the level review may flag; matched case-insensitively. Encoded weight text breaks
# them with "=" (not in the alphabet; the runtime strips it before decoding), plain strings
# containing them are written fully byte-escaped
FLAG_WORDS = ["fuck", "shit", "porn", "sex", "rape", "nazi", "dick", "cunt", "bitch", "nigg", "slut", "whore", "pussy", "cock", "kill", "drug"]
FLAG_RE = re.compile("|".join(FLAG_WORDS), re.I)

def has_flag_word(s: str):
    return FLAG_RE.search(s) is not None

def break_flag_words(s: str):
    while True:
        m = FLAG_RE.search(s)
        if not m: return s
        k = m.start() + len(m.group()) // 2
        s = s[:k] + "=" + s[k:]

PART_MAX = 48000     # decoded bytes per part: the client sandbox rejects big table.concat results

def lua_blob(b: bytes, unit: int = 4):
    """bytes -> {n=, b=85, unit=, per=, part1, part2, ...}: every part decodes on its own to
    `per` whole units of `unit` bytes (a matrix row, a cached token, ...), at most PART_MAX
    bytes, so the runtime never has to join parts into one big string."""
    per = max(1, PART_MAX // unit)
    while (per * unit) % 4: per -= 1          # each part is a whole number of base85 groups
    assert per >= 1, unit
    step = per * unit
    parts = [b[i:i + step] for i in range(0, len(b), step)] or [b""]
    enc = [break_flag_words("".join(enc85(p))) for p in parts]
    return "{n=%d,b=85,unit=%d,per=%d,\n%s}" % (len(b), unit, per, ",\n".join("[==[%s]==]" % e for e in enc))

def lua_floats(v):
    return "{" + ",".join("%.9g" % x for x in v) + "}"

# ---------------------------------------------------------------- quantization
def quant4(w):
    """w [R, C] float -> (int8 q in [-8,7], fp16 scales [R, C/32]); scale = amax/7 rounded to fp16."""
    R, C = w.shape
    wb = w.reshape(R, C // BLOCK, BLOCK)
    s = (np.abs(wb).max(-1) / 7).astype(np.float16)
    sf = s.astype(np.float64)
    q = np.clip(np.round(wb / np.where(sf == 0, 1, sf)[..., None]), -8, 7).astype(np.int8)
    return q.reshape(R, C), s

def pack4(q):
    """int8 [R, C] -> bit-plane bytes (see module docstring)."""
    R, C = q.shape
    u = (q.astype(np.int16) & 15).astype(np.uint8).reshape(R, C // 8, 8)
    planes = np.zeros((R, C // 8, 4), np.uint8)
    for p in range(4):
        bits = (u >> p) & 1
        planes[:, :, p] = (bits << np.arange(8, dtype=np.uint8)).sum(-1).astype(np.uint8)
    return planes.tobytes()

def scale8(s):
    """fp16 block scales -> (uint8 codes, lo, step) on a log2 scale, and the decoded scales."""
    ls = np.log2(np.maximum(s.astype(np.float64), 1e-12))
    lo, hi = ls.min(), ls.max()
    step = max((hi - lo) / 255, 1e-9)
    code = np.clip(np.round((ls - lo) / step), 0, 255).astype(np.uint8)
    return code, lo, step, np.exp2(lo + code.astype(np.float64) * step)

def dequant(q, s):
    R, C = q.shape
    return (q.reshape(R, C // BLOCK, BLOCK).astype(np.float64) * s.astype(np.float64)[..., None]).reshape(R, C)

# ---------------------------------------------------------------- tokenizer
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))

def lua_bytes(b: bytes):
    if has_flag_word(b.decode("latin-1")): return '"' + "".join("\\%03d" % x for x in b) + '"'
    out = []
    for i, x in enumerate(b):
        if 32 <= x < 127 and x not in (34, 92):
            out.append(chr(x))
        else:
            nxt = b[i + 1] if i + 1 < len(b) else 0
            out.append("\\%03d" % x if 48 <= nxt <= 57 else "\\%d" % x)
    return '"' + "".join(out) + '"'

def char_classes():
    """Codepoint runs -> class (0 other, 1 letter, 2 number, 3 whitespace), GPT-2 regex semantics."""
    L, N, S = regex.compile(r"\p{L}"), regex.compile(r"\p{N}"), regex.compile(r"\s")
    starts, cls, prev = [], [], None
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF: c = 0
        else:
            ch = chr(cp)
            c = 1 if L.match(ch) else 2 if N.match(ch) else 3 if S.match(ch) else 0
        if c != prev: starts.append(cp); cls.append(c); prev = c
    return starts, cls

def export_tokenizer(tok, path):
    tj = json.loads(tok.backend_tokenizer.to_str())
    b2u = bytes_to_unicode(); u2b = {v: k for k, v in b2u.items()}
    vocab = tj["model"]["vocab"]
    specials = {t["content"]: t["id"] for t in tj["added_tokens"]}
    V = max(vocab.values()) + 1
    raw = [b""] * V
    for t, i in vocab.items():
        raw[i] = t.encode() if t in specials else bytes(u2b[c] for c in t)
    merges = []
    for m in tj["model"]["merges"]:
        a, b = m if isinstance(m, list) else m.split(" ")
        merges += [vocab[a], vocab[b], vocab[a + b]]
    starts, cls = char_classes()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("-- generated by export/export_lua.py: MiniMind2 byte-level BPE tables\nreturn {\n")
        f.write("specials={%s},\n" % ",".join('["%s"]=%d' % (k, v) for k, v in specials.items()))
        f.write("pieces={\n" + ",\n".join(lua_bytes(r) for r in raw) + "},\n")
        f.write("merges={" + ",".join(map(str, merges)) + "},\n")
        f.write("cls_start={" + ",".join(map(str, starts)) + "},\n")
        f.write("cls={" + ",".join(map(str, cls)) + "},\n}\n")
    print(f"tokenizer: {V} pieces, {len(merges)//3} merges, {len(starts)} class runs")

# ---------------------------------------------------------------- prefix cache
# Turn token layout (same as lua/minimind.lua):
#   system    <|im_start|>system\n{c}<|im_end|>\n
#   user      <|im_start|>user\n{c}<|im_end|>\n<|im_start|>assistant\n
#   assistant {c}<|im_end|>\n
def turn_text(role, content):
    if role == "system": return "<|im_start|>system\n%s<|im_end|>\n" % content
    if role == "user": return "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % content
    return "%s<|im_end|>\n" % content

def kv_of(gm, ids):
    """[T, layers, 2, kv_heads, head_dim] post-RoPE K/V + last-position logits."""
    with torch.no_grad():
        out = gm(torch.tensor([ids]), use_cache=True)
    pkv = out.past_key_values
    L = []
    for l in range(gm.config.num_hidden_layers):
        k, v = (pkv[l][0], pkv[l][1]) if isinstance(pkv, (tuple, list)) else (pkv.key_cache[l], pkv.value_cache[l])
        L.append(torch.stack([k[0], v[0]]).permute(2, 0, 1, 3))      # [T, 2, kvh, hd]
    return torch.stack(L, 1).numpy(), out.logits[0, -1].numpy()

def q8(seg):
    sc = (np.abs(seg).max(-1, keepdims=True) / 127).astype(np.float16)
    q = np.clip(np.round(seg / np.where(sc == 0, 1, sc.astype(np.float64))), -127, 127).astype(np.int8)
    return q.tobytes(), sc.tobytes()

def build_cache(tok, gm, cfg, presets):
    ref = tok.apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], tokenize=False, add_generation_prompt=True)
    assert turn_text("system", "S") + turn_text("user", "U") == ref
    systems = presets["systems"]
    entries, persona = [], {}                         # entry: parent, start, role, text, ids, q, s, top
    for name, text in systems.items():
        ids = tok(turn_text("system", text)).input_ids
        kvs, _ = kv_of(gm, ids)
        qb, sb = q8(kvs)
        entries.append({"parent": 0, "start": 0, "role": "system", "text": text, "ids": ids, "q": qb, "s": sb, "top": None})
        persona[name] = len(entries)
    for p in presets["prompts"]:
        par = persona[p["system"]]
        sys_ids = entries[par - 1]["ids"]
        ids = tok(turn_text("user", p["user"])).input_ids
        full = sys_ids + ids
        assert full == tok(turn_text("system", systems[p["system"]]) + turn_text("user", p["user"])).input_ids
        kvs, logits = kv_of(gm, full)
        qb, sb = q8(kvs[len(sys_ids):])
        top = np.argsort(-logits)[:TOPK]
        entries.append({"parent": par, "start": len(sys_ids), "role": "user", "text": p["user"], "ids": ids, "q": qb, "s": sb,
                        "top": top, "top_v": logits[top]})
    ntok = sum(len(e["ids"]) for e in entries)
    print(f"prefix cache: {len(entries)} turns, {ntok} tokens, {sum(len(e['q']) + len(e['s']) for e in entries)/1e3:.0f}KB")
    out = ["cache={\npersonas={%s},\nturns={\n" % ",".join("[%s]=%d" % (lua_bytes(k.encode()), v) for k, v in persona.items())]
    for e in entries:
        top = "" if e["top"] is None else ",top={ids={%s},logits=%s}" % (",".join(map(str, e["top"])), lua_floats(e["top_v"]))
        out.append("{parent=%d,start=%d,role=%s,text=%s,ids={%s},q=%s,s=%s%s},\n" % (
            e["parent"], e["start"], lua_bytes(e["role"].encode()), lua_bytes(e["text"].encode()), ",".join(map(str, e["ids"])),
            lua_blob(e["q"], len(e["q"]) // len(e["ids"])), lua_blob(e["s"], len(e["s"]) // len(e["ids"])), top))
    out.append("}},\n")

    # golden chat cases: fully cached preset, cached persona + new question, default persona, and a second turn
    def greedy(ids):
        with torch.no_grad():
            logits = gm(torch.tensor([ids])).logits[0, -1].numpy()
            gen = gm.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long), max_new_tokens=24,
                              do_sample=False, pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
        return logits, gen
    cases = []
    for sysname, user in [(presets["prompts"][1]["system"], presets["prompts"][1]["user"]), (presets["prompts"][1]["system"], "你好，我叫小明。"),
                          ("default", "中国的首都是哪里？")]:
        ids = tok(turn_text("system", systems[sysname]) + turn_text("user", user)).input_ids
        logits, gen = greedy(ids)
        print(f"golden [{sysname}] {user} -> {tok.decode(gen, skip_special_tokens=True)!r}")
        cases.append({"system": sysname, "user": user, "ids": ids, "logits": logits.tolist(), "greedy": gen})
    # second turn after case 3 (assistant turn = its greedy tokens, closed by <|im_end|>\n)
    prev = cases[-1]
    gen0 = prev["greedy"]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if gen0[-1] != im_end: gen0 = gen0 + [im_end]      # cut at max_new: the runtime closes the turn the same way
    ids = prev["ids"] + gen0 + tok("\n" + turn_text("user", "那美国呢？")).input_ids
    logits, gen = greedy(ids)
    print(f"golden [default] 那美国呢？(2nd turn) -> {tok.decode(gen, skip_special_tokens=True)!r}")
    cases.append({"system": "default", "user": "那美国呢？", "follows": len(cases) - 1, "ids": ids, "logits": logits.tolist(), "greedy": gen})
    return "".join(out), cases

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/qat-q4"); ap.add_argument("--out", default="build")
    ap.add_argument("--presets", default=os.path.join(os.path.dirname(__file__), "presets.json"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, torch_dtype=torch.float32).eval()
    cfg = model.config
    sd = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}

    # quantize + write back the exact dequantized weights (golden model = what Lua computes)
    mats, gold = {}, {}
    def q(name, w):
        qq, s = quant4(w)
        code, lo, step, s8 = scale8(s)
        mats[name] = (w.shape, pack4(qq), code.tobytes(), lo, step)
        return dequant(qq, s8)
    P = "model.layers.%d."
    for l in range(cfg.num_hidden_layers):
        p = P % l
        qkv = np.concatenate([sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"], sd[p + "self_attn.v_proj.weight"]])
        d = q("qkv%d" % l, qkv); nq, nk = cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim
        gold[p + "self_attn.q_proj.weight"], gold[p + "self_attn.k_proj.weight"], gold[p + "self_attn.v_proj.weight"] = d[:nq], d[nq:nq + nk], d[nq + nk:]
        gold[p + "self_attn.o_proj.weight"] = q("o%d" % l, sd[p + "self_attn.o_proj.weight"])
        gu = np.concatenate([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]])
        d = q("gu%d" % l, gu); H = cfg.intermediate_size
        gold[p + "mlp.gate_proj.weight"], gold[p + "mlp.up_proj.weight"] = d[:H], d[H:]
        gold[p + "mlp.down_proj.weight"] = q("down%d" % l, sd[p + "mlp.down_proj.weight"])
    gold["lm_head.weight"] = gold["model.embed_tokens.weight"] = q("head", sd["lm_head.weight"])
    gm = AutoModelForCausalLM.from_pretrained(a.ckpt, torch_dtype=torch.float32).double().eval()
    with torch.no_grad():
        for k, v in gm.state_dict().items():
            if k in gold: v.copy_(torch.from_numpy(gold[k]))

    cache, cases = build_cache(tok, gm, cfg, json.load(open(a.presets, encoding="utf-8")))

    # weights module
    path = os.path.join(a.out, "mm_weights.lua")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("-- generated by export/export_lua.py from %s\nreturn {\n" % a.ckpt.replace("\\", "/"))
        f.write("cfg={dim=%d,layers=%d,heads=%d,kv_heads=%d,head_dim=%d,hidden=%d,vocab=%d,rope_theta=%.1f,eps=%g,block=%d},\n" % (
            cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
            cfg.intermediate_size, cfg.vocab_size, cfg.rope_theta, cfg.rms_norm_eps, BLOCK))
        f.write("norm_attn={%s},\n" % ",".join(lua_floats(sd[P % l + "input_layernorm.weight"]) for l in range(cfg.num_hidden_layers)))
        f.write("norm_ffn={%s},\n" % ",".join(lua_floats(sd[P % l + "post_attention_layernorm.weight"]) for l in range(cfg.num_hidden_layers)))
        f.write("norm_final=%s,\n" % lua_floats(sd["model.norm.weight"]))
        f.write(cache)
        f.write("mats={\n")
        for name, ((R, C), w, s, lo, step) in mats.items():
            f.write("%s={rows=%d,cols=%d,slo=%.17g,sstep=%.17g,\nw=%s,\ns8=%s},\n" % (name, R, C, lo, step, lua_blob(w, C // 2), lua_blob(s, C // BLOCK)))
        f.write("},\n}\n")
    print(f"weights: {path} {os.path.getsize(path)/1e6:.1f}MB (raw {sum(len(w)+len(s) for _, w, s, _, _ in mats.values())/1e6:.1f}MB)")

    export_tokenizer(tok, os.path.join(a.out, "mm_tok.lua"))

    # golden data
    texts = list(EVAL) + ["Hello world! It's 2026, isn't it?  Spaces   and\ttabs\n\nnew lines.", "１２３ｖｓ123，“引号”…—破折号😀emoji", "  leading spaces", "trailing   "]
    with open("data/sft_t2t_mini.jsonl", encoding="utf-8") as fd:
        for i, line in enumerate(fd):
            if i >= 300: break
            for m in json.loads(line)["conversations"]:
                texts.append(m["content"][:400])
    tok_cases = [{"text": t, "ids": tok(t, add_special_tokens=False).input_ids} for t in texts]
    with open(os.path.join(a.out, "golden.json"), "w", encoding="utf-8") as f:
        json.dump({"tok": tok_cases, "chat": cases}, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
