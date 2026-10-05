"""Supervised fine-tuning (optionally with a same-vocab teacher: logit distillation).

Data: jsonl lines {"conversations": [{"role": "system"|"user"|"assistant", "content": ...}, ...]}.
Loss only on assistant tokens (content + <|im_end|>):
    loss = (1 - kd) * CE(labels) + kd * KL(teacher || student)          (kd = 0 without --teacher)

    .venv-cuda/Scripts/python train/sft.py --model models/MiniMind2 --data data/paimon/train.jsonl \
        --epochs 3 --lr 1e-4 --out ckpt/sft-104m
    .venv-cuda/Scripts/python train/sft.py --model models/MiniMind2-Small --teacher ckpt/sft-104m --kd 0.7 ...
"""
import argparse, json, math, os, random, time, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def encode(tok, msgs, seq):
    """chat template -> ids + mask of assistant tokens (via character offsets)."""
    text, spans = "", []
    for m in msgs:
        if m["role"] == "assistant":
            text += "<|im_start|>assistant\n"
            a = len(text); text += m["content"] + "<|im_end|>"; spans.append((a, len(text)))
            text += "\n"
        else:
            text += "<|im_start|>%s\n%s<|im_end|>\n" % (m["role"], m["content"])
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, mask = enc["input_ids"], []
    for s, e in enc["offset_mapping"]:
        mask.append(int(any(a <= s < b for a, b in spans)))
    return ids[:seq], mask[:seq]

def load(path, tok, seq, limit=None):
    """chat lines {"conversations": ...} (loss on assistant tokens) and plain text lines {"text": ...}
    (continued pretraining: every token, long documents cut into windows of seq tokens)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit and len(out) >= limit: break
            d = json.loads(line)
            if "text" in d:
                ids = tok(d["text"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
                for i in range(0, len(ids), seq):
                    w = ids[i:i + seq]
                    if len(w) >= 16: out.append((w, [1] * len(w)))
                continue
            ids, mask = encode(tok, d["conversations"], seq)
            if sum(mask) > 0: out.append((ids, mask))
    return out

def batches(data, bs, device, pad):
    order = list(range(len(data)))
    random.shuffle(order)
    # bucket by length inside chunks of 50 batches to cut padding
    for c in range(0, len(order), bs * 50):
        chunk = sorted(order[c:c + bs * 50], key=lambda i: len(data[i][0]))
        bl = [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
        random.shuffle(bl)
        for b in bl:
            L = max(len(data[i][0]) for i in b)
            inp = torch.full((len(b), L), pad, dtype=torch.long)
            att = torch.zeros((len(b), L), dtype=torch.long)
            lm = torch.zeros((len(b), L))
            for j, i in enumerate(b):
                ids, m = data[i]
                inp[j, :len(ids)] = torch.tensor(ids); att[j, :len(ids)] = 1; lm[j, :len(ids)] = torch.tensor(m, dtype=torch.float)
            yield inp.to(device), att.to(device), lm.to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--teacher", default=None)
    ap.add_argument("--kd", type=float, default=0.7); ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None); ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--seq", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000, help="with --val: evaluate and keep the best checkpoint")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).train()
    teacher = None
    if a.teacher:
        teacher = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.float32).to(dev).eval()
    t0 = time.time()
    data = load(a.data, tok, a.seq, a.limit)
    val = load(a.val, tok, a.seq, 2000) if a.val else None
    ntok = sum(len(x[0]) for x in data)
    steps = int(math.ceil(len(data) / a.bs) * a.epochs)
    print(f"device {dev}; {len(data)} samples, {ntok/1e6:.1f}M tokens ({time.time()-t0:.0f}s to tokenize); {steps} steps"
          f"{'; teacher ' + a.teacher + ' kd ' + str(a.kd) if teacher else ''}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    warm = max(50, steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    pad = tok.pad_token_id or 0
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)

    def evaluate():
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad(), amp:
            for inp, att, lm in batches(val, a.bs, dev, pad):
                lg = model(inp, attention_mask=att).logits.float()[:, :-1]
                m = lm[:, 1:]
                tot += (F.cross_entropy(lg.transpose(1, 2), inp[:, 1:], reduction="none") * m).sum().item(); n += m.sum().item()
        model.train(); return tot / max(n, 1)

    def save(tag):
        os.makedirs(a.out, exist_ok=True)
        model.save_pretrained(a.out); tok.save_pretrained(a.out)
        print(f"saved {a.out} ({tag})", flush=True)

    best = float("inf")
    def check():
        nonlocal best
        v = evaluate()
        print(f"val ce {v:.3f} at step {step}{' (best)' if v < best else ''}", flush=True)
        if v < best: best = v; save(f"step {step}, val ce {v:.3f}")

    step, seen, t0 = 0, 0, time.time()
    if val: print(f"val ce before: {evaluate():.3f}", flush=True)
    while step < steps:
        for inp, att, lm in batches(data, a.bs, dev, pad):
            with amp:
                lg = model(inp, attention_mask=att).logits.float()[:, :-1]
            m = lm[:, 1:]
            ce = (F.cross_entropy(lg.transpose(1, 2), inp[:, 1:], reduction="none") * m).sum() / m.sum()
            loss = ce
            if teacher is not None:
                with torch.no_grad(), amp:
                    tl = teacher(inp, attention_mask=att).logits.float()[:, :-1]
                kl = (F.kl_div(F.log_softmax(lg, -1), F.log_softmax(tl, -1), log_target=True, reduction="none").sum(-1) * m).sum() / m.sum()
                loss = (1 - a.kd) * ce + a.kd * kl
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            step += 1; seen += int(att.sum())
            if step % 100 == 0:
                extra = f" kl {kl.item():.3f}" if teacher is not None else ""
                print(f"step {step}/{steps} ce {ce.item():.3f}{extra} lr {sched.get_last_lr()[0]:.2e} {seen/(time.time()-t0):.0f} tok/s", flush=True)
            if val and step % a.eval_every == 0: check()
            if step >= steps: break
    if val:
        check()
        print(f"val ce final: {best:.3f} (best kept) {(time.time()-t0)/60:.1f} min", flush=True)
    else:
        save(f"{(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
