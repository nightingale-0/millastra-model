"""Route B: quantization-aware fine-tuning with self-distillation.

Student = MiniMind2-Small with every Linear (incl. the tied LM head) fake-quantized
to the deploy format (symmetric, block 32, one scale per block, STE through the
rounding). Teacher = the frozen fp32 model. Loss = KL(teacher || student) on all
tokens + a small CE term on the data. Afterwards the student is hard-quantized
(same RTN as eval_quant.q_sym) and scored against the ORIGINAL fp32 model.

    .venv-cuda/Scripts/python quant/qat.py --bits 4 --head-bits 4 --steps 3000
"""
import argparse, copy, json, math, os, random, time, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import EVAL
from eval_quant import MODEL, q_sym, transcripts, logprobs

BLOCK = 32

def fake_quant(w, bits):
    """Symmetric block-32 quant with a straight-through estimator."""
    out_f, in_f = w.shape
    wb = w.reshape(out_f, in_f // BLOCK, BLOCK)
    qmax = 2 ** (bits - 1) - 1
    scale = (wb.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax).detach()
    q = (wb / scale).round().clamp(-qmax - 1, qmax) * scale
    return (wb + (q - wb).detach()).reshape(out_f, in_f)

class QLinear(torch.nn.Module):
    def __init__(self, lin, bits):
        super().__init__(); self.lin, self.bits = lin, bits
    def forward(self, x):
        return F.linear(x, fake_quant(self.lin.weight, self.bits), self.lin.bias)

def wrap(model, bits, head_bits):
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, mod in list(parent.named_children()):
                if isinstance(mod, torch.nn.Linear):
                    setattr(parent, name, QLinear(mod, bits))
    model.lm_head = QLinear(model.lm_head, head_bits)   # still tied to embed_tokens
    return model

def unwrap_hard(model):
    """Replace QLinear by plain Linear holding the hard-quantized weight."""
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, mod in list(parent.named_children()):
                if isinstance(mod, QLinear):
                    mod.lin.weight.data = q_sym(mod.lin.weight.data, mod.bits); setattr(parent, name, mod.lin)
    ql = model.lm_head
    ql.lin.weight.data = q_sym(ql.lin.weight.data, ql.bits); model.lm_head = ql.lin
    return model

def load_texts(path, tok, limit):
    """sft jsonl -> chat-formatted strings. Accepts {"conversations":[{role,content}]} or {"text":...}."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(out) >= limit: break
            try: d = json.loads(line)
            except Exception: continue
            if "conversations" in d:
                msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in d["conversations"]]
                out.append(tok.apply_chat_template(msgs, tokenize=False))
            elif "text" in d:
                out.append(d["text"])
    return out

def batches(texts, tok, seq, bs, device):
    random.shuffle(texts)
    buf = []
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids[:seq]
        if len(ids) < 16: continue
        buf.append(ids)
        if len(buf) == bs:
            L = max(len(x) for x in buf)
            inp = torch.full((bs, L), tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((bs, L), dtype=torch.long)
            for i, x in enumerate(buf):
                inp[i, :len(x)] = torch.tensor(x); mask[i, :len(x)] = 1
            yield inp.to(device), mask.to(device)
            buf = []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4); ap.add_argument("--head-bits", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ce", type=float, default=0.1); ap.add_argument("--data", default="data/sft_t2t_mini.jsonl")
    ap.add_argument("--limit", type=int, default=400000); ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=MODEL, help="fp32 teacher = student init (e.g. an SFT checkpoint)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); random.seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    teacher = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    student = wrap(copy.deepcopy(teacher), a.bits, a.head_bits).train()
    texts = load_texts(a.data, tok, a.limit)
    print(f"device {dev}, texts {len(texts)}, bits {a.bits}/{a.head_bits}", flush=True)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    step, t0, toks = 0, time.time(), 0
    while step < a.steps:
        for inp, mask in batches(texts, tok, a.seq, a.bs, dev):
            with torch.no_grad():
                tl = teacher(inp, attention_mask=mask).logits.float()
            sl = student(inp, attention_mask=mask).logits.float()
            m = mask[:, 1:].float()
            kl = (F.kl_div(F.log_softmax(sl[:, :-1], -1), F.log_softmax(tl[:, :-1], -1), log_target=True, reduction="none").sum(-1) * m).sum() / m.sum()
            ce = (F.cross_entropy(sl[:, :-1].transpose(1, 2), inp[:, 1:], reduction="none") * m).sum() / m.sum()
            loss = kl + a.ce * ce
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step(); sched.step()
            step += 1; toks += int(mask.sum())
            if step % 100 == 0:
                print(f"step {step} kl {kl.item():.4f} ce {ce.item():.3f} {toks/(time.time()-t0):.0f} tok/s", flush=True)
            if step >= a.steps: break
    # hard-quantize and score against the original fp32 model (CPU, same eval as eval_quant)
    student = unwrap_hard(student).eval().cpu()
    ref = teacher.cpu()
    seqs = transcripts(ref, tok, EVAL)
    ref_lp = logprobs(ref, seqs); lp = logprobs(student, seqs)
    ntok = sum(x.shape[0] for x in ref_lp)
    kld = sum((r.exp() * (r - q)).sum(-1).sum() for r, q in zip(ref_lp, lp)).item() / ntok
    top1 = sum((r.argmax(-1) == q.argmax(-1)).sum() for r, q in zip(ref_lp, lp)).item() / ntok
    print(f"RESULT bits {a.bits}/{a.head_bits} steps {a.steps}: KLD {kld:.4f} top1 {top1*100:.1f}%", flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True); student.save_pretrained(a.out); tok.save_pretrained(a.out)

if __name__ == "__main__":
    main()
