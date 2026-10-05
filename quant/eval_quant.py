"""Weight-only quantization study for MiniMind2-Small (llama.cpp-style block formats).

For each scheme every Linear weight (and the tied embedding / LM head) is
quantized -> dequantized in place, then compared with the fp32 model on the
EVAL transcripts: mean KL divergence of the next-token distribution and top-1
agreement (llama.cpp `--kl-divergence` style). Also reports packed size.

    python quant/eval_quant.py
"""
import copy, math, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import CALIB, EVAL

MODEL = "models/MiniMind2-Small"
BLOCK = 32
torch.manual_seed(0)

# ---------------------------------------------------------------- quantizers
def q_sym(w, bits, block=BLOCK, imp=None):
    """Symmetric block quant (Q8_0 / Q4_0 style). imp: per-input-column importance
    (imatrix); if given, the block scale is searched to minimise weighted error."""
    out_f, in_f = w.shape
    wb = w.reshape(out_f, in_f // block, block)
    qmax = 2 ** (bits - 1) - 1
    amax = wb.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    if imp is None:
        scale = amax / qmax
        q = (wb / scale).round().clamp(-qmax - 1, qmax)
        return (q * scale).reshape(out_f, in_f)
    ib = imp.reshape(1, in_f // block, block)
    best, best_err = None, None
    for f in torch.linspace(0.7, 1.05, 15):  # scale search, llama.cpp make_qx_quants-like
        scale = amax * f / qmax
        q = (wb / scale).round().clamp(-qmax - 1, qmax)
        dq = q * scale
        err = ((dq - wb) ** 2 * ib).sum(-1, keepdim=True)
        if best is None:
            best, best_err = dq, err
        else:
            m = err < best_err
            best = torch.where(m, dq, best); best_err = torch.where(m, err, best_err)
    return best.reshape(out_f, in_f)

def q_asym(w, bits, block=BLOCK):
    """Asymmetric block quant with scale + min (Q4_1 style)."""
    out_f, in_f = w.shape
    wb = w.reshape(out_f, in_f // block, block)
    lo, hi = wb.amin(-1, keepdim=True), wb.amax(-1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
    q = ((wb - lo) / scale).round().clamp(0, 2 ** bits - 1)
    return (q * scale + lo).reshape(out_f, in_f)

def q_ternary(w, block=BLOCK):
    """Ternary {-1,0,1}*scale per block (post-training; expected to be poor)."""
    out_f, in_f = w.shape
    wb = w.reshape(out_f, in_f // block, block)
    scale = wb.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    q = (wb / scale).round().clamp(-1, 1)
    return (q * scale).reshape(out_f, in_f)

# bits per weight incl. one fp16 scale (+ fp16 min for asym) per block
def bpw(kind, bits):
    extra = 16 * (2 if kind == "asym" else 1) / BLOCK
    return (1.58 if kind == "tern" else bits) + extra

SCHEMES = [
    ("q8_0", "sym", 8, False), ("q6_0", "sym", 6, False),
    ("q4_0", "sym", 4, False), ("q4_1", "asym", 4, False), ("q4_0+imatrix", "sym", 4, True),
    ("q3_0", "sym", 3, False), ("q3_0+imatrix", "sym", 3, True),
    ("q2_0+imatrix", "sym", 2, True), ("ternary", "tern", 2, False),
]

# ---------------------------------------------------------------- data
def chat_ids(tok, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids

def transcripts(model, tok, prompts, n_new=64):
    seqs = []
    with torch.no_grad():
        for p in prompts:
            ids = chat_ids(tok, p)
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                                 do_sample=False, repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
            seqs.append((out, ids.shape[1]))
    return seqs

def imatrix(model, tok, prompts):
    """Mean squared input activation per column for every Linear (llama.cpp imatrix)."""
    acc, cnt, hooks = {}, {}, []
    def mk(name):
        def h(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            acc[name] = acc.get(name, 0) + (x ** 2).sum(0); cnt[name] = cnt.get(name, 0) + x.shape[0]
        return h
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(mk(name)))
    with torch.no_grad():
        for out, _ in transcripts(model, tok, prompts):
            model(out)
    for h in hooks: h.remove()
    return {k: acc[k] / cnt[k] for k in acc}

def logprobs(model, seqs):
    res = []
    with torch.no_grad():
        for out, start in seqs:
            lg = model(out).logits[0, start - 1:-1].float()
            res.append(torch.log_softmax(lg, -1))
    return res

# ---------------------------------------------------------------- main
def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ref = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    print("calibrating imatrix ...", flush=True)
    imp = imatrix(ref, tok, CALIB)
    print("building eval transcripts ...", flush=True)
    seqs = transcripts(ref, tok, EVAL)
    ref_lp = logprobs(ref, seqs)
    ntok = sum(x.shape[0] for x in ref_lp)
    n_params = sum(p.numel() for p in ref.parameters())
    print(f"eval tokens {ntok}, params {n_params/1e6:.2f}M\n")
    print(f"{'scheme':16}{'bpw':>6}{'size MB':>9}{'KLD':>9}{'top1 %':>8}")
    for name, kind, bits, use_imp in SCHEMES:
        m = copy.deepcopy(ref)
        with torch.no_grad():
            for mname, mod in m.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    w = mod.weight.data
                    if kind == "sym":
                        # the tied LM head shares the embedding; its imatrix is the final hidden state
                        im = imp.get(mname) if use_imp else None
                        mod.weight.data = q_sym(w, bits, imp=im)
                    elif kind == "asym":
                        mod.weight.data = q_asym(w, bits)
                    else:
                        mod.weight.data = q_ternary(w)
        lp = logprobs(m, seqs)
        kld = sum((r.exp() * (r - q)).sum(-1).sum() for r, q in zip(ref_lp, lp)).item() / ntok
        top1 = sum((r.argmax(-1) == q.argmax(-1)).sum() for r, q in zip(ref_lp, lp)).item() / ntok
        b = bpw(kind, bits)
        print(f"{name:16}{b:6.2f}{n_params*b/8/1e6:9.1f}{kld:9.4f}{top1*100:8.1f}", flush=True)
        del m

if __name__ == "__main__":
    main()
