"""Effect of storing the block scales as 8-bit log values instead of fp16.

    python quant/scale8.py ckpt/final-26m-q4

Per matrix: lo = min log2(scale), step = (max - lo) / 255, code = round((log2 s - lo) / step),
s' = 2^(lo + code*step). The 4-bit codes are kept as they are. Reports KL / top-1 of the
8-bit-scale model against the fp16-scale model on the usual eval transcripts.
"""
import sys, os, copy, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_quant import transcripts, logprobs
from prompts import EVAL

BLOCK = 32

def requant(w):
    R, C = w.shape
    wb = w.reshape(R, C // BLOCK, BLOCK).double()
    s = (wb.abs().amax(-1) / 7).clamp(min=1e-12).half().double()          # fp16 scales (as exported)
    q = (wb / s[..., None]).round().clamp(-8, 7)
    ls = torch.log2(s); lo, hi = ls.min(), ls.max()
    step = (hi - lo) / 255
    s8 = torch.exp2(lo + ((ls - lo) / step).round() * step)
    return (q * s[..., None]).reshape(R, C), (q * s8[..., None]).reshape(R, C), (s8 / s - 1).abs().max().item()

def main():
    ck = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(ck)
    m16 = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.float32).double().eval()
    m8 = copy.deepcopy(m16)
    worst = 0
    with torch.no_grad():
        for (n, p16), (_, p8) in zip(m16.named_parameters(), m8.named_parameters()):
            if p16.dim() == 2 and ("proj" in n or "embed" in n or "lm_head" in n):
                a, b, e = requant(p16.data); p16.copy_(a); p8.copy_(b); worst = max(worst, e)
    m16, m8 = m16.float(), m8.float()
    seqs = transcripts(m16, tok, EVAL)
    l16, l8 = logprobs(m16, seqs), logprobs(m8, seqs)
    ntok = sum(x.shape[0] for x in l16)
    kld = sum((r.exp() * (r - q)).sum(-1).sum() for r, q in zip(l16, l8)).item() / ntok
    top1 = sum((r.argmax(-1) == q.argmax(-1)).sum() for r, q in zip(l16, l8)).item() / ntok
    print(f"8-bit log scales vs fp16 scales: max relative scale error {worst*100:.2f}%, KLD {kld:.5f}, top1 agreement {top1*100:.2f}%")

if __name__ == "__main__":
    main()
