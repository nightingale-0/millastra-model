"""Weight averaging of fine-tunes that share an init ("model soup"): out = sum w_i * model_i

    python train/soup.py --out ckpt/soup-26m ckpt/a 0.5 ckpt/b 0.5

All inputs must come from the same starting checkpoint (here ckpt/dapt-26m), so their weights live
in one basin and averaging keeps what each learned (round 10: rcp-s0 is the better chatter, the
round-10 seeds have the extra skills).
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("pairs", nargs="+", help="checkpoint weight checkpoint weight ...")
    a = ap.parse_args()
    cks, ws = a.pairs[0::2], [float(x) for x in a.pairs[1::2]]
    assert len(cks) == len(ws) and abs(sum(ws) - 1) < 1e-6, "weights must sum to 1"
    base = AutoModelForCausalLM.from_pretrained(cks[0], torch_dtype=torch.float32)
    acc = {k: v.clone() * ws[0] for k, v in base.state_dict().items()}
    for ck, w in zip(cks[1:], ws[1:]):
        sd = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.float32).state_dict()
        for k in acc: acc[k] += sd[k] * w
    base.load_state_dict(acc)
    base.save_pretrained(a.out); AutoTokenizer.from_pretrained(cks[0]).save_pretrained(a.out)
    print("soup", " + ".join(f"{w:g}*{c}" for c, w in zip(cks, ws)), "->", a.out)

if __name__ == "__main__":
    main()
