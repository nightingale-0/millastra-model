"""Batched greedy answers of a student checkpoint to the probe questions (failure-driven distillation).

    python distill/student_answers.py ckpt/pd-26m-q4 [round_dir]   # round_dir/probe_q.jsonl -> round_dir/probe_answers.jsonl
"""
import json, os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_prep"))
from paimon import SYSTEM

D = "data/distill"

def main():
    ck = sys.argv[1]
    R = sys.argv[2] if len(sys.argv) > 2 else D
    held = {q["q"] for q in json.load(open(os.path.join(D, "eval_questions.json"), encoding="utf-8"))}
    qs, seen = [], set()
    with open(os.path.join(R, "probe_q.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for q in r["items"]:
                if q not in seen and q not in held and len(q) <= 60:
                    seen.add(q); qs.append({"q": q, "kind": r["kind"]})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ck); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.float32).to(dev).eval()
    out = open(os.path.join(R, "probe_answers.jsonl"), "w", encoding="utf-8")
    bs = 64
    for i in range(0, len(qs), bs):
        chunk = qs[i:i + bs]
        texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["q"]}],
                                         tokenize=False, add_generation_prompt=True) for q in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            o = m.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=80, do_sample=False, repetition_penalty=1.1, pad_token_id=tok.pad_token_id or 0)
        for q, row in zip(chunk, o[:, enc.input_ids.shape[1]:]):
            q["a"] = tok.decode(row, skip_special_tokens=True).strip()
            out.write(json.dumps(q, ensure_ascii=False) + "\n")
        if (i // bs) % 10 == 0: print(f"{i + len(chunk)}/{len(qs)}", flush=True)
    out.close()
    print(f"{len(qs)} probe answers from {ck}")

if __name__ == "__main__":
    main()
