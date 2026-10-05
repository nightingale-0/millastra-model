"""Answers of several checkpoints to the eval questions (held-out synthesized + fixed player prompts).

    python distill/answer.py --out reports/answers.json ckpt/a ckpt/b ...
"""
import argparse, json, os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_prep"))
from paimon import SYSTEM
# fixed player prompts, first in every report
PROMPTS = ["你是谁？", "派蒙，我们今天去哪儿？", "你饿了吗？", "给我讲个笑话吧。", "蒙德有什么好吃的？",
           "你觉得我这个人怎么样？", "应急食品是什么意思？", "这里是千星奇域，你来过吗？", "璃月的神是谁？",
           "你会飞吗？", "我想休息一下。", "中国的首都是哪里？"]
from transformers import LogitsProcessor, LogitsProcessorList

class GenOnlyRepetitionPenalty(LogitsProcessor):
    """repetition penalty on generated tokens only (HF's also penalizes the prompt, i.e. the question's own words)."""
    def __init__(self, penalty, prompt_len): self.p, self.n = penalty, prompt_len
    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.n:]
        if gen.shape[1] == 0: return scores
        s = torch.gather(scores, 1, gen)
        s = torch.where(s < 0, s * self.p, s / self.p)
        return scores.scatter(1, gen, s)

class GenOnlyNoRepeat(LogitsProcessor):
    """ban a token that would repeat an n-gram of this answer (lua/minimind.lua no_repeat_ngram); the prompt is not counted."""
    def __init__(self, n, prompt_len): self.k, self.n = n, prompt_len
    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            out = input_ids[b, self.n:].tolist(); L, k = len(out), self.k
            if L < k: continue
            pre = out[L - k + 1:]
            for s in range(L - k + 1):
                if out[s:s + k - 1] == pre: scores[b, out[s + k - 1]] = -float("inf")
        return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--questions", default="data/distill/eval_questions.json")
    ap.add_argument("--rp", type=float, default=1.1, help="repetition penalty")
    ap.add_argument("--rp-gen-only", action="store_true", help="apply it to generated tokens only")
    ap.add_argument("--no-repeat", type=int, default=6, help="no_repeat_ngram in the answer (0 = off), as in the game: 6 tokens, since 派蒙 alone is several byte-level tokens")
    ap.add_argument("--suffix", default="", help="appended to model names in the output")
    a = ap.parse_args()
    qs = [{"q": p, "topic": "固定问题"} for p in PROMPTS]
    if os.path.exists(a.questions): qs += json.load(open(a.questions, encoding="utf-8"))
    multi = os.path.join(os.path.dirname(a.questions), "eval_multi.json")
    if os.path.exists(multi): qs += json.load(open(multi, encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"questions": qs, "answers": {}}
    for ck in a.ckpts:
        tok = AutoTokenizer.from_pretrained(ck)
        m = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.float32).to(dev).eval()
        outs = []
        for q in qs:
            msgs = [{"role": "system", "content": SYSTEM}]
            for hq, ha in q.get("history", []):
                msgs += [{"role": "user", "content": hq}, {"role": "assistant", "content": ha}]
            msgs.append({"role": "user", "content": q["q"]})
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                o = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=80, do_sample=False,
                               repetition_penalty=1.0 if a.rp_gen_only else a.rp, pad_token_id=tok.eos_token_id,
                               logits_processor=LogitsProcessorList(([GenOnlyRepetitionPenalty(a.rp, ids.shape[1])] if a.rp_gen_only else []) +
                                                                    ([GenOnlyNoRepeat(a.no_repeat, ids.shape[1])] if a.no_repeat else [])))
            outs.append(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip())
        res["answers"][os.path.basename(ck.rstrip("/")) + a.suffix] = outs
        print(f"{ck}: {len(outs)} answers", flush=True)
        del m; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
