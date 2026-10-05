"""Greedy answers of a checkpoint with the game's decoding (repetition penalty 1.1, no repeated 6-gram in the answer):
    CKPT=ckpt/r8-26m python distill/ask.py 提瓦特 原神"""
import os, sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
sys.path.insert(0, os.path.dirname(__file__))
from answer import GenOnlyNoRepeat

P = os.environ.get("CKPT", "ckpt/final-26m-q4")
tok = AutoTokenizer.from_pretrained(P); m = AutoModelForCausalLM.from_pretrained(P).eval()
sysmsg = json.load(open("data/distill/presets.json", encoding="utf-8"))["systems"]["paimon"]
for q in sys.argv[1:]:
    ids = tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt")
    out = m.generate(ids, max_new_tokens=64, do_sample=False, repetition_penalty=1.1,
                     logits_processor=LogitsProcessorList([GenOnlyNoRepeat(int(os.environ.get("NR", "6")), ids.shape[1])] if os.environ.get("NR", "6") != "0" else []))
    print(q, "->", tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())
