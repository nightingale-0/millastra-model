// Where decode time goes, by operator kind, at a short and a long context: node lua/bench_ops.mjs
// Prefill is not timed; 16 decode steps are.
import fs from 'node:fs';
const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const mod = (name, src) => `${name} = (function(...)\n${src}\nend)()\n`;
await lua.doString(mod('BPE', read('./bpe.lua')) + mod('MM', read('./minimind.lua')) + mod('TOKT', read('../build/mm_tok.lua')));
await lua.doString(mod('W', read('../build/mm_weights.lua')));
console.log(await lua.doString(`
local loader = MM.loader(W); W = nil; while not loader:step(2000000) do end
local model = loader.model; local tk = BPE.new(TOKT)
local KIND = { "norm+LUT", "qkv matvec", "RoPE+KV append", "attention (incl. softmax exp)", "LUT", "o matvec",
               "residual+norm+LUT", "gate/up matvec", "SiLU (exp)+LUT", "down matvec", "residual" }
local body, head = model.ops_body, model.ops_head
local pre = #body - 11 * model.nl
local acc = {}
local function wrap(op, label)
  local fn = op.fn
  op.fn = function(...) local t0 = os.clock(); fn(...); acc[label] = (acc[label] or 0) + os.clock() - t0 end
end
for i, op in ipairs(body) do wrap(op, i <= pre and "embedding/RoPE tables" or KIND[(i - pre - 1) % 11 + 1]) end
wrap(head[1], "final norm+LUT"); wrap(head[2], "LM head matvec (6400 rows)")
local out = {}
local function profile(text, label)
  model:reset()
  local gen = model:reply(tk, { { role = "user", content = text } }, { persona = "paimon", temperature = 0, max_new = 17 })
  local timing = false
  while true do
    local _, d = gen:step(20000)
    if not timing and #gen.out >= 1 then timing = true; for k in pairs(acc) do acc[k] = nil end end
    if d then break end
  end
  local tot = 0; for _, v in pairs(acc) do tot = tot + v end
  local rows = {}
  for k, v in pairs(acc) do rows[#rows + 1] = { k, v } end
  table.sort(rows, function(x, y) return x[2] > y[2] end)
  out[#out + 1] = string.format("%s (position %d), %.3f s/token:", label, model.pos, tot / (#gen.out - 1))
  for _, r in ipairs(rows) do out[#out + 1] = string.format("  %-30s %5.1f%%", r[1], 100 * r[2] / tot) end
end
profile("你好呀", "short context")
profile(string.rep("今天我们从蒙德出发，一路走到璃月港，看了很多风景，也吃了很多好吃的东西。", 10) .. "你还记得吗？", "long context")
return table.concat(out, "\\n")`));
