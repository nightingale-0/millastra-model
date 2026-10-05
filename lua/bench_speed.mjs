// Decode / prefill speed of a minimind.lua variant: node lua/bench_speed.mjs [path/to/minimind.lua]
import fs from 'node:fs';
const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const src = process.argv[2] ? fs.readFileSync(process.argv[2], 'utf8') : read('./minimind.lua');
const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const mod = (name, s) => `${name} = (function(...)\n${s}\nend)()\n`;
await lua.doString(mod('BPE', read('./bpe.lua')) + mod('MM', src) + mod('TOKT', read('../build/mm_tok.lua')));
await lua.doString(mod('W', read('../build/mm_weights.lua')));
console.log(await lua.doString(`
local loader = MM.loader(W); W = nil; while not loader:step(2000000) do end
local model = loader.model; local tk = BPE.new(TOKT); collectgarbage()
local out = {}
local Q = { "派蒙，给我讲讲蒙德城有什么好玩的地方吧？", string.rep("今天我们从蒙德出发，一路走到璃月港，看了很多风景。", 8) .. "你还记得吗？" }
for rep = 1, 4 do
  model:reset()
  local q = Q[(rep - 1) % 2 + 1]
  local marks, c0 = {}, os.clock()
  local gen = model:reply(tk, { { role = "user", content = q } },
    { persona = "paimon", temperature = 0, max_new = 24, on_logits = function() marks[#marks + 1] = os.clock() end })
  while true do local _, d = gen:step(1e9); if d then break end end
  local np = #gen.ids - gen.reused
  out[#out + 1] = string.format("run %d: prefill %3d tokens %.3f s/token | decode at position %3d: %.3f s/token",
    rep, np, (marks[1] - c0) / np, model.pos, (marks[#marks] - marks[1]) / (#marks - 1))
end
return table.concat(out, string.char(10))`));
