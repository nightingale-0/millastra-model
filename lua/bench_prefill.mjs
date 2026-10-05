// TTFT vs prompt length (what a RAG snippet costs): node lua/bench_prefill.mjs
// Same Lua runtime as lua/test.mjs (wasmoon); the real client measured ~1.2x slower.
import fs from 'node:fs';
const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const luaStr = (t) => '"' + [...Buffer.from(t, 'utf8')].map((b) => String.fromCharCode(92) + String(b).padStart(3, '0')).join('') + '"';
const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const mod = (name, src) => `${name} = (function(...)\n${src}\nend)()\n`;
const SNIP = '资料：沙尔·芬德尼尔是龙脊雪山上的古国王城，法鲁希带领族人在此建城，后来寒天之钉落下，整座城被冰雪埋葬，如今只剩雪葬之都的废墟。';
const Q = '芬德尼尔是什么地方？';
const cases = [0, 12, 24, 40, 64].map((n) => (n ? [...SNIP].slice(0, n).join('') + '\n' : '') + Q);
const report = await lua.doString(`
local out = {}
local function log(...) out[#out + 1] = string.format(...) end
${mod('BPE', read('./bpe.lua'))}
${mod('MM', read('./minimind.lua'))}
${mod('TOKT', read('../build/mm_tok.lua'))}
${mod('W', read('../build/mm_weights.lua'))}
local loader = MM.loader(W)
while not loader:step(2000000) do end
local model = loader.model; W = nil; collectgarbage()
local tk = BPE.new(TOKT)
for _, q in ipairs({${cases.map(luaStr).join(',')}}) do
  for rep = 1, 2 do                       -- first run warms up
    model:reset()
    local marks, c0 = {}, os.clock()
    local gen = model:chat(tk, q, { persona = "paimon", temperature = 0, max_new = 6, on_logits = function() marks[#marks + 1] = os.clock() end })
    local text = {}
    while true do local s, done = gen:step(50000); text[#text + 1] = s; if done then break end end
    if rep == 2 then
      log("%3d chars -> %3d new tokens prefilled (%d reused): TTFT %.2f s; decode %.3f s/token | %s", utf8.len(q), #gen.ids - gen.reused, gen.reused,
          marks[1] - c0, #marks > 1 and (marks[#marks] - marks[1]) / (#marks - 1) or 0, table.concat(text))
    end
  end
end
return table.concat(out, "\\n")
`);
console.log(report);
