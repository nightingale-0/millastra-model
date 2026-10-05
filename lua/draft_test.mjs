// Draft prefill (Model:draft) must not change answers: type a question character by character
// (with a backspace), feeding drafts, then reply; compare with a plain reply. node lua/draft_test.mjs
import fs from 'node:fs';
const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const luaStr = (t) => '"' + [...Buffer.from(t, 'utf8')].map((b) => String.fromCharCode(92) + String(b).padStart(3, '0')).join('') + '"';
const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const mod = (name, s) => `${name} = (function(...)\n${s}\nend)()\n`;
await lua.doString(mod('BPE', read('./bpe.lua')) + mod('MM', read('./minimind.lua')) + mod('TOKT', read('../build/mm_tok.lua')));
await lua.doString(mod('W', read('../build/mm_weights.lua')));
const Q1 = '派蒙，蒙德城有什么好吃的？', Q2 = '那璃月呢？有没有更辣的菜';
console.log(await lua.doString(`
local loader = MM.loader(W); W = nil; while not loader:step(2000000) do end
local model = loader.model; local tk = BPE.new(TOKT)
local O = { persona = "paimon", temperature = 0, repetition_penalty = 1.1, no_repeat_ngram = 6, max_new = 20 }
local function run(g) local t = {}; while true do local s, d = g:step(1e9); t[#t + 1] = s; if d then break end end; return table.concat(t), table.concat(g.out, ",") end
local function chars(s) local t = {} for _, c in utf8.codes(s) do t[#t + 1] = utf8.char(c) end return t end
local function typeit(msgs, q)          -- one draft per typed character, a backspace in the middle
  local c, text = chars(q), ""
  for i = 1, #c do
    text = text .. c[i]
    if i == 4 then text = text .. "错"; local f = model:draft(tk, msgs, text); while f and not f:step(1e9) do end; text = table.concat(chars(text), "", 1, #chars(text) - 1) end
    local f = model:draft(tk, msgs, text)
    while f and not f:step(1e9) do end
  end
end
local out, ok = {}, true
local sys = { role = "system", content = model:persona("paimon") }
-- plain
model:reset()
local a1, i1 = run(model:reply(tk, { sys, { role = "user", content = ${luaStr(Q1)} } }, O))
local a2, i2 = run(model:reply(tk, { sys, { role = "user", content = ${luaStr(Q1)} }, { role = "assistant", content = a1 }, { role = "user", content = ${luaStr(Q2)} } }, O))
-- drafted
model:reset()
typeit({ sys }, ${luaStr(Q1)})
local g1 = model:reply(tk, { sys, { role = "user", content = ${luaStr(Q1)} } }, O)
local d1 = g1.drafted; local nfeed1 = #g1.feed
local b1, j1 = run(g1)
local m2 = { sys, { role = "user", content = ${luaStr(Q1)} }, { role = "assistant", content = b1 } }
typeit(m2, ${luaStr(Q2)})
m2[#m2 + 1] = { role = "user", content = ${luaStr(Q2)} }
local g2 = model:reply(tk, m2, O)
local d2 = g2.drafted; local nfeed2 = #g2.feed
local b2, j2 = run(g2)
out[#out + 1] = string.format("turn 1: %s; drafted %d tokens, %d left to prefill at send", i1 == j1 and "identical" or "DIFFERENT", d1, nfeed1)
out[#out + 1] = string.format("turn 2: %s; drafted %d tokens, %d left to prefill at send", i2 == j2 and "identical" or "DIFFERENT", d2, nfeed2)
out[#out + 1] = "  " .. b1 .. " | " .. b2
if i1 ~= j1 or i2 ~= j2 then out[#out + 1] = "FAIL" end
return table.concat(out, string.char(10))`));
