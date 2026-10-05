// Lua heap by stage (what may get the script killed on low-memory devices): node lua/bench_mem.mjs
// The weights module is defined in its own chunk, like the game script, so whether its base85
// literals (chunk constants) are freed after decoding shows up in "after load".
import fs from 'node:fs';
const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const mod = (name, src) => `${name} = (function(...)\n${src}\nend)()\n`;
const heap = `collectgarbage(); collectgarbage(); return collectgarbage("count") / 1024`;
const out = [];
const mark = async (label) => out.push(`${label.padEnd(34)} ${(await lua.doString(heap)).toFixed(1)} MB`);
await mark('empty');
await lua.doString(mod('BPE', read('./bpe.lua')) + mod('MM', read('./minimind.lua')) + mod('TOKT', read('../build/mm_tok.lua')));
await mark('code + tokenizer tables');
await lua.doString(mod('W', read('../build/mm_weights.lua')));
await mark('weights module (base85 text)');
await lua.doString(`LOADER = MM.loader(W); W = nil; while not LOADER:step(2000000) do end; MODEL = LOADER.model; LOADER = nil`);
await mark('after load (text dropped)');
const sizes = await lua.doString(`
local wb, sn = 0, 0
for _, m in pairs(MODEL.mats) do for _, p in ipairs(m.W) do wb = wb + #p end; sn = sn + #m.S end
local lut = 0; for _, l in ipairs(MODEL.lut) do lut = lut + #l end
return string.format("weight strings %.1f MB, scales %d numbers, LUT %d numbers", wb / 1048576, sn, lut)`);
await lua.doString(`TK = BPE.new(TOKT); TOKT = nil
local msgs = { { role = "user", content = "你好呀" } }
for i = 1, 6 do
  local gen = MODEL:reply(TK, msgs, { persona = "paimon", temperature = 0, max_new = 48 })
  local t = {}; while true do local s, d = gen:step(1e9); t[#t + 1] = s; if d then break end end
  msgs[#msgs + 1] = { role = "assistant", content = table.concat(t) }
  msgs[#msgs + 1] = { role = "user", content = "然后呢？再多说一点吧" }
end
POS = MODEL.pos`);
await mark(`after a long chat (pos ${await lua.doString('return POS')})`);
console.log(out.join('\n') + '\n' + sizes);
