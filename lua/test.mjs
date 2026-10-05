// Checks the Lua runtime against build/golden.json (from export/export_lua.py) and times it.
//   node lua/test.mjs [--tokens 24]
// wasmoon breaks after a few dozen JS<->Lua string round trips (heap growth), so
// all data goes in as Lua source and each phase is a single doString.
import fs from 'node:fs';
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const luaStr = (t) => '"' + [...Buffer.from(t, 'utf8')].map((b) => String.fromCharCode(92) + String(b).padStart(3, '0')).join('') + '"';
const golden = JSON.parse(read('../build/golden.json'));
const N = +arg('--tokens', 24);

const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const mod = (name, src) => `${name} = (function(...)\n${src}\nend)()\n`;
const cases = golden.tok.map((c) => `{${luaStr(c.text)},"${c.ids.join(',')}"}`).join(',\n');

const report = await lua.doString(`
do local c = table.concat; table.concat = function(t, ...) local r = c(t, ...); if #r > 49152 then error('table.concat result too large (' .. #r .. ' bytes)') end; return r end end
local out = {}
local function log(...) out[#out + 1] = string.format(...) end
local c0 = os.clock()
${mod('BPE', read('./bpe.lua'))}
${mod('MM', read('./minimind.lua'))}
${mod('TOKT', read('../build/mm_tok.lua'))}
local c1 = os.clock()
${mod('W', read('../build/mm_weights.lua'))}
log("parse: code+tokenizer %.2f s, weights table %.2f s", c1 - c0, os.clock() - c1)

local loader = MM.loader(W)
c0 = os.clock()
local steps = 1
while not loader:step(2000000) do steps = steps + 1 end
local model = loader.model
W = nil
collectgarbage()
log("decode weights: %.2f s in %d steps of 2M; Lua heap %.1f MB", os.clock() - c0, steps, collectgarbage("count") / 1024)

local tk = BPE.new(TOKT)
local CASES = {${cases}}
local bad = 0
c0 = os.clock()
for _, c in ipairs(CASES) do
  local got = table.concat(tk:encode(c[1]), ",")
  if got ~= c[2] then
    bad = bad + 1
    if bad <= 3 then log("TOK MISMATCH %q\\n  want %s\\n  got  %s", c[1]:sub(1, 60), c[2], got) end
  end
end
log("tokenizer: %d/%d identical (%.2f s)", #CASES - bad, #CASES, os.clock() - c0)

-- chat cases: a fully cached preset, a filled template slot, an uncached prompt
local CHAT = {${[...golden.chat, { ...golden.chat[golden.chat.length - 1], rewind: true, follows: undefined }].map((c) => `{persona=${luaStr(c.system)}, user=${luaStr(c.user)}, follows=${c.follows !== undefined}, rewind=${!!c.rewind}, ids="${c.ids.join(',')}", greedy="${c.greedy.join(',')}",
  logits={${c.logits.map((v) => v.toPrecision(9)).join(',')}}}`).join(',\n')}}
for _, c in ipairs(CHAT) do
  local maxd, marks = 0, {}
  local c0 = os.clock()
  local o = { persona = c.persona, temperature = 0, max_new = ${N}, on_logits = function(l)
    marks[#marks + 1] = os.clock()
    if #marks == 1 then       -- cached logits hold only the top 64: compare those
      for i = 1, #l do if l[i] > -1e29 then maxd = math.max(maxd, math.abs(l[i] - c.logits[i])) end end
    end
  end }
  local gen
  if c.follows then gen = model:say(tk, c.user, o)
  elseif c.rewind then          -- back to the end of the first answer, ask again
    local msgs = model:messages()
    msgs[#msgs] = nil; msgs[#msgs] = nil
    msgs[#msgs + 1] = { role = "user", content = c.user }
    gen = model:reply(tk, msgs, o)
  else gen = model:chat(tk, c.user, o) end
  local text = {}
  while true do
    local s, done = gen:step(50000)
    text[#text + 1] = s
    if done then break end
  end
  local gi = table.concat(gen.out, ",")
  log("[%s] %s  ids %s, %d/%d tokens reused", c.persona, c.rewind and (c.user .. " (rewind)") or c.user, table.concat(gen.ids, ",") == c.ids and "ok" or "MISMATCH", gen.reused, #gen.ids)
  log("  first logits max |diff| %.2e; greedy %s: %s", maxd, gi == c.greedy and "identical" or "DIFFERENT", table.concat(text))
  if gi ~= c.greedy then log("  want %s\\n  got  %s", c.greedy, gi) end
  log("  TTFT %.3f s (%d tokens prefilled); decode %.3f s/token",
      marks[1] - c0, #gen.ids - gen.reused, #marks > 1 and (marks[#marks] - marks[1]) / (#marks - 1) or 0)
end
return table.concat(out, "\\n")
`);
console.log(report);
