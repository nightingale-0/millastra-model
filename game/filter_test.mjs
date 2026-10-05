// Lua question filter vs the Python reference (game/embed/filter_ref.py): node game/filter_test.mjs
import fs from 'node:fs';
const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const read = p => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const ref = JSON.parse(read('./build/filter_ref.json'));
const luaStr = t => '"' + [...Buffer.from(t, 'utf8')].map(b => String.fromCharCode(92) + String(b).padStart(3, '0')).join('') + '"';
const mods = [['IME', './ime.lua'], ['IMEDICT', './build/ime_dict.lua'], ['FILTER', './build/filter.lua'], ['LORE', './build/lore.lua']];
const src = `local MODE='mock' local SUGGEST={} local THEMES={} local THEME_ROLES={} local LAYERS={}\nlocal __m={}\n` +
  mods.map(([n, p]) => `__m['${n}']=(function()\n${read(p)}\nend)()\n`).join('') + read('./chat.lua');
await lua.doString(src);
const res = await lua.doString(`local Q = {${ref.map(r => luaStr(r.q)).join(',')}}
local out = {}
for i, q in ipairs(Q) do local cat, p = PAIMON_DEBUG.classify(q); out[i] = string.format('%s|%.6f', cat or '', p or -1) end
return table.concat(out, string.char(10))`);
let maxd = 0, n = 0, kw = 0, agree = 0;
res.split('\n').forEach((line, i) => {
  const [cat, p] = line.split('|'); const pr = ref[i].p;
  if (+p === 1 && cat) { kw++; return; }                 // keyword hit: no probability to compare
  n++; maxd = Math.max(maxd, Math.abs(+p - pr)); if ((+p >= 0.6) === (pr >= 0.6)) agree++;
});
console.log(`compared ${n} (keyword hits ${kw}); max |p_lua - p_py| = ${maxd.toExponential(2)}; same decision ${agree}/${n}`);
const last = res.split('\n').slice(-4);
console.log(ref.slice(-4).map((r, i) => `${r.q}: ${last[i]}`).join('\n'));

// retrieval: names and aliases hit their entry, ordinary questions do not
const LOOK = [['芬德尼尔是什么地方？', '沙尔·芬德尼尔'], ['迪卢克是谁', '迪卢克'], ['你好呀派蒙', null], ['今天天气怎么样', null], ['我想去沙漠玩', null],
  ['甘雨是谁？', null], ['夏洛蒂的工作是什么？', '夏洛蒂'], ['派蒙，地球是圆的还是平的？如果你说是圆的，那我从璃月走到蒙德怎么没绕回去？', null], ['要是我和钟离吵架了，你站哪边？', null], ['说说望风角', '望风角'],
  ['夏洛蒂是不是去过挪德卡莱？', '夏洛蒂'], ['挪德卡莱在哪？', '挪德卡莱'], ['刻晴和甘雨是什么关系？', '刻晴'], ['璃月的神是谁？', null], ['蒙德有什么好吃的？', null]];
const got = (await lua.doString(`local o = {} for i, q in ipairs({${LOOK.map(([q]) => luaStr(q)).join(',')}}) do o[i] = PAIMON_DEBUG.lookup(q) or '-' end return table.concat(o, string.char(10))`)).split(String.fromCharCode(10));
let okL = 0;
LOOK.forEach(([q, want], i) => { const hit = got[i] === '-' ? null : got[i].split('：')[0]; if (hit === want) okL++; else console.log(`lookup MISMATCH ${q}: ${got[i]}`); });
console.log(`lookup ${okL}/${LOOK.length}: ${got[0]}`);
if (okL !== LOOK.length) process.exit(1);
