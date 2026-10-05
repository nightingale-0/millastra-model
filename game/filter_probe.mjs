// probe the in-game question filter: node game/filter_probe.mjs 你好 原神 ...  -> class|p per input (class nil = allowed)
import fs from 'node:fs';
const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const read = p => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const luaStr = t => '"' + [...Buffer.from(t, 'utf8')].map(b => String.fromCharCode(92) + String(b).padStart(3, '0')).join('') + '"';
const mods = [['IME', './ime.lua'], ['IMEDICT', './build/ime_dict.lua'], ['FILTER', './build/filter.lua']];
await lua.doString(`local MODE='mock' local SUGGEST={} local THEMES={} local THEME_ROLES={} local LAYERS={}\nlocal __m={}\n` +
  mods.map(([n, p]) => `__m['${n}']=(function()\n${read(p)}\nend)()\n`).join('') + read('./chat.lua'));
const qs = process.argv.slice(2).length ? process.argv.slice(2) : ['你好', '你好呀', '你好派蒙', '嗨', 'hi', '你是谁', '早上好', '谢谢', '你好啊'];
console.log(await lua.doString(`local Q={${qs.map(luaStr).join(',')}} local o={} for i,q in ipairs(Q) do local c,p=PAIMON_DEBUG.classify(q); o[i]=tostring(c)..'|'..tostring(p) end return table.concat(o,' ; ')`));
