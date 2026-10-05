// Quick IME engine check in wasmoon: node game/ime_test.mjs
import fs from 'node:fs';
const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const mod = (name, path) => `${name} = (function(...)\n${fs.readFileSync(new URL(path, import.meta.url), 'utf8')}\nend)()\n`;
console.log(await lua.doString(`
${mod('IME', './ime.lua')}
${mod('DICT', './build/ime_dict.lua')}
local ime = IME.new(DICT)
local out = {}
local function show(label)
  local items, p, n = ime:page_items()
  local w = {}
  for i, c in ipairs(items) do w[i] = c.word end
  out[#out + 1] = string.format('%-12s buf=%-8s text=%-12s cands[%d/%d]: %s', label, ime.buf, ime.text, p, n, table.concat(w, ' '))
end
for _, s in ipairs({'nihao', 'paimeng', 'nh', 'lvxingzhe', 'jintianquna', 'shenme', 'zh', 'liyue', 'tiantianhua', 'xian', 'xianzai', 'women', 'zhongguo', 'shoudu'}) do
  ime:clear(); for ch in s:gmatch('.') do ime:key(ch) end; show(s)
end
ime:clear(); for ch in ('jintianquna'):gmatch('.') do ime:key(ch) end
ime:pick(1); show('pick1'); ime:pick(1); show('pick1 again'); ime:punct('？'); show('punct')
ime:backspace(); show('bksp'); ime:backspace(); show('bksp2')
ime:clear(); for ch in ('shi'):gmatch('.') do ime:key(ch) end; ime:turn(1); show('page2')
return table.concat(out, string.char(10))
`));
