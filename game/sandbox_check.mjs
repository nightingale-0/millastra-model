// Run a built client script's main chunk with read-only globals and libraries (like the real
// client sandbox) and report any write: node game/sandbox_check.mjs game/out/scripts/<name>.main.lua
import fs from 'node:fs';
const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const src = fs.readFileSync(process.argv[2], 'utf8');
lua.global.set('SRC_PATH', process.argv[2]);
await lua.doString(`
WRITES = {}
local allowed = { OnStart = true, OnUpdate = true }
local function freeze(t, name)
  local proxy = setmetatable({}, { __index = t, __newindex = function(_, k) error('attempt to modify a readonly table (' .. name .. '.' .. tostring(k) .. ')') end })
  return proxy
end
for _, lib in ipairs({ 'table', 'string', 'math', 'os' }) do _G[lib] = freeze(_G[lib], lib) end
setmetatable(_G, { __newindex = function(t, k, v)
  if allowed[k] then rawset(t, k, v) else WRITES[#WRITES + 1] = tostring(k); error('attempt to modify a readonly table (_G.' .. tostring(k) .. ')') end
end })`);
try {
  await lua.doString(src);
  console.log('main chunk ran; entry points:', await lua.doString('return tostring(OnStart ~= nil) .. "/" .. tostring(OnUpdate ~= nil)'),
              '; blocked global writes (inside pcall):', await lua.doString('return table.concat(WRITES, ",")') || 'none');
} catch (e) { console.log('FAILED:', String(e).slice(0, 300)); process.exitCode = 1; }
