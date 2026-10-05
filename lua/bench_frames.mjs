// Worst single-frame work during loading and answering (a too-long frame can get the script killed
// on slow devices): node lua/bench_frames.mjs [--budget 8000]
// Loading: one loader:step(budget) per "frame"; answering: one gen:step(budget) per frame.
import fs from 'node:fs';
const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const BUDGET = +arg('--budget', 8000);
const { LuaFactory } = await import('wasmoon');
const lua = await new LuaFactory().createEngine();
const mod = (name, src) => `${name} = (function(...)\n${src}\nend)()\n`;
const report = await lua.doString(`
local out = {}
local function log(...) out[#out + 1] = string.format(...) end
${mod('BPE', read('./bpe.lua'))}
${mod('MM', read('./minimind.lua'))}
${mod('TOKT', read('../build/mm_tok.lua'))}
${mod('W', read('../build/mm_weights.lua'))}
local function frames(stepper)
  local times, n, total = {}, 0, 0
  while true do
    local t0 = os.clock(); local done = stepper(); local dt = os.clock() - t0
    n = n + 1; total = total + dt; times[#times + 1] = { dt, n }
    if done then break end
  end
  table.sort(times, function(a, b) return a[1] > b[1] end)
  return n, total, times
end
local loader = MM.loader(W)
local jt = {}
for i, job in ipairs(loader.jobs) do
  local fn, cost = job.fn, job.cost
  job.fn = function() local t0 = os.clock(); fn(); jt[#jt + 1] = { os.clock() - t0, i, cost } end
end
local n, total, times = frames(function() return loader:step(${BUDGET}) end)
log("load: %d frames, %.1f s total; slowest frames (ms): %s", n, total,
    table.concat((function() local t = {} for i = 1, 8 do t[i] = string.format("%.0f@%d", times[i][1] * 1000, times[i][2]) end return t end)(), " "))
table.sort(jt, function(a, b) return a[1] > b[1] end)
log("slowest jobs (ms@index/cost): %s  of %d", table.concat((function() local t = {} for i = 1, 8 do t[i] = string.format("%.0f@%d/%d", jt[i][1] * 1000, jt[i][2], jt[i][3]) end return t end)(), " "), #jt)
local model = loader.model; W = nil; collectgarbage()
log("Lua heap after load: %.1f MB", collectgarbage("count") / 1024)
local tk = BPE.new(TOKT)
local gen = model:chat(tk, "芬德尼尔是什么地方？", { persona = "paimon", temperature = 0, max_new = 16 })
n, total, times = frames(function() local _, done = gen:step(${BUDGET}); return done end)
log("answer: %d frames, %.1f s total; slowest frames (ms): %.0f %.0f %.0f; median %.0f", n, total,
    times[1][1] * 1000, times[2][1] * 1000, times[3][1] * 1000, times[math.floor(#times / 2)][1] * 1000)
return table.concat(out, "\\n")
`);
console.log(report);
