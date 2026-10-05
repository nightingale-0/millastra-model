-- TeyvatAI 派蒙 (client script). Prepended by game/build.mjs: MODE ('mock'|'full'), SUGGEST,
-- THEMES, THEME_ROLES, LAYERS and __m (bundled modules: IME, IMEDICT, FILTER and, on the
-- full build, BPE, MM, TOKT, W). Everything lives in this one script: the sandbox has no require.
--
-- A small offline "AI engine":
--  * a question filter runs first (the custom IME bypasses the platform's chat filter):
--    linear classifier over hashed character n-grams + a keyword list; answers are checked too
--  * real multi-turn context (model:reply with the whole message list; the runtime reuses
--    the K/V of every leading turn it still holds) with a 512-token window; when a new turn
--    would not fit, the oldest exchanges are dropped and a note says so
--  * 回退 on any of your messages: back to just before it, the text returns to the input
--  * per-frame work budget follows the frame time; stats: first-token latency, chars/s, context
local IME = __m.IME
local ime = IME.new(__m.IMEDICT); __m.IMEDICT = nil

local root
local loader, model, tk, gen
local turn                       -- the exchange in progress: {q, id, t0, tFirst, chars, src, lookup}
local msgs, ctx = {}, {}        -- UI messages {who='p'|'u'|'s', text, id}; model context {role, content, id, toks}
local nextId, scroll, compressed = 0, 0, 0
local pending                   -- a question waiting for the model to finish loading
-- crash safety: every entry point (OnStart, OnUpdate, clicks, keys) runs under xpcall. An error
-- ends the current answer (or the loading) with a message instead of leaving the UI waiting
-- forever; errors and periodic diagnostics go to the log as "[paimon] ..." for testers' reports.
local TURN_TIMEOUT = 180                -- seconds: an answer still running after this is ended
local traceback = (debug and debug.traceback) or function(e) return tostring(e) end
local errCount, loadFailed, recover = 0, false, nil
local diag = { maxDt = 0, last = 0, t0 = 0 }
local function heapMB() local ok, kb = pcall(collectgarbage, 'count'); return ok and kb / 1024 or -1 end
local function guard(where, fn)
  return function(...)
    local ok, err = xpcall(fn, traceback, ...)
    if not ok then
      errCount = errCount + 1
      print(string.format('[paimon] error #%d in %s (heap %.1f MB): %s', errCount, where, heapMB(), tostring(err)))
      if recover then pcall(recover, where) end
    end
  end
end
-- Work budget per frame (unit ~ one weight-plane lookup; a token costs ~13M). It follows the frame
-- time relative to the game's own frame time when the model is idle (idleDt): while working, a frame
-- may take max(1/20 s, 2.5 x idleDt) (the UI is still while Paimon thinks). A fixed fps target starved slow devices (whose idle fps is
-- already low: the budget fell to its floor, ~30 s per token) and capped fast ones (600k/frame at
-- 60 fps = 2.8 tok/s, while the CPU could do ~5).
local budget, BUDGET_MIN, BUDGET_MAX = 60000, 30000, 3000000
local idleDt, idleN, WARMUP = 1 / 60, 0, 0.5        -- seconds of idle frames measured before loading
-- draft prefill (lua/minimind.lua Model:draft): while the player types, the request so far is fed in
-- idle frames, so sending only has the end-of-turn template left (~7 tokens instead of ~25)
local draftText, feeder = nil, nil
local clock = 0
-- 结算: 30 s after entering, the server node graph is told to mark the player as won (PaimonWin);
-- from then on the 结算 button asks it to settle the level (PaimonSettle). Earlier clicks do nothing.
local SETTLE_AFTER, won, settleShown = 30, false, nil
local function signal(name) pcall(function() game.ServerSignal(name):SendSignal() end) end
local theme, menuOpen = 'light', false
local stats = ''
local SLOTS, CTX_LIMIT, MAX_NEW, SYS_TOKENS = 5, 512, 64, 80
local GEN_OPTS = { temperature = 0, repetition_penalty = 1.1, no_repeat_ngram = 6, max_new = MAX_NEW }

local function chars(s) local t = {}; for c in s:gmatch('[%z\1-\127\194-\244][\128-\191]*') do t[#t + 1] = c end; return t end

-- ---------------------------------------------------------------- question filter
-- Same features as game/embed/train_filter.py: punctuation dropped, ASCII lower-cased; character
-- 1/2/3-grams hashed over their UTF-8 bytes with h = (h*61 + b) % 16777213 (salt = n), which stays
-- below 2^31 so 32-bit (fengari) and 64-bit Lua agree. logits = mean of int8 rows * scale + bias.
local PUNCT = {}
for _, c in ipairs(chars('，。？！、：；“”‘’（）…—～《》「」『』【】·')) do PUNCT[c] = true end
local function normText(q)
  local t = {}
  for _, c in ipairs(chars(q)) do
    if #c > 1 then if not PUNCT[c] then t[#t + 1] = c end
    elseif c:match('%w') then t[#t + 1] = c:lower() end
  end
  return t
end
local function dec4(s)
  local a, b, c, d = s:byte(1, 4)
  local n = ((a - 63) << 18) | ((b - 63) << 12) | ((c - 63) << 6) | (d - 63)
  return string.char(n >> 16, (n >> 8) & 255, n & 255)
end
local B85 = {}
do local k = 0; for c = 33, 126 do local ch = string.char(c); if c ~= 61 and not ('aeiouUIO'):find(ch, 1, true) and k < 85 then B85[c] = k + 0.0; k = k + 1 end end end   -- = export_lua.py B85
local function dec5(s)                            -- base85, in floats (32-bit-integer Lua builds too)
  local a, b, c, d, e = s:byte(1, 5)
  local n = (((B85[a] * 85 + B85[b]) * 85 + B85[c]) * 85 + B85[d]) * 85 + B85[e]
  local b1 = n // 16777216; n = n - b1 * 16777216
  local b2 = n // 65536; n = n - b2 * 65536
  local b3 = n // 256
  return string.char(math.floor(b1), math.floor(b2), math.floor(b3), math.floor(n - b3 * 256))
end
-- a blob decodes into parts of whole units (never joined: the sandbox rejects big concat results)
local function blob(b)
  local parts, step = { per = b.per, unit = b.unit }, b.per * b.unit
  for i = 1, #b do
    local s = b[i]
    if s:find('=', 1, true) then s = s:gsub('=', '') end   -- word breakers (export_lua.break_flag_words)
    s = s:gsub('.....', dec5)
    local want = math.min(step, b.n - (i - 1) * step)
    parts[i] = #s > want and s:sub(1, want) or s
  end
  return parts
end
local F = __m.FILTER; __m.FILTER = nil
-- retrieval (game/make_lore.py): a lore name in the question puts "（资料：名字：snippet）" in front of
-- it, as in training (distill/gen_lore_rag.py); a snippet is decoded only when its name is asked about
local LORE = __m.LORE; __m.LORE = nil
F.W = blob(F.W)
-- keywords arrive only as hashes (no sensitive text in the script; game/embed/train_filter.py
-- kw_hash): two 24-bit hashes of the substring's bytes -> class index (1 = ok)
local function kwKey(c, i, j)
  local h1, h2 = 7, 11
  for k = i, j do
    local s = c[k]
    for b = 1, #s do local v = s:byte(b); h1 = (h1 * 61 + v) % 16777213; h2 = (h2 * 67 + v) % 16777199 end
  end
  return h1 * 16777216.0 + h2
end
local function keywordClass(c)
  for i = 1, #c - 1 do
    for j = i + 1, math.min(#c, i + F.kwmax - 1) do
      local k = F.kw[kwKey(c, i, j)]
      if k then return k end
    end
  end
end
local function bucket(s, n)
  local h = n
  for i = 1, #s do h = (h * 61 + s:byte(i)) % 16777213 end
  return h % F.buckets
end
-- block threshold by length: short input leans on the keyword rules ("你好" / "原神" once scored
-- above 0.6), longer input on the classifier (train_filter.py threshold)
local function threshold(n)
  local ts, tl, ns, nl = F.th[1], F.th[2], F.th[3], F.th[4]
  if n <= ns then return ts elseif n >= nl then return tl end
  return ts + (tl - ts) * (n - ns) / (nl - ns)
end
local function classify(q)
  local c = normText(q)
  local kc = keywordClass(c)
  if kc then return kc, 1 end
  local feats = {}
  for i = 1, #c do feats[#feats + 1] = bucket(c[i], 1) end
  for i = 1, #c - 1 do feats[#feats + 1] = bucket(c[i] .. c[i + 1], 2) end
  for i = 1, #c - 2 do feats[#feats + 1] = bucket(c[i] .. c[i + 1] .. c[i + 2], 3) end
  if #feats == 0 then return nil end
  local C, logit = F.classes, {}
  for k = 1, C do logit[k] = 0 end
  for _, f in ipairs(feats) do
    local ci = f // F.W.per
    local o = (f - ci * F.W.per) * C
    local row = { F.W[ci + 1]:byte(o + 1, o + C) }
    for k = 1, C do local v = row[k]; logit[k] = logit[k] + (v >= 128 and v - 256 or v) end
  end
  local mx, best, bestK = -math.huge, -math.huge, 2
  for k = 1, C do
    logit[k] = logit[k] / #feats * F.scale + F.bias[k]
    if logit[k] > mx then mx = logit[k] end
    if k > 1 and logit[k] > best then best, bestK = logit[k], k end
  end
  local z = 0
  for k = 1, C do z = z + math.exp(logit[k] - mx) end
  local pBlock = 1 - math.exp(logit[1] - mx) / z
  if pBlock >= threshold(#c) then return bestK, pBlock end
  return nil, pBlock
end
local ASK = { '是什么', '是谁', '是啥', '在哪', '哪里', '哪儿', '介绍', '讲讲', '说说', '聊聊', '知道', '认识', '听说',
  '怎么样', '厉害', '什么样', '什么地方', '干什么', '做什么', '有什么', '是哪', '来历', '故事', '了解',
  '是不是', '有没有', '会不会', '去过', '见过', '喜欢', '关系', '和谁', '属于' }
local LORE_MAX = 2                                   -- snippets per question (each ~15-20 tokens of prefill)
local function decodeEntry(k)
  local e = LORE.e[k]
  local s = e[2]:gsub('.....', dec5)
  return s:sub(1, e[1])
end
-- Lore names in the question, left to right without overlap (longest at each position), at most
-- LORE_MAX entries; returned only when the question is about them: what is left after removing the
-- names is short, or asks (是什么 / 是不是 / 去过…), or the first name is the subject at the start.
-- A name mentioned in passing ("从璃月走到蒙德…") made the model repeat the snippet instead of
-- answering (distill/probe_suite.py: 1.6 vs 3.1 correct).
local function lookup(q)
  if not LORE then return nil end
  local c, hits, seen, inName = normText(q), {}, {}, {}
  local i = 1
  while i < #c do
    local found = false
    for j = math.min(#c, i + LORE.kmax - 1), i + 1, -1 do
      local k = LORE.idx[kwKey(c, i, j)]
      if k then
        if not seen[k] and #hits < LORE_MAX then seen[k] = true; hits[#hits + 1] = { k = k, i = i } end
        for x = i, j do inName[x] = true end
        i = j + 1; found = true; break
      end
    end
    if not found then i = i + 1 end
  end
  if #hits == 0 then return nil end
  local s0 = 1                                       -- skip a leading 派蒙 / 旅行者
  if c[1] == '派' and c[2] == '蒙' then s0 = 3 elseif c[1] == '旅' and c[2] == '行' and c[3] == '者' then s0 = 4 end
  local rest = {}
  for x = s0, #c do if not inName[x] then rest[#rest + 1] = c[x] end end
  rest = table.concat(rest)
  for _ = 1, 4 do rest = rest:gsub('派蒙$', ''):gsub('呀$', ''):gsub('啊$', ''):gsub('呢$', ''):gsub('吗$', ''):gsub('吧$', ''):gsub('哦$', ''):gsub('嘛$', '') end
  local n = select(2, rest:gsub('[^\128-\191]', ''))   -- characters (no utf8 library in some sandboxes)
  local ask = false
  for _, w in ipairs(ASK) do if rest:find(w, 1, true) then ask = true; break end end
  if not (n <= 6 or (n <= 14 and ask) or (hits[1].i == s0 and n <= 16)) then return nil end
  local out = {}
  for x, h in ipairs(hits) do out[x] = decodeEntry(h.k) end
  return table.concat(out, '；')
end
local function unsafeAnswer(text) return keywordClass(normText(text)) end   -- keyword check on generated text
local REFUSE_DEFAULT = '唔……这个话题派蒙不能聊哦。换个问题吧，比如今天想去哪里冒险？'
-- a canned reply is "typed out" a few characters per frame
local function streamText(text)
  local c, i = chars(text), 0
  return { step = function() i = i + 2; local s = table.concat(c, '', i - 1, math.min(i, #c)); return s, i >= #c end }
end

-- ---------------------------------------------------------------- mock model (UI build)
local Mock = { pos = 0 }
function Mock.loader()
  local L = { n = 0 }
  function L:step() self.n = self.n + 1; if self.n >= 40 then self.model = Mock end; return self.n >= 40 end
  function L:progress() return math.min(1, self.n / 40) end
  return L
end
function Mock:persona() return '你是派蒙' end
function Mock:reply(_, list)
  local q = list[#list].content
  local reply = '派蒙听到啦：「' .. q .. '」——这是界面测试版，还没装上派蒙的脑袋哦。'
  local c, i, t = chars(reply), 0, 0
  local total = 0
  for _, m in ipairs(list) do total = total + #chars(m.content) end
  return { step = function()
    t = t + 1
    if t < 8 then return '', false end
    if q == 'crash' and t == 12 then error('mock fault (game/test.mjs crash check)') end
    i = i + 1
    if i >= #c then Mock.pos = total + #c end
    return c[i] or '', i >= #c
  end }
end

-- ---------------------------------------------------------------- helpers
local function control(name) return root:FindChild(name) end
local function text(name, value) local c = control(name); if c then c.text = tostring(value) end end
local function visible(name, v) local c = control(name); if c then c:SetActive(v and true or false) end end
local function click(name, fn) local c = control(name); if c then c:AddCursorEventListener(Enum.CursorEventType.CursorClick, guard('click ' .. name, fn)) end end
local ALL = {}
local busy = function() return gen ~= nil end
local ready = function() return model ~= nil end
local function tokens(s)
  if MODE == 'full' and tk then return #tk:encode(s) end
  return math.floor(#chars(s) * 0.8) + 1                  -- mock estimate
end
-- tokens the next turn starts from: system prompt + every exchange still in the context
local function contextUsed()
  local n = SYS_TOKENS
  for _, e in ipairs(ctx) do e.toks = e.toks or tokens(e.content) + 6; n = n + e.toks end
  return #ctx > 0 and n or 0
end
-- your-message bubble: width class from the estimated text width (CJK 15 px, ASCII 8 px)
local BUBBLE = { S = 200, M = 340, L = 512 }
local function bubbleFor(t)
  local w = 28
  for _, c in ipairs(chars(t)) do w = w + (#c > 1 and 15 or 8) end
  if w <= 200 then return 'S' elseif w <= 340 then return 'M' elseif w <= 512 then return 'L' end
  return 'T'
end
local backX = {}

local function applyTheme()
  local p = THEMES[theme]
  for _, r in ipairs(THEME_ROLES) do
    local c = control(r[1])
    if c then pcall(function() if r[2] == 'bg' then c.bgColor = p[r[3]] else c.fontColor = p[r[3]] end end) end
  end
  settleShown = nil                                -- settleTick recolours 结算 for the new theme
end

-- grey status in Paimon's slot until her first character arrives
local function waitText()
  local s = math.floor(clock - turn.t0)
  if not ready() then return string.format('派蒙正在醒来…（%d秒）', s) end
  return string.format(turn.lookup and '派蒙正在查阅资料…（%d秒）' or '派蒙思考中…（%d秒）', s)
end
local KEY_PUNCT = { KeyComma = { '，', ', ' }, KeyQuestion = { '？', '?' }, KeyExclaim = { '！', '!' } }   -- 中 / 英
local function render()
  local last = #msgs - scroll
  local first = math.max(1, last - SLOTS + 1)
  for k = 1, SLOTS do
    local i = first + k - 1
    local m = i <= last and msgs[i] or nil
    local w = m and m.who
    visible('MsgP' .. k, w == 'p'); visible('MsgP' .. k .. 'Icon', w == 'p')
    local b = w == 'u' and bubbleFor(m.text)
    visible('MsgU' .. k, b and b ~= 'T'); visible('MsgU' .. k .. 'B', b == 'T')
    for _, v in ipairs({ 'S', 'M', 'L', 'T' }) do visible('MsgU' .. k .. v, b == v) end
    visible('Back' .. k, w == 'u' and not busy() and not menuOpen)
    if b then                                      -- 回退 sits just left of the bubble
      local c = control('Back' .. k)
      pcall(function()
        backX[k] = backX[k] or c.anchoredPositionX
        c.anchoredPositionX = backX[k] + (512 - (BUBBLE[b] or 512))
      end)
    end
    visible('MsgS' .. k, w == 's')
    local waiting = w == 'p' and m.text == '' and turn and m == msgs[#msgs]
    visible('MsgW' .. k, waiting); if waiting then visible('MsgP' .. k, false); text('MsgW' .. k, waitText()) end
    if w == 'p' and not waiting then text('MsgP' .. k, m.text == '' and '……' or m.text) end
    if b then text(b == 'T' and ('MsgU' .. k .. 'B') or ('MsgU' .. k), m.text) end
    if w == 's' then text('MsgS' .. k, m.text) end
  end
  visible('Hero', #msgs == 0); visible('HeroSub', #msgs == 0)
  visible('ScrollUp', first > 1); visible('ScrollDown', scroll > 0)
  -- composer
  text('InputText', ime.text ~= '' and ime.text or (ime.buf ~= '' and '' or '问问派蒙…'))
  text('Pinyin', ime.buf)
  local items, page, pages = ime:page_items()
  for k = 1, 7 do
    local c = items[k]
    visible('Cand' .. k, c ~= nil)
    if c then text('Cand' .. k .. '/Cand' .. k .. 'Text', c.word) end
  end
  visible('CandPrev', page > 1); visible('CandNext', page < pages)
  local hint
  if ime.buf ~= '' then hint = ''
  elseif not ready() then hint = string.format('派蒙正在醒来… %d%%', math.floor((loader and loader:progress() or 1) * 100))
  else hint = string.format('上下文 %d / %d%s', contextUsed(), CTX_LIMIT, compressed > 0 and ('，已压缩 ' .. compressed .. ' 轮') or '') end
  text('Hint', hint)
  text('Lang/LangText', ime.en and '英' or '中')
  text('KeySpace/KeySpaceText', ime.en and '空格' or '空格选字')
  for k, p in pairs(KEY_PUNCT) do text(k .. '/' .. k .. 'Text', ime.en and p[2] or p[1]) end
  text('Send/SendText', busy() and '…' or '发送')
  text('Theme/ThemeText', theme == 'light' and '深色' or '浅色')
  text('Stats', stats)
  for k = 1, #SUGGEST do visible('Menu' .. k, menuOpen) end
  visible('MenuTitle', menuOpen); visible('MenuPanel', menuOpen); visible('MenuEdge', menuOpen)
end

local function newId() nextId = nextId + 1; return nextId end

-- drop the oldest exchanges until system + context + the new question + the answer fit
local function fitContext(q)
  local total = SYS_TOKENS + tokens(q) + 8 + MAX_NEW
  for _, e in ipairs(ctx) do e.toks = e.toks or tokens(e.content) + 6; total = total + e.toks end
  local dropped = 0
  while total > CTX_LIMIT and #ctx >= 2 do
    total = total - ctx[1].toks - ctx[2].toks
    table.remove(ctx, 1); table.remove(ctx, 1)
    dropped = dropped + 1
  end
  if dropped > 0 then
    compressed = compressed + dropped
    table.insert(msgs, #msgs - 1, { who = 's', text = string.format('— 上下文已满，较早的 %d 轮对话已压缩 —', dropped), id = newId() })   -- before this question
  end
end

-- (turn: the exchange in progress, declared at the top)
local function startDraft(text)
  local list = { { role = 'system', content = model:persona('paimon') } }
  for _, e in ipairs(ctx) do list[#list + 1] = { role = e.role, content = e.content } end
  local lore = lookup(text)
  return model:draft(tk, list, lore and ('（资料：' .. lore .. '）\n' .. text) or text)
end

local function startModel(q)
  -- what the model sees: the question, with the retrieved snippet in front (kept in the history as sent,
  -- so the next turn reuses this K/V instead of prefilling it again)
  feeder, draftText = nil, nil                    -- reply() parks a running draft and uses what it fed
  turn.sent = turn.lore and ('（资料：' .. turn.lore .. '）\n' .. q) or q
  fitContext(turn.sent)
  local list = { { role = 'system', content = model:persona('paimon') } }
  for _, e in ipairs(ctx) do list[#list + 1] = { role = e.role, content = e.content } end
  list[#list + 1] = { role = 'user', content = turn.sent }
  if MODE == 'full' then gen = model:reply(tk, list, GEN_OPTS) else gen = model:reply(nil, list) end
  turn.src = 'model'
end

local function ask(q)
  if busy() or q == '' then return end
  menuOpen = false
  local id = newId()
  msgs[#msgs + 1] = { who = 'u', text = q, id = id }
  msgs[#msgs + 1] = { who = 'p', text = '', id = id }
  scroll = 0
  turn = { q = q, id = id, t0 = clock, chars = 0, raw = {} }
  local cat = classify(q)
  if not cat then turn.lore = lookup(q); turn.lookup = turn.lore ~= nil end
  if cat then gen = streamText(F.refuse[cat] ~= '' and F.refuse[cat] or REFUSE_DEFAULT); turn.src = 'filter'; turn.cat = cat
  elseif ready() then startModel(q)
  elseif loadFailed then gen = streamText('唔…派蒙好像醒不过来了，重新进入奇域试试吧。'); turn.src = 'filter'
  else pending = turn; gen = { step = function() return '', false end }; turn.src = 'wait' end
  stats = turn.src == 'filter' and '已拦截' or '派蒙在想…'
end

local function finishTurn()
  local m = msgs[#msgs]
  local raw = table.concat(turn.raw)
  local bad = turn.src == 'model' and unsafeAnswer(raw)
  if bad then raw = REFUSE_DEFAULT end
  if turn.src == 'model' then                   -- refused questions stay out of the model's context
    ctx[#ctx + 1] = { role = 'user', content = turn.sent or turn.q, id = turn.id }
    ctx[#ctx + 1] = { role = 'assistant', content = raw, id = turn.id }
  end
  m.text = raw:gsub('^%s+', ''):gsub('%s+$', '')
  if m.text == '' then m.text = '唔…派蒙不知道该说什么了。' end
  if turn.src == 'filter' then stats = '已拦截'
  elseif bad then stats = '回答未通过检查'
  else
    local speed = turn.tFirst and clock > turn.tFirst and (turn.chars - 1) / (clock - turn.tFirst) or 0
    stats = string.format('首字 %.2fs · %.1f 字/秒', (turn.tFirst or clock) - turn.t0, speed)
    print(string.format('[paimon] answer: %s, %d chars, heap %.1f MB', stats, turn.chars, heapMB()))
  end
  gen, turn = nil, nil
end

-- after an error (or a too-long answer): end what was running, keep the UI usable
recover = function(where)
  if loader then
    loader, loadFailed, pending = nil, true, nil
    stats = '派蒙加载失败（日志里有详情）'
  end
  if turn then
    local m = msgs[#msgs]
    if m and m.who == 'p' then
      m.text = (m.text ~= '' and (m.text .. '……') or '') ..
        (loadFailed and '唔…派蒙好像醒不过来了，重新进入奇域试试吧。' or '唔…派蒙脑袋卡住了，换个说法再问一次吧。')
    end
    gen, turn = nil, nil
    if model then pcall(function() model:reset() end) end   -- the K/V state may be half-written
    if not loadFailed then stats = where == 'timeout' and '回答超时，已重置这一轮' or '出错了，已重置这一轮' end
  end
  pcall(render)
end

-- 回退: back to just before message i (yours); its text goes back into the input
local function rewind(i)
  if busy() then return end
  local m = msgs[i]
  for j = #msgs, i, -1 do msgs[j] = nil end
  for j = #ctx, 1, -1 do if ctx[j].id >= m.id then ctx[j] = nil end end
  ime:clear(); ime.text = m.text
  scroll = 0
  stats = '已回退到这一轮'
end

local function settleTick()
  if not won and clock >= SETTLE_AFTER then won = true; signal('PaimonWin') end
  local label = won and '结算' or string.format('结算 %ds', math.ceil(SETTLE_AFTER - clock))
  if label ~= settleShown then
    settleShown = label; text('Settle/SettleText', label)
    pcall(function() control('Settle/SettleText').fontColor = THEMES[theme][won and 'ink' or 'muted'] end)
  end
end

local function send()
  if busy() then return end
  if ime.buf ~= '' then ime:pick(1) end
  local q = ime.text
  if q ~= '' then ime:clear(); ask(q) end
end

-- physical keyboard (PC): bind 奇域自定义按键 1-26 to A-Z, 27 删除, 28 空格, 29 回车 in the level settings
local KEYMAP = {}
for i = 1, 26 do KEYMAP[i] = string.char(96 + i) end
KEYMAP[27], KEYMAP[28], KEYMAP[29] = 'back', 'space', 'enter'

-- ---------------------------------------------------------------- lifecycle
local function start()
  pcall(collectgarbage, 'generational')   -- loading allocates many short strings: avoids 100+ ms GC pauses
  print(string.format('[paimon] start: %s, %s', tostring(_VERSION), MODE))   -- which Lua the client runs (kernel tuning)
  root = script.object; root.showCursor = true
  for _, group in ipairs(LAYERS) do
    for _, name in ipairs(group[2]) do
      local c = root:FindChild(group[1] == '' and name or (group[1] .. '/' .. name))
      if c then c:SetAsFirstSibling() end
    end
  end
  for _, group in ipairs(LAYERS) do if group[1] == '' then ALL = group[2] end end
  local w, h = game.GetUICanvasSize()
  local s = math.min(w / 1280, h / 720); root.localScaleX = s; root.localScaleY = s

  local function keyPress(k)
    if k == 'back' then ime:backspace() elseif k == 'space' then ime:space()
    elseif k == 'enter' then send() else ime:key(k) end
    render()
  end
  for ch in ('abcdefghijklmnopqrstuvwxyz'):gmatch('.') do click('Key' .. ch:upper(), function() keyPress(ch) end) end
  for i, k in pairs(KEYMAP) do
    pcall(function() root:AddKeyEventListener(Enum.KeyEventType['KeyboardCraftspersonKey' .. i .. 'Down'], guard('key ' .. k, function() keyPress(k) end)) end)
  end
  for k, p in pairs(KEY_PUNCT) do click(k, function() ime:punct(ime.en and p[2] or p[1]); render() end) end
  click('Lang', function() ime:toggle(); render() end)
  click('KeyBack', function() keyPress('back') end)
  click('KeySpace', function() keyPress('space') end)
  click('KeyClear', function() ime:clear(); render() end)
  for k = 1, 7 do click('Cand' .. k, function() ime:pick(k); render() end) end
  click('CandPrev', function() ime:turn(-1); render() end)
  click('CandNext', function() ime:turn(1); render() end)
  click('Send', function() send(); render() end)
  click('Plus', function() menuOpen = not menuOpen; render() end)
  for k, q in ipairs(SUGGEST) do click('Menu' .. k, function() ask(q); render() end) end
  for k = 1, SLOTS do
    click('Back' .. k, function()
      local last = #msgs - scroll
      local i = math.max(1, last - SLOTS + 1) + k - 1
      if msgs[i] and msgs[i].who == 'u' then rewind(i); render() end
    end)
  end
  click('ScrollUp', function() if #msgs - scroll > SLOTS then scroll = scroll + 1 end; render() end)
  click('ScrollDown', function() if scroll > 0 then scroll = scroll - 1 end; render() end)
  click('NewChat', function() if busy() then return end; msgs, ctx, scroll, compressed, stats = {}, {}, 0, 0, ''; render() end)
  click('Theme', function() theme = theme == 'light' and 'dark' or 'light'; applyTheme(); render() end)
  click('Settle', function()
    if clock < SETTLE_AFTER then return end
    signal('PaimonSettle'); stats = '正在结算…'; render()
  end)

  loader = MODE == 'full' and __m.MM.loader(__m.W) or Mock.loader()
  __m.W = nil
  applyTheme()
  script:EnableUpdate(true)
  render(); settleTick()
end

local function update(dt)
  local before = clock
  clock = clock + dt
  if dt > diag.maxDt then diag.maxDt = dt end
  if (loader or gen) and clock - diag.last >= 10 then        -- periodic diagnostics while working
    diag.last = clock
    print(string.format('[paimon] %s heap %.1f MB, slowest frame %.2f s, idle frame %.3f s, budget %.0f', loader and string.format('loading %d%%', math.floor(loader:progress() * 100))
      or (turn and turn.tFirst and 'answering' or 'thinking'), heapMB(), diag.maxDt, idleDt, budget))
    diag.maxDt = 0
  end
  if turn and clock - turn.t0 > TURN_TIMEOUT then print('[paimon] answer timed out'); recover('timeout'); return end
  settleTick()
  if turn and not turn.tFirst and math.floor(clock - turn.t0) ~= math.floor(before - turn.t0) then render() end   -- waitText seconds
  local drafting = false
  if MODE == 'full' and model and not gen and not loader and clock >= WARMUP then
    if ime.text ~= '' and ime.text ~= draftText then draftText = ime.text; feeder = startDraft(ime.text) end
    drafting = feeder ~= nil
  end
  if drafting then                                    -- typing: keep the keyboard responsive
    local target = math.max(1 / 30, idleDt * 1.5)
    if dt > target then budget = math.max(BUDGET_MIN, math.floor(budget * 0.8))
    elseif dt < target * 0.8 then budget = math.min(BUDGET_MAX, math.floor(budget * 1.12)) end
    if feeder:step(budget) then feeder = nil end
    return
  end
  if (not loader and not gen) or clock < WARMUP then
    idleN = idleN + 1
    idleDt = idleN == 1 and dt or idleDt * 0.9 + math.min(dt, 0.5) * 0.1
    return
  end
  -- before the first character (prefill) the screen is still: give the model more of the frame
  local prefill = turn and not turn.tFirst
  local target = prefill and math.max(1 / 12, idleDt * 4) or math.max(1 / 20, idleDt * 2.5)
  if dt > target then budget = math.max(BUDGET_MIN, math.floor(budget * 0.8))
  elseif dt < target * 0.8 then budget = math.min(BUDGET_MAX, math.floor(budget * 1.12)) end
  if loader then
    if loader:step(budget) then
      model = loader.model; loader = nil
      print(string.format('[paimon] model ready after %.1f s, heap %.1f MB', clock - diag.t0, heapMB()))
      if MODE == 'full' then tk = __m.BPE.new(__m.TOKT); __m.TOKT = nil end
      if pending then gen = nil; pending = nil; startModel(turn.q) end
    end
    render()
    return
  end
  local piece, done = gen:step(budget)
  if piece ~= '' then
    if not turn.tFirst then turn.tFirst = clock end
    turn.raw[#turn.raw + 1] = piece
    turn.chars = turn.chars + #chars(piece)
    msgs[#msgs].text = msgs[#msgs].text .. piece
  end
  if done then finishTurn() end
  if piece ~= '' or done then render() end
end

local safeStart, safeUpdate = guard('OnStart', start), guard('OnUpdate', update)
function OnStart() diag.t0 = clock; safeStart() end
function OnUpdate(dt) safeUpdate(dt) end

-- test hook (game/filter_test.mjs): the filter is checked against the Python reference
pcall(function() PAIMON_DEBUG = { classify = classify, normText = normText, lookup = lookup } end)   -- globals may be read-only in the sandbox
