-- Pinyin input method for an on-screen keyboard (Lua 5.3, no external modules).
--   local ime = IME.new(<ime_dict table>)    -- flat {pinyin, word, freq, ...}, most frequent first
--   ime:key('n') ime:key('i') ...            -- letters go into the pinyin buffer
--   ime:candidates()                          -- {{word=, take=<letters consumed>}, ...}
--   ime:pick(i)                               -- commit candidate i (rest of the buffer stays)
--   ime:backspace()  ime:punct('？')  ime:clear()
--   ime.text (committed), ime.buf (pinyin being typed)
-- Matching, in order: whole buffer = a word's pinyin; buffer = start of a word's pinyin
-- (last syllable unfinished); the first k whole syllables = a word; the first syllable
-- = a character; buffer = a word's initials ("nh" -> 你好).
local IME = {}
IME.__index = IME

local MAX_TEXT, MAX_BUF = 40, 24

-- UTF-8 helpers (no dependency on the utf8 library)
local function ulen(s)
  local n = 0
  for i = 1, #s do local b = s:byte(i); if b < 0x80 or b >= 0xC0 then n = n + 1 end end
  return n
end
local function drop_last(s)
  local i = #s
  while i > 0 do local b = s:byte(i); if b < 0x80 or b >= 0xC0 then break end; i = i - 1 end
  return s:sub(1, i - 1)
end
IME.ulen = ulen

function IME.new(dict)
  local self = setmetatable({ text = '', buf = '', page = 1, per_page = 7 }, IME)
  self.entries, self.syll, self.exact = {}, {}, {}
  for i = 1, #dict, 3 do
    local py, w, f = dict[i], dict[i + 1], dict[i + 2]
    local n = ulen(w)
    local e = { py = py, w = w, f = f, n = n }
    self.entries[#self.entries + 1] = e
    if n == 1 and (self.syll[py] or 0) < f then self.syll[py] = f end   -- syllable -> its most frequent character
    local l = self.exact[py]
    if not l then l = {}; self.exact[py] = l end
    l[#l + 1] = e
  end
  -- initials of multi-character words, from their own syllables
  for _, e in ipairs(self.entries) do
    if e.n > 1 then e.ini = self:initials(e.py); e.ini1 = e.ini and e.ini:gsub("([zcs])h", "%1") end   -- "shm" and "sm" both find 什么
  end
  self.cache = {}
  return self
end

-- Split into known syllables by dynamic programming: cover as much of the string as
-- possible, then maximise sum(log f - 13) over the syllables (f = frequency of the
-- syllable's commonest character), which prefers common and fewer syllables:
-- "quna" -> qu na (not qun a), "xian" -> xian (not xi an).
-- Returns the syllables and the unsplittable rest.
function IME:split(s)
  local hit = self.splits and self.splits[s]
  if hit then return hit[1], hit[2] end
  local n = #s
  local best = { [0] = { 0, 0 } }        -- best[i] = {covered score key, score}, ending at i
  local from = {}
  for i = 0, n - 1 do
    if best[i] then
      for len = 1, math.min(6, n - i) do
        local piece = s:sub(i + 1, i + len)
        local f = self.syll[piece]
        if f then
          local sc = best[i][2] + math.log(f) - 13
          local j = i + len
          if not best[j] or sc > best[j][2] then best[j] = { j, sc }; from[j] = i end
        end
      end
    end
  end
  local stop = n
  while stop > 0 and not best[stop] do stop = stop - 1 end
  local out, j = {}, stop
  while j > 0 do local i = from[j]; table.insert(out, 1, s:sub(i + 1, j)); j = i end
  self.splits = self.splits or {}
  self.splits[s] = { out, s:sub(stop + 1) }
  return out, s:sub(stop + 1)
end

function IME:initials(py)
  local sy, rest = self:split(py)
  if rest ~= '' then return nil end
  local t = {}
  for i, s in ipairs(sy) do
    local c = s:sub(1, 1)
    if (c == 'z' or c == 'c' or c == 's') and s:sub(2, 2) == 'h' then c = c .. 'h' end
    t[i] = c
  end
  return table.concat(t)
end

function IME:candidates()
  local buf = self.buf
  if buf == '' then return {} end
  local hit = self.cache[buf]
  if hit then return hit end
  local out, seen = {}, {}
  local function add(e, take)
    if not seen[e.w] then seen[e.w] = true; out[#out + 1] = { word = e.w, take = take, f = e.f } end
  end
  local function addList(list, take, limit)
    if not list then return end
    for i = 1, math.min(#list, limit or #list) do add(list[i], take) end
  end
  local function initials()
    if #buf < 2 then return end
    local n = 0
    for _, e in ipairs(self.entries) do
      if e.ini == buf or e.ini1 == buf then add(e, #buf); n = n + 1; if n >= 10 then break end end
    end
  end
  if not buf:find('[aeiouv]') then initials() end      -- "nh", "zh": abbreviations come first
  -- 1) the whole buffer is a word / character
  addList(self.exact[buf], #buf)
  -- 2) the buffer starts a longer word (unfinished last syllable), most frequent first
  local pre = {}
  for _, e in ipairs(self.entries) do
    if e.n > 1 and #e.py > #buf and e.py:sub(1, #buf) == buf then
      pre[#pre + 1] = e
      if #pre >= 12 then break end
    end
  end
  for _, e in ipairs(pre) do add(e, #buf) end
  -- 3) the first k whole syllables form a word, longest first
  local sy = self:split(buf)
  for k = #sy - 1, 2, -1 do
    local key = table.concat(sy, '', 1, k)
    addList(self.exact[key], #key, 6)
  end
  -- 4) characters for the first syllable (or, if it is unfinished, any syllable it starts)
  if sy[1] then
    for _, e in ipairs(self.exact[sy[1]] or {}) do if e.n == 1 then add(e, #sy[1]) end end
  else
    local n = 0
    for _, e in ipairs(self.entries) do
      if e.n == 1 and e.py:sub(1, #buf) == buf then add(e, #buf); n = n + 1; if n >= 30 then break end end
    end
  end
  -- 5) initials ("nh" -> 你好)
  initials()
  self.cache[buf] = out
  return out
end

function IME:page_items()
  local c = self:candidates()
  local pages = math.max(1, math.ceil(#c / self.per_page))
  if self.page > pages then self.page = pages end
  local items = {}
  for i = (self.page - 1) * self.per_page + 1, math.min(#c, self.page * self.per_page) do items[#items + 1] = c[i] end
  return items, self.page, pages
end

function IME:turn(d)
  local _, _, pages = self:page_items()
  self.page = math.min(pages, math.max(1, self.page + d))
end

local function append(self, s)
  if ulen(self.text .. s) <= MAX_TEXT then self.text = self.text .. s end
end

-- English mode (self.en): letters and spaces go straight into the text
function IME:key(ch)
  if self.en then append(self, ch); return end
  if #self.buf < MAX_BUF then self.buf = self.buf .. ch; self.page = 1 end
end

-- commit candidate i of the current page (default: the first)
function IME:pick(i)
  local items = self:page_items()
  local c = items[i or 1]
  if not c then return false end
  append(self, c.word)
  self.buf = self.buf:sub(c.take + 1)
  self.page = 1
  return true
end

function IME:punct(p)
  if self.buf ~= '' then self:pick(1) end
  append(self, p)
end

function IME:backspace()
  if self.buf ~= '' then
    self.buf = self.buf:sub(1, -2); self.page = 1
  elseif self.text ~= '' then
    self.text = drop_last(self.text)
  end
end

function IME:space()
  if self.en then append(self, ' ') elseif self.buf ~= '' then self:pick(1) end
end

-- 中/英: pending pinyin goes in as the typed letters
function IME:toggle()
  if self.buf ~= '' then append(self, self.buf); self.buf, self.page = '', 1 end
  self.en = not self.en
end

function IME:clear() self.text, self.buf, self.page = '', '', 1 end

return IME
