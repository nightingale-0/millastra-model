-- Byte-level BPE tokenizer (GPT-2 pre-tokenizer rules) for MiniMind2, Lua 5.3.
-- Tables come from build/mm_tok.lua (export/export_lua.py).
--   local BPE = <this chunk>; local tk = BPE.new(<mm_tok table>)
--   tk:encode(text) -> ids    tk:decode(ids) -> string    tk:piece(id) -> raw bytes
local BPE = {}
BPE.__index = BPE

local byte, sub, floor = string.byte, string.sub, math.floor
local OTHER, LETTER, NUMBER, SPACE = 0, 1, 2, 3

function BPE.new(t)
  local self = setmetatable({}, BPE)
  self.pieces, self.specials = t.pieces, t.specials
  self.cls_start, self.cls = t.cls_start, t.cls
  self.byte_id = {}
  for id = 1, #t.pieces do
    local p = t.pieces[id]
    if #p == 1 and not self.byte_id[byte(p)] then self.byte_id[byte(p)] = id - 1 end
  end
  -- pair key a*8192+b -> rank / merged id
  self.rank, self.merged = {}, {}
  local m = t.merges
  for i = 1, #m, 3 do
    local key = m[i] * 8192 + m[i + 1]
    if not self.rank[key] then self.rank[key] = (i + 2) // 3; self.merged[key] = m[i + 2] end
  end
  self.cache = {}
  self.special_list = {}
  for s, id in pairs(t.specials) do self.special_list[#self.special_list + 1] = { s, id } end
  return self
end

function BPE:class(cp)
  if cp < 128 then
    if cp == 32 or (cp >= 9 and cp <= 13) then return SPACE end
    if (cp >= 65 and cp <= 90) or (cp >= 97 and cp <= 122) then return LETTER end
    if cp >= 48 and cp <= 57 then return NUMBER end
    return OTHER
  end
  local st = self.cls_start
  local lo, hi = 1, #st
  while lo < hi do
    local mid = (lo + hi + 1) // 2
    if st[mid] <= cp then lo = mid else hi = mid - 1 end
  end
  return self.cls[lo]
end

-- UTF-8 decode -> arrays of (byte start, codepoint, class); invalid bytes count as OTHER.
local function decode_utf8(self, s)
  local pos, cps, cls = {}, {}, {}
  local i, n, k = 1, #s, 0
  while i <= n do
    local c = byte(s, i)
    local len, cp = 1, c
    if c >= 0xF0 and i + 3 <= n then len = 4; cp = ((c & 7) << 18) | ((byte(s, i + 1) & 63) << 12) | ((byte(s, i + 2) & 63) << 6) | (byte(s, i + 3) & 63)
    elseif c >= 0xE0 and i + 2 <= n then len = 3; cp = ((c & 15) << 12) | ((byte(s, i + 1) & 63) << 6) | (byte(s, i + 2) & 63)
    elseif c >= 0xC0 and i + 1 <= n then len = 2; cp = ((c & 31) << 6) | (byte(s, i + 1) & 63)
    elseif c >= 0x80 then cp = -1 end
    k = k + 1
    pos[k], cps[k] = i, cp
    cls[k] = cp < 0 and OTHER or self:class(cp)
    i = i + len
  end
  pos[k + 1] = n + 1
  return pos, cps, cls, k
end

local CONTRACT = { "s", "t", "re", "ve", "m", "ll", "d" }

-- GPT-2 regex: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
function BPE:pretokenize(s, out)
  local pos, cps, cls, n = decode_utf8(self, s)
  local i = 1
  while i <= n do
    local j = nil
    if cps[i] == 39 then                              -- '
      for _, c in ipairs(CONTRACT) do
        if sub(s, pos[i] + 1, pos[i] + #c) == c then j = i + #c + 1; break end
      end
    end
    if not j then
      local c, start = cls[i], i
      if cps[i] == 32 and i < n and cls[i + 1] ~= SPACE then start = i + 1; c = cls[i + 1] end
      if c ~= SPACE then
        j = start + 1
        while j <= n and cls[j] == c do j = j + 1 end
      else
        j = i + 1
        while j <= n and cls[j] == SPACE do j = j + 1 end
        if j <= n and j - i > 1 then j = j - 1 end    -- \s+(?!\S): leave the last space to the next token
      end
    end
    out[#out + 1] = sub(s, pos[i], pos[j] - 1)
    i = j
  end
  return out
end

function BPE:bpe(word, ids)
  local hit = self.cache[word]
  if not hit then
    local sym, n = {}, #word
    for i = 1, n do sym[i] = self.byte_id[byte(word, i)] end
    local rank, merged = self.rank, self.merged
    while n > 1 do
      local best, bi = nil, 0
      for i = 1, n - 1 do
        local r = rank[sym[i] * 8192 + sym[i + 1]]
        if r and (not best or r < best) then best, bi = r, i end
      end
      if not best then break end
      sym[bi] = merged[sym[bi] * 8192 + sym[bi + 1]]
      table.remove(sym, bi + 1)
      n = n - 1
    end
    hit = sym
    self.cache[word] = hit
  end
  for i = 1, #hit do ids[#ids + 1] = hit[i] end
  return ids
end

-- encode plain text; special-token strings (<|im_start|> etc.) are recognised.
function BPE:encode(text, ids)
  ids = ids or {}
  local i = 1
  while i <= #text do
    local best, bid, blen = nil, nil, 0
    for _, sp in ipairs(self.special_list) do
      local a = string.find(text, sp[1], i, true)
      if a and (not best or a < best) then best, bid, blen = a, sp[2], #sp[1] end
    end
    local seg = sub(text, i, (best or #text + 1) - 1)
    if #seg > 0 then
      for _, w in ipairs(self:pretokenize(seg, {})) do self:bpe(w, ids) end
    end
    if not best then break end
    ids[#ids + 1] = bid
    i = best + blen
  end
  return ids
end

function BPE:piece(id) return self.pieces[id + 1] end

function BPE:decode(ids, skip_special)
  local out = {}
  for i = 1, #ids do
    local id = ids[i]
    if not (skip_special and id < 3) then out[#out + 1] = self.pieces[id + 1] end
  end
  return table.concat(out)
end

return BPE
