-- MiniMind2 (Llama) inference for the 千星 client: Lua 5.3, no JIT / coroutines / IO.
-- Every heavy step is sliced so a frame can spend a fixed work budget:
--
--   local MM = <this chunk>
--   local loader = MM.loader(<mm_weights table>)      -- decodes weights, sliced
--   repeat until loader:step(200000)                  -- per frame; returns true when done
--   local model = loader.model
--   local tk = BPE.new(<mm_tok table>)
--   local gen = model:chat(tk, "你好", { persona = "guide", temperature = 0.7, top_k = 20 })
--   -- per frame:  local text, done = gen:step(200000)   (text = newly finished UTF-8 chars)
--   local gen2 = model:say(tk, "然后呢？")              -- next turn of the same conversation
--
-- Budget unit ~ one weight-plane table lookup (a 4-bit weight costs 0.5).
--
-- Prefix reuse works per message turn. The model remembers which turns its K/V
-- holds (content + end position); a request (a message list) reuses the longest
-- run of leading turns it shares with that state, or with the exported presets
-- (export/presets.json: persona system turns + preset first questions, the
-- latter with first-step logits, so answering them costs no prefill). Rewinding
-- to an earlier turn is free: the position just moves back.
local MM = {}

local byte, char, gsub, sub, find, concat = string.byte, string.char, string.gsub, string.sub, string.find, table.concat
local exp, sqrt, cos, sin, floor, random, huge = math.exp, math.sqrt, math.cos, math.sin, math.floor, math.random, math.huge

-- ---------------------------------------------------------------- decoding
local function dec4(s)
  local a, b, c, d = byte(s, 1, 4)
  local n = ((a - 63) << 18) | ((b - 63) << 12) | ((c - 63) << 6) | (d - 63)
  return char(n >> 16, (n >> 8) & 255, n & 255)
end

-- base85 (export/export_lua.py enc85): 5 chars from printable ASCII without '=' -> 4 bytes.
-- Computed in floats (exact below 2^53), so 32-bit-integer Lua builds decode it too.
local B85 = {}
do
  local k = 0
  for c = 33, 126 do local ch = string.char(c); if c ~= 61 and not ("aeiouUIO"):find(ch, 1, true) and k < 85 then B85[c] = k + 0.0; k = k + 1 end end   -- same alphabet as export_lua.py B85
end
local function dec5(s)
  local a, b, c, d, e = byte(s, 1, 5)
  local n = (((B85[a] * 85 + B85[b]) * 85 + B85[c]) * 85 + B85[d]) * 85 + B85[e]
  local b1 = n // 16777216; n = n - b1 * 16777216
  local b2 = n // 65536; n = n - b2 * 65536
  local b3 = n // 256
  return char(math.floor(b1), math.floor(b2), math.floor(b3), math.floor(n - b3 * 256))
end
MM.dec5 = dec5

local F16 = nil   -- fp16 bit pattern -> float
local function f16_table()
  if F16 then return F16 end
  F16 = {}
  for h = 0, 65535 do
    local e, m = (h >> 10) & 31, h & 1023
    local v
    if e == 0 then v = m * 2.0 ^ -24
    elseif e == 31 then v = m == 0 and huge or 0 / 0
    else v = (1024 + m) * 2.0 ^ (e - 25) end
    F16[h] = (h & 32768) ~= 0 and -v or v
  end
  return F16
end

local function f16_floats(s, out)
  local F = f16_table()
  out = out or {}
  for i = 1, #s // 2 do
    local lo, hi = byte(s, 2 * i - 1, 2 * i)
    out[i] = F[lo | (hi << 8)]
  end
  return out
end

-- ---------------------------------------------------------------- kernels
-- LUT of subset sums: L[g*256 + m + 1] = sum of x[g*8+k+1] over bits k set in m.
local function build_lut(x, n, L)
  for g = 0, n // 8 - 1 do
    local o, xo = g * 256 + 1, g * 8
    L[o] = 0.0
    local x0, x1, x2, x3, x4, x5, x6, x7 = x[xo + 1], x[xo + 2], x[xo + 3], x[xo + 4], x[xo + 5], x[xo + 6], x[xo + 7], x[xo + 8]
    L[o + 1] = x0
    L[o + 2] = x1; L[o + 3] = x0 + x1
    for m = 0, 3 do L[o + 4 + m] = L[o + m] + x2 end
    for m = 0, 7 do L[o + 8 + m] = L[o + m] + x3 end
    for m = 0, 15 do L[o + 16 + m] = L[o + m] + x4 end
    for m = 0, 31 do L[o + 32 + m] = L[o + m] + x5 end
    for m = 0, 63 do L[o + 64 + m] = L[o + m] + x6 end
    for m = 0, 127 do L[o + 128 + m] = L[o + m] + x7 end
  end
end

-- Weights are lists of parts (W[1], W[2], ...), each holding W.per whole rows: the client
-- sandbox rejects big table.concat results, so no matrix is ever one string.
-- y[yo + r + 1] = W[r] . x  for rows r0 .. r1-1 (0-based), x given by its LUT.
-- S = the matrix's 8-bit scale codes: row-aligned byte-string parts (S.per rows each) and
-- S.tab[code] = the block scale (strings: 1 byte per block instead of a 16-byte table slot)
local function matvec(Wp, S, L, cols, r0, r1, y, yo)
  local nb, rowbytes, per = cols // 32, cols // 2, Wp.per
  for r = r0, r1 - 1 do
    local ci = r // per
    local W = Wp[ci + 1]
    local pos, s, o = (r - ci * per) * rowbytes + 1, r * nb, 1
    local acc = 0.0
    for b = 1, nb do
      local a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = byte(W, pos, pos + 15)
      local o2, o3, o4 = o + 256, o + 512, o + 768
      acc = acc + ((L[o + a0] + L[o2 + b0] + L[o3 + c0] + L[o4 + d0])
                 + 2 * (L[o + a1] + L[o2 + b1] + L[o3 + c1] + L[o4 + d1])
                 + 4 * (L[o + a2] + L[o2 + b2] + L[o3 + c2] + L[o4 + d2])
                 - 8 * (L[o + a3] + L[o2 + b3] + L[o3 + c3] + L[o4 + d3])) * S[s + b]
      pos = pos + 16; o = o + 1024
    end
    y[yo + r + 1] = acc
  end
end

-- Batched version for prefill: B inputs (LUTs Ls[1..B]) share every weight read.
-- ys[b][yo + r + 1] = W[r] . x_b
local function matvec_batch(Wp, S, Ls, B, cols, r0, r1, ys, yo)
  local nb, rowbytes, per = cols // 32, cols // 2, Wp.per
  local acc = {}
  for r = r0, r1 - 1 do
    local ci = r // per
    local W = Wp[ci + 1]
    local pos, s, o = (r - ci * per) * rowbytes + 1, r * nb, 1
    for b = 1, B do acc[b] = 0.0 end
    for blk = 1, nb do
      local a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = byte(W, pos, pos + 15)
      local sc = S[s + blk]
      local oa0, oa1, oa2, oa3 = o + a0, o + a1, o + a2, o + a3
      local ob0, ob1, ob2, ob3 = o + 256 + b0, o + 256 + b1, o + 256 + b2, o + 256 + b3
      local oc0, oc1, oc2, oc3 = o + 512 + c0, o + 512 + c1, o + 512 + c2, o + 512 + c3
      local od0, od1, od2, od3 = o + 768 + d0, o + 768 + d1, o + 768 + d2, o + 768 + d3
      for b = 1, B do
        local L = Ls[b]
        acc[b] = acc[b] + ((L[oa0] + L[ob0] + L[oc0] + L[od0])
                         + 2 * (L[oa1] + L[ob1] + L[oc1] + L[od1])
                         + 4 * (L[oa2] + L[ob2] + L[oc2] + L[od2])
                         - 8 * (L[oa3] + L[ob3] + L[oc3] + L[od3])) * sc
      end
      pos = pos + 16; o = o + 1024
    end
    for b = 1, B do ys[b][yo + r + 1] = acc[b] end
  end
end

-- dequantized row r (0-based) of W -> out[1..cols]
-- prefill kernels for B = 2..8 tokens (generated, see matvec_batch): LUTs, accumulators and outputs in
-- locals instead of tables; the same sums in the same order per token as matvec_batch
local MVB = {}
MVB[2] = function(Wp, S, Ls, cols, r0, r1, ys, yo)
  local nb, rowbytes, per = cols // 32, cols // 2, Wp.per
  local L1, L2 = Ls[1], Ls[2]
  local Y1, Y2 = ys[1], ys[2]
  for r = r0, r1 - 1 do
    local ci = r // per
    local W = Wp[ci + 1]
    local pos, s, o = (r - ci * per) * rowbytes + 1, r * nb, 1
    local x1, x2 = 0.0, 0.0
    for blk = 1, nb do
      local a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = byte(W, pos, pos + 15)
      local sc = S[s + blk]
      local oa0, oa1, oa2, oa3 = o + a0, o + a1, o + a2, o + a3
      local o2, o3, o4 = o + 256, o + 512, o + 768
      local ob0, ob1, ob2, ob3 = o2 + b0, o2 + b1, o2 + b2, o2 + b3
      local oc0, oc1, oc2, oc3 = o3 + c0, o3 + c1, o3 + c2, o3 + c3
      local od0, od1, od2, od3 = o4 + d0, o4 + d1, o4 + d2, o4 + d3
        x1 = x1 + ((L1[oa0] + L1[ob0] + L1[oc0] + L1[od0])
                 + 2 * (L1[oa1] + L1[ob1] + L1[oc1] + L1[od1])
                 + 4 * (L1[oa2] + L1[ob2] + L1[oc2] + L1[od2])
                 - 8 * (L1[oa3] + L1[ob3] + L1[oc3] + L1[od3])) * sc
        x2 = x2 + ((L2[oa0] + L2[ob0] + L2[oc0] + L2[od0])
                 + 2 * (L2[oa1] + L2[ob1] + L2[oc1] + L2[od1])
                 + 4 * (L2[oa2] + L2[ob2] + L2[oc2] + L2[od2])
                 - 8 * (L2[oa3] + L2[ob3] + L2[oc3] + L2[od3])) * sc
      pos = pos + 16; o = o + 1024
    end
      Y1[yo + r + 1] = x1
      Y2[yo + r + 1] = x2
  end
end
MVB[3] = function(Wp, S, Ls, cols, r0, r1, ys, yo)
  local nb, rowbytes, per = cols // 32, cols // 2, Wp.per
  local L1, L2, L3 = Ls[1], Ls[2], Ls[3]
  local Y1, Y2, Y3 = ys[1], ys[2], ys[3]
  for r = r0, r1 - 1 do
    local ci = r // per
    local W = Wp[ci + 1]
    local pos, s, o = (r - ci * per) * rowbytes + 1, r * nb, 1
    local x1, x2, x3 = 0.0, 0.0, 0.0
    for blk = 1, nb do
      local a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = byte(W, pos, pos + 15)
      local sc = S[s + blk]
      local oa0, oa1, oa2, oa3 = o + a0, o + a1, o + a2, o + a3
      local o2, o3, o4 = o + 256, o + 512, o + 768
      local ob0, ob1, ob2, ob3 = o2 + b0, o2 + b1, o2 + b2, o2 + b3
      local oc0, oc1, oc2, oc3 = o3 + c0, o3 + c1, o3 + c2, o3 + c3
      local od0, od1, od2, od3 = o4 + d0, o4 + d1, o4 + d2, o4 + d3
        x1 = x1 + ((L1[oa0] + L1[ob0] + L1[oc0] + L1[od0])
                 + 2 * (L1[oa1] + L1[ob1] + L1[oc1] + L1[od1])
                 + 4 * (L1[oa2] + L1[ob2] + L1[oc2] + L1[od2])
                 - 8 * (L1[oa3] + L1[ob3] + L1[oc3] + L1[od3])) * sc
        x2 = x2 + ((L2[oa0] + L2[ob0] + L2[oc0] + L2[od0])
                 + 2 * (L2[oa1] + L2[ob1] + L2[oc1] + L2[od1])
                 + 4 * (L2[oa2] + L2[ob2] + L2[oc2] + L2[od2])
                 - 8 * (L2[oa3] + L2[ob3] + L2[oc3] + L2[od3])) * sc
        x3 = x3 + ((L3[oa0] + L3[ob0] + L3[oc0] + L3[od0])
                 + 2 * (L3[oa1] + L3[ob1] + L3[oc1] + L3[od1])
                 + 4 * (L3[oa2] + L3[ob2] + L3[oc2] + L3[od2])
                 - 8 * (L3[oa3] + L3[ob3] + L3[oc3] + L3[od3])) * sc
      pos = pos + 16; o = o + 1024
    end
      Y1[yo + r + 1] = x1
      Y2[yo + r + 1] = x2
      Y3[yo + r + 1] = x3
  end
end
MVB[4] = function(Wp, S, Ls, cols, r0, r1, ys, yo)
  local nb, rowbytes, per = cols // 32, cols // 2, Wp.per
  local L1, L2, L3, L4 = Ls[1], Ls[2], Ls[3], Ls[4]
  local Y1, Y2, Y3, Y4 = ys[1], ys[2], ys[3], ys[4]
  for r = r0, r1 - 1 do
    local ci = r // per
    local W = Wp[ci + 1]
    local pos, s, o = (r - ci * per) * rowbytes + 1, r * nb, 1
    local x1, x2, x3, x4 = 0.0, 0.0, 0.0, 0.0
    for blk = 1, nb do
      local a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = byte(W, pos, pos + 15)
      local sc = S[s + blk]
      local oa0, oa1, oa2, oa3 = o + a0, o + a1, o + a2, o + a3
      local o2, o3, o4 = o + 256, o + 512, o + 768
      local ob0, ob1, ob2, ob3 = o2 + b0, o2 + b1, o2 + b2, o2 + b3
      local oc0, oc1, oc2, oc3 = o3 + c0, o3 + c1, o3 + c2, o3 + c3
      local od0, od1, od2, od3 = o4 + d0, o4 + d1, o4 + d2, o4 + d3
        x1 = x1 + ((L1[oa0] + L1[ob0] + L1[oc0] + L1[od0])
                 + 2 * (L1[oa1] + L1[ob1] + L1[oc1] + L1[od1])
                 + 4 * (L1[oa2] + L1[ob2] + L1[oc2] + L1[od2])
                 - 8 * (L1[oa3] + L1[ob3] + L1[oc3] + L1[od3])) * sc
        x2 = x2 + ((L2[oa0] + L2[ob0] + L2[oc0] + L2[od0])
                 + 2 * (L2[oa1] + L2[ob1] + L2[oc1] + L2[od1])
                 + 4 * (L2[oa2] + L2[ob2] + L2[oc2] + L2[od2])
                 - 8 * (L2[oa3] + L2[ob3] + L2[oc3] + L2[od3])) * sc
        x3 = x3 + ((L3[oa0] + L3[ob0] + L3[oc0] + L3[od0])
                 + 2 * (L3[oa1] + L3[ob1] + L3[oc1] + L3[od1])
                 + 4 * (L3[oa2] + L3[ob2] + L3[oc2] + L3[od2])
                 - 8 * (L3[oa3] + L3[ob3] + L3[oc3] + L3[od3])) * sc
        x4 = x4 + ((L4[oa0] + L4[ob0] + L4[oc0] + L4[od0])
                 + 2 * (L4[oa1] + L4[ob1] + L4[oc1] + L4[od1])
                 + 4 * (L4[oa2] + L4[ob2] + L4[oc2] + L4[od2])
                 - 8 * (L4[oa3] + L4[ob3] + L4[oc3] + L4[od3])) * sc
      pos = pos + 16; o = o + 1024
    end
      Y1[yo + r + 1] = x1
      Y2[yo + r + 1] = x2
      Y3[yo + r + 1] = x3
      Y4[yo + r + 1] = x4
  end
end
MVB[5] = function(Wp, S, Ls, cols, r0, r1, ys, yo)
  local nb, rowbytes, per = cols // 32, cols // 2, Wp.per
  local L1, L2, L3, L4, L5 = Ls[1], Ls[2], Ls[3], Ls[4], Ls[5]
  local Y1, Y2, Y3, Y4, Y5 = ys[1], ys[2], ys[3], ys[4], ys[5]
  for r = r0, r1 - 1 do
    local ci = r // per
    local W = Wp[ci + 1]
    local pos, s, o = (r - ci * per) * rowbytes + 1, r * nb, 1
    local x1, x2, x3, x4, x5 = 0.0, 0.0, 0.0, 0.0, 0.0
    for blk = 1, nb do
      local a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = byte(W, pos, pos + 15)
      local sc = S[s + blk]
      local oa0, oa1, oa2, oa3 = o + a0, o + a1, o + a2, o + a3
      local o2, o3, o4 = o + 256, o + 512, o + 768
      local ob0, ob1, ob2, ob3 = o2 + b0, o2 + b1, o2 + b2, o2 + b3
      local oc0, oc1, oc2, oc3 = o3 + c0, o3 + c1, o3 + c2, o3 + c3
      local od0, od1, od2, od3 = o4 + d0, o4 + d1, o4 + d2, o4 + d3
        x1 = x1 + ((L1[oa0] + L1[ob0] + L1[oc0] + L1[od0])
                 + 2 * (L1[oa1] + L1[ob1] + L1[oc1] + L1[od1])
                 + 4 * (L1[oa2] + L1[ob2] + L1[oc2] + L1[od2])
                 - 8 * (L1[oa3] + L1[ob3] + L1[oc3] + L1[od3])) * sc
        x2 = x2 + ((L2[oa0] + L2[ob0] + L2[oc0] + L2[od0])
                 + 2 * (L2[oa1] + L2[ob1] + L2[oc1] + L2[od1])
                 + 4 * (L2[oa2] + L2[ob2] + L2[oc2] + L2[od2])
                 - 8 * (L2[oa3] + L2[ob3] + L2[oc3] + L2[od3])) * sc
        x3 = x3 + ((L3[oa0] + L3[ob0] + L3[oc0] + L3[od0])
                 + 2 * (L3[oa1] + L3[ob1] + L3[oc1] + L3[od1])
                 + 4 * (L3[oa2] + L3[ob2] + L3[oc2] + L3[od2])
                 - 8 * (L3[oa3] + L3[ob3] + L3[oc3] + L3[od3])) * sc
        x4 = x4 + ((L4[oa0] + L4[ob0] + L4[oc0] + L4[od0])
                 + 2 * (L4[oa1] + L4[ob1] + L4[oc1] + L4[od1])
                 + 4 * (L4[oa2] + L4[ob2] + L4[oc2] + L4[od2])
                 - 8 * (L4[oa3] + L4[ob3] + L4[oc3] + L4[od3])) * sc
        x5 = x5 + ((L5[oa0] + L5[ob0] + L5[oc0] + L5[od0])
                 + 2 * (L5[oa1] + L5[ob1] + L5[oc1] + L5[od1])
                 + 4 * (L5[oa2] + L5[ob2] + L5[oc2] + L5[od2])
                 - 8 * (L5[oa3] + L5[ob3] + L5[oc3] + L5[od3])) * sc
      pos = pos + 16; o = o + 1024
    end
      Y1[yo + r + 1] = x1
      Y2[yo + r + 1] = x2
      Y3[yo + r + 1] = x3
      Y4[yo + r + 1] = x4
      Y5[yo + r + 1] = x5
  end
end
MVB[6] = function(Wp, S, Ls, cols, r0, r1, ys, yo)
  local nb, rowbytes, per = cols // 32, cols // 2, Wp.per
  local L1, L2, L3, L4, L5, L6 = Ls[1], Ls[2], Ls[3], Ls[4], Ls[5], Ls[6]
  local Y1, Y2, Y3, Y4, Y5, Y6 = ys[1], ys[2], ys[3], ys[4], ys[5], ys[6]
  for r = r0, r1 - 1 do
    local ci = r // per
    local W = Wp[ci + 1]
    local pos, s, o = (r - ci * per) * rowbytes + 1, r * nb, 1
    local x1, x2, x3, x4, x5, x6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for blk = 1, nb do
      local a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = byte(W, pos, pos + 15)
      local sc = S[s + blk]
      local oa0, oa1, oa2, oa3 = o + a0, o + a1, o + a2, o + a3
      local o2, o3, o4 = o + 256, o + 512, o + 768
      local ob0, ob1, ob2, ob3 = o2 + b0, o2 + b1, o2 + b2, o2 + b3
      local oc0, oc1, oc2, oc3 = o3 + c0, o3 + c1, o3 + c2, o3 + c3
      local od0, od1, od2, od3 = o4 + d0, o4 + d1, o4 + d2, o4 + d3
        x1 = x1 + ((L1[oa0] + L1[ob0] + L1[oc0] + L1[od0])
                 + 2 * (L1[oa1] + L1[ob1] + L1[oc1] + L1[od1])
                 + 4 * (L1[oa2] + L1[ob2] + L1[oc2] + L1[od2])
                 - 8 * (L1[oa3] + L1[ob3] + L1[oc3] + L1[od3])) * sc
        x2 = x2 + ((L2[oa0] + L2[ob0] + L2[oc0] + L2[od0])
                 + 2 * (L2[oa1] + L2[ob1] + L2[oc1] + L2[od1])
                 + 4 * (L2[oa2] + L2[ob2] + L2[oc2] + L2[od2])
                 - 8 * (L2[oa3] + L2[ob3] + L2[oc3] + L2[od3])) * sc
        x3 = x3 + ((L3[oa0] + L3[ob0] + L3[oc0] + L3[od0])
                 + 2 * (L3[oa1] + L3[ob1] + L3[oc1] + L3[od1])
                 + 4 * (L3[oa2] + L3[ob2] + L3[oc2] + L3[od2])
                 - 8 * (L3[oa3] + L3[ob3] + L3[oc3] + L3[od3])) * sc
        x4 = x4 + ((L4[oa0] + L4[ob0] + L4[oc0] + L4[od0])
                 + 2 * (L4[oa1] + L4[ob1] + L4[oc1] + L4[od1])
                 + 4 * (L4[oa2] + L4[ob2] + L4[oc2] + L4[od2])
                 - 8 * (L4[oa3] + L4[ob3] + L4[oc3] + L4[od3])) * sc
        x5 = x5 + ((L5[oa0] + L5[ob0] + L5[oc0] + L5[od0])
                 + 2 * (L5[oa1] + L5[ob1] + L5[oc1] + L5[od1])
                 + 4 * (L5[oa2] + L5[ob2] + L5[oc2] + L5[od2])
                 - 8 * (L5[oa3] + L5[ob3] + L5[oc3] + L5[od3])) * sc
        x6 = x6 + ((L6[oa0] + L6[ob0] + L6[oc0] + L6[od0])
                 + 2 * (L6[oa1] + L6[ob1] + L6[oc1] + L6[od1])
                 + 4 * (L6[oa2] + L6[ob2] + L6[oc2] + L6[od2])
                 - 8 * (L6[oa3] + L6[ob3] + L6[oc3] + L6[od3])) * sc
      pos = pos + 16; o = o + 1024
    end
      Y1[yo + r + 1] = x1
      Y2[yo + r + 1] = x2
      Y3[yo + r + 1] = x3
      Y4[yo + r + 1] = x4
      Y5[yo + r + 1] = x5
      Y6[yo + r + 1] = x6
  end
end
MVB[7] = function(Wp, S, Ls, cols, r0, r1, ys, yo)
  local nb, rowbytes, per = cols // 32, cols // 2, Wp.per
  local L1, L2, L3, L4, L5, L6, L7 = Ls[1], Ls[2], Ls[3], Ls[4], Ls[5], Ls[6], Ls[7]
  local Y1, Y2, Y3, Y4, Y5, Y6, Y7 = ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7]
  for r = r0, r1 - 1 do
    local ci = r // per
    local W = Wp[ci + 1]
    local pos, s, o = (r - ci * per) * rowbytes + 1, r * nb, 1
    local x1, x2, x3, x4, x5, x6, x7 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for blk = 1, nb do
      local a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = byte(W, pos, pos + 15)
      local sc = S[s + blk]
      local oa0, oa1, oa2, oa3 = o + a0, o + a1, o + a2, o + a3
      local o2, o3, o4 = o + 256, o + 512, o + 768
      local ob0, ob1, ob2, ob3 = o2 + b0, o2 + b1, o2 + b2, o2 + b3
      local oc0, oc1, oc2, oc3 = o3 + c0, o3 + c1, o3 + c2, o3 + c3
      local od0, od1, od2, od3 = o4 + d0, o4 + d1, o4 + d2, o4 + d3
        x1 = x1 + ((L1[oa0] + L1[ob0] + L1[oc0] + L1[od0])
                 + 2 * (L1[oa1] + L1[ob1] + L1[oc1] + L1[od1])
                 + 4 * (L1[oa2] + L1[ob2] + L1[oc2] + L1[od2])
                 - 8 * (L1[oa3] + L1[ob3] + L1[oc3] + L1[od3])) * sc
        x2 = x2 + ((L2[oa0] + L2[ob0] + L2[oc0] + L2[od0])
                 + 2 * (L2[oa1] + L2[ob1] + L2[oc1] + L2[od1])
                 + 4 * (L2[oa2] + L2[ob2] + L2[oc2] + L2[od2])
                 - 8 * (L2[oa3] + L2[ob3] + L2[oc3] + L2[od3])) * sc
        x3 = x3 + ((L3[oa0] + L3[ob0] + L3[oc0] + L3[od0])
                 + 2 * (L3[oa1] + L3[ob1] + L3[oc1] + L3[od1])
                 + 4 * (L3[oa2] + L3[ob2] + L3[oc2] + L3[od2])
                 - 8 * (L3[oa3] + L3[ob3] + L3[oc3] + L3[od3])) * sc
        x4 = x4 + ((L4[oa0] + L4[ob0] + L4[oc0] + L4[od0])
                 + 2 * (L4[oa1] + L4[ob1] + L4[oc1] + L4[od1])
                 + 4 * (L4[oa2] + L4[ob2] + L4[oc2] + L4[od2])
                 - 8 * (L4[oa3] + L4[ob3] + L4[oc3] + L4[od3])) * sc
        x5 = x5 + ((L5[oa0] + L5[ob0] + L5[oc0] + L5[od0])
                 + 2 * (L5[oa1] + L5[ob1] + L5[oc1] + L5[od1])
                 + 4 * (L5[oa2] + L5[ob2] + L5[oc2] + L5[od2])
                 - 8 * (L5[oa3] + L5[ob3] + L5[oc3] + L5[od3])) * sc
        x6 = x6 + ((L6[oa0] + L6[ob0] + L6[oc0] + L6[od0])
                 + 2 * (L6[oa1] + L6[ob1] + L6[oc1] + L6[od1])
                 + 4 * (L6[oa2] + L6[ob2] + L6[oc2] + L6[od2])
                 - 8 * (L6[oa3] + L6[ob3] + L6[oc3] + L6[od3])) * sc
        x7 = x7 + ((L7[oa0] + L7[ob0] + L7[oc0] + L7[od0])
                 + 2 * (L7[oa1] + L7[ob1] + L7[oc1] + L7[od1])
                 + 4 * (L7[oa2] + L7[ob2] + L7[oc2] + L7[od2])
                 - 8 * (L7[oa3] + L7[ob3] + L7[oc3] + L7[od3])) * sc
      pos = pos + 16; o = o + 1024
    end
      Y1[yo + r + 1] = x1
      Y2[yo + r + 1] = x2
      Y3[yo + r + 1] = x3
      Y4[yo + r + 1] = x4
      Y5[yo + r + 1] = x5
      Y6[yo + r + 1] = x6
      Y7[yo + r + 1] = x7
  end
end
MVB[8] = function(Wp, S, Ls, cols, r0, r1, ys, yo)
  local nb, rowbytes, per = cols // 32, cols // 2, Wp.per
  local L1, L2, L3, L4, L5, L6, L7, L8 = Ls[1], Ls[2], Ls[3], Ls[4], Ls[5], Ls[6], Ls[7], Ls[8]
  local Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8 = ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7], ys[8]
  for r = r0, r1 - 1 do
    local ci = r // per
    local W = Wp[ci + 1]
    local pos, s, o = (r - ci * per) * rowbytes + 1, r * nb, 1
    local x1, x2, x3, x4, x5, x6, x7, x8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for blk = 1, nb do
      local a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 = byte(W, pos, pos + 15)
      local sc = S[s + blk]
      local oa0, oa1, oa2, oa3 = o + a0, o + a1, o + a2, o + a3
      local o2, o3, o4 = o + 256, o + 512, o + 768
      local ob0, ob1, ob2, ob3 = o2 + b0, o2 + b1, o2 + b2, o2 + b3
      local oc0, oc1, oc2, oc3 = o3 + c0, o3 + c1, o3 + c2, o3 + c3
      local od0, od1, od2, od3 = o4 + d0, o4 + d1, o4 + d2, o4 + d3
        x1 = x1 + ((L1[oa0] + L1[ob0] + L1[oc0] + L1[od0])
                 + 2 * (L1[oa1] + L1[ob1] + L1[oc1] + L1[od1])
                 + 4 * (L1[oa2] + L1[ob2] + L1[oc2] + L1[od2])
                 - 8 * (L1[oa3] + L1[ob3] + L1[oc3] + L1[od3])) * sc
        x2 = x2 + ((L2[oa0] + L2[ob0] + L2[oc0] + L2[od0])
                 + 2 * (L2[oa1] + L2[ob1] + L2[oc1] + L2[od1])
                 + 4 * (L2[oa2] + L2[ob2] + L2[oc2] + L2[od2])
                 - 8 * (L2[oa3] + L2[ob3] + L2[oc3] + L2[od3])) * sc
        x3 = x3 + ((L3[oa0] + L3[ob0] + L3[oc0] + L3[od0])
                 + 2 * (L3[oa1] + L3[ob1] + L3[oc1] + L3[od1])
                 + 4 * (L3[oa2] + L3[ob2] + L3[oc2] + L3[od2])
                 - 8 * (L3[oa3] + L3[ob3] + L3[oc3] + L3[od3])) * sc
        x4 = x4 + ((L4[oa0] + L4[ob0] + L4[oc0] + L4[od0])
                 + 2 * (L4[oa1] + L4[ob1] + L4[oc1] + L4[od1])
                 + 4 * (L4[oa2] + L4[ob2] + L4[oc2] + L4[od2])
                 - 8 * (L4[oa3] + L4[ob3] + L4[oc3] + L4[od3])) * sc
        x5 = x5 + ((L5[oa0] + L5[ob0] + L5[oc0] + L5[od0])
                 + 2 * (L5[oa1] + L5[ob1] + L5[oc1] + L5[od1])
                 + 4 * (L5[oa2] + L5[ob2] + L5[oc2] + L5[od2])
                 - 8 * (L5[oa3] + L5[ob3] + L5[oc3] + L5[od3])) * sc
        x6 = x6 + ((L6[oa0] + L6[ob0] + L6[oc0] + L6[od0])
                 + 2 * (L6[oa1] + L6[ob1] + L6[oc1] + L6[od1])
                 + 4 * (L6[oa2] + L6[ob2] + L6[oc2] + L6[od2])
                 - 8 * (L6[oa3] + L6[ob3] + L6[oc3] + L6[od3])) * sc
        x7 = x7 + ((L7[oa0] + L7[ob0] + L7[oc0] + L7[od0])
                 + 2 * (L7[oa1] + L7[ob1] + L7[oc1] + L7[od1])
                 + 4 * (L7[oa2] + L7[ob2] + L7[oc2] + L7[od2])
                 - 8 * (L7[oa3] + L7[ob3] + L7[oc3] + L7[od3])) * sc
        x8 = x8 + ((L8[oa0] + L8[ob0] + L8[oc0] + L8[od0])
                 + 2 * (L8[oa1] + L8[ob1] + L8[oc1] + L8[od1])
                 + 4 * (L8[oa2] + L8[ob2] + L8[oc2] + L8[od2])
                 - 8 * (L8[oa3] + L8[ob3] + L8[oc3] + L8[od3])) * sc
      pos = pos + 16; o = o + 1024
    end
      Y1[yo + r + 1] = x1
      Y2[yo + r + 1] = x2
      Y3[yo + r + 1] = x3
      Y4[yo + r + 1] = x4
      Y5[yo + r + 1] = x5
      Y6[yo + r + 1] = x6
      Y7[yo + r + 1] = x7
      Y8[yo + r + 1] = x8
  end
end

local function dequant_row(Wp, S, cols, r, out)
  local nb = cols // 32
  local ci = r // Wp.per
  local W = Wp[ci + 1]
  local pos = (r - ci * Wp.per) * (cols // 2) + 1
  for g = 0, cols // 8 - 1 do
    local p0, p1, p2, p3 = byte(W, pos, pos + 3)
    local sc = S[r * nb + g // 4 + 1]
    for k = 0, 7 do
      local q = ((p0 >> k) & 1) + 2 * ((p1 >> k) & 1) + 4 * ((p2 >> k) & 1) - 8 * ((p3 >> k) & 1)
      out[g * 8 + k + 1] = q * sc
    end
    pos = pos + 4
  end
  return out
end

MM.build_lut, MM.matvec, MM.matvec_batch = build_lut, matvec, matvec_batch

-- ---------------------------------------------------------------- loader
-- Work list: every encoded chunk of every blob, then per-matrix finalisation.
function MM.loader(t)
  local L = { t = t, jobs = {}, j = 1, parts = {} }
  local mats = {}
  -- a blob is {n=, b=85, unit=, per=, part...}: each part decodes to `per` whole units
  -- each(i, s), if given, runs in part i's own job right after it is decoded (spreads heavy
  -- per-byte work over frames instead of one long frame in done)
  local function blob(b, done, each)
    local parts = { per = b.per, unit = b.unit, n = b.n }
    local step = b.per * b.unit
    for i = 1, #b do
      L.jobs[#L.jobs + 1] = { cost = #b[i] * 4, fn = function()
        local s = b[i]
        if find(s, "=", 1, true) then s = (gsub(s, "=", "")) end   -- word breakers (export_lua.break_flag_words)
        s = gsub(s, ".....", dec5)
        local want = math.min(step, b.n - (i - 1) * step)
        if #s > want then s = sub(s, 1, want) end      -- base85 padding of the last part
        parts[i] = s; b[i] = nil
        if each then each(i, s) end
      end }
    end
    L.jobs[#L.jobs + 1] = { cost = 1000, fn = function() done(parts) end }
  end
  local function bytesOf(parts, fn)                    -- fn(byte) for every byte, in order
    for _, p in ipairs(parts) do for i = 1, #p do fn(byte(p, i)) end end
  end
  local function f16Parts(parts)
    local F, out, lo = f16_table(), {}, nil
    bytesOf(parts, function(b) if lo then out[#out + 1] = F[lo | (b << 8)]; lo = nil else lo = b end end)
    return out
  end
  for name, m in pairs(t.mats) do
    local mm = { rows = m.rows, cols = m.cols }
    mats[name] = mm
    blob(m.w, function(s) mm.W = s end)
    if m.s8 then                              -- 8-bit log2 scale codes: s = 2^(slo + code*sstep)
      local tab = {}
      for k = 0, 255 do tab[k] = 2.0 ^ (m.slo + k * m.sstep) end
      local S = {}
      blob(m.s8, function() mm.S = S end, function(_, s)
        local n = #S
        for k = 1, #s do S[n + k] = tab[byte(s, k)] end
      end)
    else
      error("weights without 8-bit scales: re-export with export/export_lua.py")
    end
  end
  for _, e in ipairs(t.cache.turns) do
    blob(e.q, function(parts) e.Q = parts; e.q = nil end)
    blob(e.s, function(parts) e.S = f16Parts(parts); e.s = nil end)
  end
  L.jobs[#L.jobs + 1] = { cost = 1000, fn = function()
    L.model = MM.new(t, mats, t.cache)
    t.mats, t.cache = nil, nil
  end }
  function L:step(budget)
    while self.j <= #self.jobs do
      local job = self.jobs[self.j]
      job.fn(); self.jobs[self.j] = false; self.j = self.j + 1
      budget = budget - job.cost
      if budget <= 0 then break end
    end
    return self.j > #self.jobs
  end
  function L:progress() return (self.j - 1) / #self.jobs end
  return L
end

-- ---------------------------------------------------------------- model
local Model = {}
Model.__index = Model

function MM.new(t, mats, cache, opts)
  local c = t.cfg
  opts = opts or {}
  local self = setmetatable({ cfg = c, mats = mats, cache = cache }, Model)
  self.dim, self.nl, self.nh, self.nkv, self.hd, self.hidden, self.vocab = c.dim, c.layers, c.heads, c.kv_heads, c.head_dim, c.hidden, c.vocab
  self.kvdim = self.nkv * self.hd
  self.norm_attn, self.norm_ffn, self.norm_final = t.norm_attn, t.norm_ffn, t.norm_final
  self.ctx = opts.ctx or c.ctx or 512
  self.max_batch = opts.max_batch or 8        -- prefill tokens per pass (shares weight reads)
  self.inv_freq = {}
  for i = 0, self.hd // 2 - 1 do self.inv_freq[i + 1] = c.rope_theta ^ (-2 * i / self.hd) end
  -- per-slot buffers (slot b = b-th token of the current pass)
  local function zeros(n) local z = {} for i = 1, n do z[i] = 0.0 end return z end
  local function slots(n) local t = {} for b = 1, self.max_batch do t[b] = zeros(n) end return t end
  self.x, self.h, self.qkv, self.att = slots(self.dim), slots(self.dim), slots(self.dim + 2 * self.kvdim), slots(self.dim)
  self.tmp, self.gu, self.act = slots(self.dim), slots(2 * self.hidden), slots(self.hidden)
  self.lut = slots(self.hidden * 32)
  self.rc, self.rs = slots(self.hd // 2), slots(self.hd // 2)
  self.tokens = {}
  self.B = 1
  self.logits = zeros(self.vocab)
  self.scores = zeros(self.ctx)
  self.sc4 = { zeros(self.ctx), zeros(self.ctx), zeros(self.ctx), zeros(self.ctx) }   -- GQA attention, 4 query heads per K/V head
  self.K, self.V = {}, {}
  for l = 1, self.nl do self.K[l], self.V[l] = {}, {} end
  self:reset()
  self.ops_body, self.ops_head = self:build_ops()
  return self
end

function Model:reset()
  self.dr, self.feeding = nil, nil
  self.pos = 0
  self.history = {}      -- token ids at positions 0 .. pos-1 (repetition penalty)
  self.turns = {}        -- { role, content, stop = position after the turn, open = assistant turn not closed yet }
end

-- K/V of the first n tokens of a cached turn -> positions nd.start ..
-- layout: [token][layer][K|V][kv head][head_dim] int8, one fp16 scale per head vector
function Model:copy_node_kv(nd, n)
  local nl, nkv, hd, kvdim = self.nl, self.nkv, self.hd, self.kvdim
  local Qp, S = nd.Q, nd.S
  for t = 0, n - 1 do
    local pos = nd.start + t
    local ci = t // Qp.per
    local Q, tokBase = Qp[ci + 1], (t - ci * Qp.per) * Qp.unit
    for l = 1, nl do
      for kind = 0, 1 do
        local dst = kind == 0 and self.K[l] or self.V[l]
        for h = 0, nkv - 1 do
          local si = ((t * nl + l - 1) * 2 + kind) * nkv + h
          local sc, qb, off = S[si + 1], tokBase + (((l - 1) * 2 + kind) * nkv + h) * hd + 1, pos * kvdim + h * hd
          for d = 0, hd - 1, 8 do
            local b1, b2, b3, b4, b5, b6, b7, b8 = byte(Q, qb + d, qb + d + 7)
            dst[off + d + 1] = (b1 >= 128 and b1 - 256 or b1) * sc
            dst[off + d + 2] = (b2 >= 128 and b2 - 256 or b2) * sc
            dst[off + d + 3] = (b3 >= 128 and b3 - 256 or b3) * sc
            dst[off + d + 4] = (b4 >= 128 and b4 - 256 or b4) * sc
            dst[off + d + 5] = (b5 >= 128 and b5 - 256 or b5) * sc
            dst[off + d + 6] = (b6 >= 128 and b6 - 256 or b6) * sc
            dst[off + d + 7] = (b7 >= 128 and b7 - 256 or b7) * sc
            dst[off + d + 8] = (b8 >= 128 and b8 - 256 or b8) * sc
          end
        end
      end
    end
  end
end

local function rmsnorm(x, w, n, eps, out)
  local ss = 0.0
  for i = 1, n do ss = ss + x[i] * x[i] end
  local k = 1 / sqrt(ss / n + eps)
  for i = 1, n do out[i] = x[i] * k * w[i] end
end

-- One pass processes s.B tokens (s.tokens[1..B], positions s.pos .. s.pos+B-1).
-- op = { rows = n | function(s), cpr = cost per row, number | function(s), fn = function(s, r0, r1) }
-- "per token" ops use rows = B (row index = slot - 1).
function Model:build_ops()
  local body, head = {}, {}
  local dim, kvdim, hd, nh, hidden = self.dim, self.kvdim, self.hd, self.nh, self.hidden
  local nkv, group, half = self.nkv, self.nh // self.nkv, self.hd // 2
  local eps = self.cfg.eps
  local M = self.mats
  local function per_token(s) return s.B end
  local function mv(m, cols, src_lut, dst)            -- matrix op over the whole pass
    -- measured: a pass of B tokens costs about (0.4 + 0.6 B) single-token passes
    return { rows = m.rows, cpr = function(s) return s.B == 1 and cols // 2 or cols * (4 + 6 * s.B) // 20 end, fn = function(s, r0, r1)
      if s.B == 1 then matvec(m.W, m.S, s[src_lut][1], cols, r0, r1, s[dst][1], 0)
      elseif MVB[s.B] then MVB[s.B](m.W, m.S, s[src_lut], cols, r0, r1, s[dst], 0)
      else matvec_batch(m.W, m.S, s[src_lut], s.B, cols, r0, r1, s[dst], 0) end
    end }
  end

  body[#body + 1] = { rows = per_token, cpr = dim * 8, fn = function(s, b0, b1)
    for b = b0 + 1, b1 do
      dequant_row(M.head.W, M.head.S, dim, s.tokens[b], s.x[b])
      local rc, rs = s.rc[b], s.rs[b]
      for i = 1, half do
        local a = (s.pos + b - 1) * s.inv_freq[i]
        rc[i], rs[i] = cos(a), sin(a)
      end
    end
  end }
  for l = 1, self.nl do
    local qkv, o, gu, down = M["qkv" .. (l - 1)], M["o" .. (l - 1)], M["gu" .. (l - 1)], M["down" .. (l - 1)]
    local K, V = self.K[l], self.V[l]
    body[#body + 1] = { rows = per_token, cpr = dim * 40, fn = function(s, b0, b1)
      for b = b0 + 1, b1 do rmsnorm(s.x[b], s.norm_attn[l], dim, eps, s.h[b]); build_lut(s.h[b], dim, s.lut[b]) end
    end }
    body[#body + 1] = mv(qkv, dim, "lut", "qkv")
    body[#body + 1] = { rows = per_token, cpr = dim * 4, fn = function(s, b0, b1)   -- RoPE on q, k; append k, v
      for b = b0 + 1, b1 do
        local q, rc, rs = s.qkv[b], s.rc[b], s.rs[b]
        for base = 0, dim + kvdim - 1, hd do
          for i = 1, half do
            local x1, x2 = q[base + i], q[base + i + half]
            q[base + i] = x1 * rc[i] - x2 * rs[i]
            q[base + i + half] = x2 * rc[i] + x1 * rs[i]
          end
        end
        local off = (s.pos + b - 1) * kvdim
        for i = 1, kvdim do K[off + i] = q[dim + i]; V[off + i] = q[dim + kvdim + i] end
      end
    end }
    if group == 4 then
    -- causal attention, GQA: row = (slot, kv head); its 4 query heads share every K/V read
    -- (per head the sums run in the same order as one head at a time, so results are identical)
    body[#body + 1] = { rows = function(s) return s.B * nkv end, cpr = function(s) return (s.pos + s.B) * hd * 8 end, fn = function(s, r0, r1)
      local S1, S2, S3, S4 = s.sc4[1], s.sc4[2], s.sc4[3], s.sc4[4]
      local scale = 1 / sqrt(hd)
      for r = r0, r1 - 1 do
        local b, kh = r // nkv + 1, r % nkv
        local q, att, T = s.qkv[b], s.att[b], s.pos + b
        local ko, q1 = kh * hd, kh * 4 * hd
        local q2, q3, q4 = q1 + hd, q1 + 2 * hd, q1 + 3 * hd
        local m1, m2, m3, m4 = -huge, -huge, -huge, -huge
        for t = 0, T - 1 do
          local kb = t * kvdim + ko
          local x1, x2, x3, x4 = 0.0, 0.0, 0.0, 0.0
          for d = 1, hd do
            local k = K[kb + d]
            x1 = x1 + q[q1 + d] * k; x2 = x2 + q[q2 + d] * k; x3 = x3 + q[q3 + d] * k; x4 = x4 + q[q4 + d] * k
          end
          x1, x2, x3, x4 = x1 * scale, x2 * scale, x3 * scale, x4 * scale
          S1[t + 1], S2[t + 1], S3[t + 1], S4[t + 1] = x1, x2, x3, x4
          if x1 > m1 then m1 = x1 end; if x2 > m2 then m2 = x2 end
          if x3 > m3 then m3 = x3 end; if x4 > m4 then m4 = x4 end
        end
        local z1, z2, z3, z4 = 0.0, 0.0, 0.0, 0.0
        for t = 1, T do
          local e1, e2, e3, e4 = exp(S1[t] - m1), exp(S2[t] - m2), exp(S3[t] - m3), exp(S4[t] - m4)
          S1[t], S2[t], S3[t], S4[t] = e1, e2, e3, e4
          z1, z2, z3, z4 = z1 + e1, z2 + e2, z3 + e3, z4 + e4
        end
        for d = 1, hd do att[q1 + d] = 0.0; att[q2 + d] = 0.0; att[q3 + d] = 0.0; att[q4 + d] = 0.0 end
        for t = 0, T - 1 do
          local vb = t * kvdim + ko
          local p1, p2, p3, p4 = S1[t + 1] / z1, S2[t + 1] / z2, S3[t + 1] / z3, S4[t + 1] / z4
          for d = 1, hd do
            local v = V[vb + d]
            att[q1 + d] = att[q1 + d] + p1 * v; att[q2 + d] = att[q2 + d] + p2 * v
            att[q3 + d] = att[q3 + d] + p3 * v; att[q4 + d] = att[q4 + d] + p4 * v
          end
        end
      end
    end }
    else
      -- causal attention; row = (slot, head)
      body[#body + 1] = { rows = function(s) return s.B * nh end, cpr = function(s) return (s.pos + s.B) * hd * 2 end, fn = function(s, r0, r1)
        local sc = s.scores
        local scale = 1 / sqrt(hd)
        for r = r0, r1 - 1 do
          local b, h = r // nh + 1, r % nh
          local q, att, T = s.qkv[b], s.att[b], s.pos + b
          local qo, ko = h * hd, (h // group) * hd
          local mx = -huge
          for t = 0, T - 1 do
            local kb, acc = t * kvdim + ko, 0.0
            for d = 1, hd do acc = acc + q[qo + d] * K[kb + d] end
            acc = acc * scale
            sc[t + 1] = acc
            if acc > mx then mx = acc end
          end
          local sum = 0.0
          for t = 1, T do local e = exp(sc[t] - mx); sc[t] = e; sum = sum + e end
          for d = 1, hd do att[qo + d] = 0.0 end
          for t = 0, T - 1 do
            local p, vb = sc[t + 1] / sum, t * kvdim + ko
            for d = 1, hd do att[qo + d] = att[qo + d] + p * V[vb + d] end
          end
        end
      end }
    end
    body[#body + 1] = { rows = per_token, cpr = dim * 32, fn = function(s, b0, b1)
      for b = b0 + 1, b1 do build_lut(s.att[b], dim, s.lut[b]) end
    end }
    body[#body + 1] = mv(o, dim, "lut", "tmp")
    body[#body + 1] = { rows = per_token, cpr = dim * 40, fn = function(s, b0, b1)
      for b = b0 + 1, b1 do
        local x, t = s.x[b], s.tmp[b]
        for i = 1, dim do x[i] = x[i] + t[i] end
        rmsnorm(x, s.norm_ffn[l], dim, eps, s.h[b]); build_lut(s.h[b], dim, s.lut[b])
      end
    end }
    body[#body + 1] = mv(gu, dim, "lut", "gu")
    body[#body + 1] = { rows = per_token, cpr = hidden * 36, fn = function(s, b0, b1)
      for b = b0 + 1, b1 do
        local g, a = s.gu[b], s.act[b]
        for i = 1, hidden do
          local v = g[i]
          a[i] = v / (1 + exp(-v)) * g[hidden + i]
        end
        build_lut(a, hidden, s.lut[b])
      end
    end }
    body[#body + 1] = mv(down, hidden, "lut", "tmp")
    body[#body + 1] = { rows = per_token, cpr = dim, fn = function(s, b0, b1)
      for b = b0 + 1, b1 do
        local x, t = s.x[b], s.tmp[b]
        for i = 1, dim do x[i] = x[i] + t[i] end
      end
    end }
  end
  -- logits for the last token of the pass only
  head[#head + 1] = { rows = 1, cpr = dim * 40, fn = function(s)
    local B = s.B
    rmsnorm(s.x[B], s.norm_final, dim, eps, s.h[B]); build_lut(s.h[B], dim, s.lut[B])
  end }
  head[#head + 1] = { rows = M.head.rows, cpr = dim // 2, fn = function(s, r0, r1)
    matvec(M.head.W, M.head.S, s.lut[s.B], dim, r0, r1, s.logits, 0)
  end }
  return body, head
end

-- ---------------------------------------------------------------- sampling
local function sample(logits, n, history, o)
  local rp = o.repetition_penalty or 1.0
  if rp ~= 1.0 then
    local seen = {}
    for i = 1, #history do
      local id = history[i] + 1
      if not seen[id] then
        seen[id] = true
        local v = logits[id]
        logits[id] = v > 0 and v / rp or v * rp
      end
    end
  end
  local temp = o.temperature or 0
  if temp <= 0 then
    local best, bi = -huge, 1
    for i = 1, n do if logits[i] > best then best, bi = logits[i], i end end
    return bi - 1
  end
  -- top-k by partial selection, then top-p, then sample
  local k = o.top_k or 40
  local ids, vals = {}, {}
  for i = 1, n do
    local v = logits[i]
    if #ids < k or v > vals[#ids] then
      local j = #ids < k and #ids + 1 or #ids
      while j > 1 and vals[j - 1] < v do ids[j], vals[j] = ids[j - 1], vals[j - 1]; j = j - 1 end
      ids[j], vals[j] = i - 1, v
    end
  end
  local mx, sum = vals[1], 0.0
  for i = 1, #vals do vals[i] = exp((vals[i] - mx) / temp); sum = sum + vals[i] end
  local top_p, cum, cut = o.top_p or 1.0, 0.0, #vals
  for i = 1, #vals do
    cum = cum + vals[i] / sum
    if cum >= top_p then cut = i; break end
  end
  local tot = 0.0
  for i = 1, cut do tot = tot + vals[i] end
  local r = random() * tot
  for i = 1, cut do
    r = r - vals[i]
    if r <= 0 then return ids[i] end
  end
  return ids[cut]
end
MM.sample = sample

-- ---------------------------------------------------------------- generation
local Gen = {}
Gen.__index = Gen

-- length of the longest complete-UTF-8 prefix of s
local function utf8_complete(s)
  local n = #s
  local i = n
  while i >= 1 and i > n - 4 do
    local c = byte(s, i)
    if c < 0x80 then return n end
    if c >= 0xC0 then
      local need = c >= 0xF0 and 4 or c >= 0xE0 and 3 or 2
      return (n - i + 1 >= need) and n or i - 1
    end
    i = i - 1
  end
  return n
end

-- feed `ids` after the current position, then sample. top: cached logits for
-- the current position (only when ids is empty).
function Model:generate(tk, ids, opts, top)
  opts = opts or {}
  local g = setmetatable({ m = self, tk = tk, o = opts, feed = ids, fi = 1, max_new = opts.max_new or 128,
                           out = {}, text = {}, pending = "", stop = opts.stop or { [2] = true, [0] = true },
                           oi = 1, row = 0, done = false, ready = false }, Gen)
  if #ids == 0 then
    local lg = self.logits
    for i = 1, self.vocab do lg[i] = -1e30 end
    for k = 1, #top.ids do lg[top.ids[k] + 1] = top.logits[k] end
    g.ready = true
  else
    g:begin_pass()
  end
  return g
end

-- token layout of a turn (same as export/export_lua.py turn_text)
local function turn_text(role, content)
  if role == "system" then return "<|im_start|>system\n" .. content .. "<|im_end|>\n" end
  if role == "user" then return "<|im_start|>user\n" .. content .. "<|im_end|>\n<|im_start|>assistant\n" end
  return content .. "<|im_end|>\n"
end

function Model:persona(name)
  local i = self.cache.personas[name or "default"]
  return i and self.cache.turns[i].text or "You are a helpful assistant"
end

-- drop turns after the first k and move the position back to the end of turn k
function Model:rewind(k)
  local turns = self.turns
  for i = #turns, k + 1, -1 do turns[i] = nil end
  local pos = k > 0 and turns[k].stop or 0
  for i = #self.history, pos + 1, -1 do self.history[i] = nil end
  self.pos = pos
end

-- Leading turns of msgs already in the K/V, or in the exported cache (copied in): returns k = turns
-- now in the K/V (the position is right after them) and the cached first-step logits if all of msgs
-- came from the cache. all = false (reply): the last message is the new one and is not matched
-- against the K/V; all = true (draft): every message is a complete turn.
function Model:lead(msgs, all)
  -- turns whose tokens were fully processed (a cancelled generation may leave stale ones)
  local turns = self.turns
  while #turns > 0 and turns[#turns].stop > self.pos do turns[#turns] = nil end
  -- 1) leading turns already in the K/V
  local k = 0
  while k < #msgs - (all and 0 or 1) and k < #turns and turns[k + 1].role == msgs[k + 1].role and turns[k + 1].content == msgs[k + 1].content do k = k + 1 end
  -- 2) leading turns in the exported cache
  local ct, chain, parent = self.cache.turns, {}, 0
  while #chain < #msgs do
    local m, found = msgs[#chain + 1], nil
    for i, e in ipairs(ct) do
      if e.parent == parent and e.role == m.role and e.text == m.content then found = i; break end
    end
    if not found then break end
    chain[#chain + 1] = found; parent = found
  end
  local p = #chain
  if not all and p == #msgs and not ct[chain[p]].top then p = p - 1 end
  local top = nil
  if p > k then
    self:rewind(k)
    for i = k + 1, p do
      local e = ct[chain[i]]
      self:copy_node_kv(e, #e.ids)
      for j = 1, #e.ids do self.history[e.start + j] = e.ids[j] end
      turns[i] = { role = e.role, content = e.text, stop = e.start + #e.ids }
      self.pos = e.start + #e.ids
    end
    if p == #msgs then top = ct[chain[p]].top end
    k = p
  else
    self:rewind(k)
  end
  return k, top
end

-- Draft prefill while the player types (game/chat.lua): msgs = the complete turns the next request
-- will start with (system first), partial = the user message typed so far. Feeds, in K/V-only passes,
-- the tokens reply() will need up to the end of partial (if BPE later merges the last ones with what
-- is typed next, the longest common prefix just gets shorter), then parks: the position goes back, the K/V beyond it stays, and reply()
-- skips the tokens that still match. Returns a feeder with step(budget) -> done, or nil.
function Model:draft(tk, msgs, partial)
  if msgs[1].role ~= "system" then
    local m2 = { { role = "system", content = self:persona() } }
    for i = 1, #msgs do m2[i + 1] = msgs[i] end
    msgs = m2
  end
  local old = self.dr
  self:park()
  local k = self:lead(msgs, true)
  local ids = {}
  local last = self.turns[#self.turns]
  if last and last.open then
    if last.tail then ids[#ids + 1] = last.tail end
    tk:encode("<|im_end|>\n", ids)
  end
  for i = k + 1, #msgs do tk:encode(turn_text(msgs[i].role, msgs[i].content), ids) end
  tk:encode("<|im_start|>user\n" .. partial, ids)
  local base, n, start = self.pos, #ids, 0
  if old and old.base == base then
    local m = math.min(old.fed, n)
    while start < m and old.ids[start + 1] == ids[start + 1] do start = start + 1 end
  end
  self.dr = { base = base, ids = ids, fed = start }
  if start >= n then return nil end
  for j = 1, start do self.history[base + j] = ids[j] end
  self.pos = base + start
  local feed = {}
  for j = start + 1, n do feed[#feed + 1] = ids[j] end
  local g = self:generate(tk, feed, { feed_only = true })
  self.feeding = g
  local model = self
  return { step = function(_, budget)
    if model.feeding ~= g then return true end
    local _, done = g:step(budget)
    model.dr.fed = model.pos - base
    if done then model:park() end
    return done
  end }
end

-- stop a running draft feed: back to the draft's base position (its K/V stays for reply())
function Model:park()
  local d, g = self.dr, self.feeding
  self.feeding = nil
  if d and g then
    d.fed = math.max(0, self.pos - d.base)
    self.pos = d.base
    for i = #self.history, d.base + 1, -1 do self.history[i] = nil end
  end
end

-- Answer a conversation: msgs = { {role=, content=}, ... } (system optional,
-- then alternating user / assistant, ending with a user message).
function Model:reply(tk, msgs, opts)
  opts = opts or {}
  self:park()
  if msgs[1].role ~= "system" then
    local m2 = { { role = "system", content = self:persona() } }
    for i = 1, #msgs do m2[i + 1] = msgs[i] end
    msgs = m2
  end
  local k, top = self:lead(msgs, false)
  local turns = self.turns
  local reused = self.pos
  -- 3) tokens for the remaining turns
  local ids = {}
  local last = turns[#turns]
  if last and last.open then
    if last.tail then ids[#ids + 1] = last.tail end
    tk:encode("<|im_end|>\n", ids)
    last.open, last.stop = false, self.pos + #ids
  end
  for i = k + 1, #msgs do
    tk:encode(turn_text(msgs[i].role, msgs[i].content), ids)
    turns[i] = { role = msgs[i].role, content = msgs[i].content, stop = self.pos + #ids }
  end
  if self.pos + #ids + (opts.max_new or 128) >= self.ctx and reused > 0 then
    self:reset()                                  -- context full: start over (presets still apply)
    return self:reply(tk, msgs, opts)
  end
  -- tokens fed while the player was typing (Model:draft) that match this request: already in the K/V
  local d, skip = self.dr, 0
  self.dr = nil
  if d and d.base == self.pos then
    local n = math.min(d.fed, #ids - 1)
    while skip < n and d.ids[skip + 1] == ids[skip + 1] do skip = skip + 1 end
  end
  local feed = ids
  if skip > 0 then
    for j = 1, skip do self.history[self.pos + j] = ids[j] end
    self.pos = self.pos + skip
    feed = {}
    for j = skip + 1, #ids do feed[#feed + 1] = ids[j] end
  end
  local g = self:generate(tk, feed, opts, top)
  g.drafted = skip
  g.reused = reused
  g.ids = {}
  for i = 1, reused do g.ids[i] = self.history[i] end
  for i = 1, #ids do g.ids[reused + i] = ids[i] end
  return g
end

-- new conversation with a persona (opts.persona, default "default") or opts.system
function Model:chat(tk, text, opts)
  opts = opts or {}
  return self:reply(tk, { { role = "system", content = opts.system or self:persona(opts.persona) }, { role = "user", content = text } }, opts)
end

-- next user turn of the current conversation
function Model:say(tk, text, opts)
  local msgs = self:messages()
  msgs[#msgs + 1] = { role = "user", content = text }
  return self:reply(tk, msgs, opts)
end

function Model:messages()
  local msgs = {}
  for i, t in ipairs(self.turns) do msgs[i] = { role = t.role, content = t.content } end
  return msgs
end

-- next pass: up to max_batch not-yet-processed tokens of feed
function Gen:begin_pass()
  local m = self.m
  local B = #self.feed - self.fi + 1
  if B > m.max_batch then B = m.max_batch end
  m.B = B
  for b = 1, B do m.tokens[b] = self.feed[self.fi + b - 1] end
  self.ops, self.oi, self.row = m.ops_body, 1, 0
end

-- run up to `budget` work units; returns (new text, done)
function Gen:step(budget)
  local m = self.m
  local emitted = {}
  while budget > 0 and not self.done do
    if self.ready then
      self.ready = false
      self:after_logits(emitted)
      budget = budget - 10000
      if self.done then break end
    end
    local op = self.ops[self.oi]
    local rows, cpr = op.rows, op.cpr
    if type(rows) == "function" then rows = rows(m) end
    if type(cpr) == "function" then cpr = cpr(m) end
    local take = rows - self.row
    local fit = budget // cpr
    if fit < 1 then fit = 1 end
    if take > fit then take = fit end
    op.fn(m, self.row, self.row + take)
    budget = budget - take * cpr
    self.row = self.row + take
    if self.row >= rows then
      self.row = 0
      self.oi = self.oi + 1
      if self.oi > #self.ops then self:pass_done(emitted) end
    end
  end
  return concat(emitted), self.done
end

function Gen:pass_done(emitted)
  local m = self.m
  if self.ops == m.ops_body then
    for b = 1, m.B do m.history[#m.history + 1] = m.tokens[b] end
    m.pos = m.pos + m.B
    self.fi = self.fi + m.B
    if self.fi <= #self.feed then self:begin_pass()   -- more prompt tokens: no logits needed
    elseif self.o.feed_only then self.done = true     -- draft prefill: K/V only (Model:draft)
    else self.ops, self.oi = m.ops_head, 1 end        -- logits for the last token
    return
  end
  self:after_logits(emitted)
end

function Gen:after_logits(emitted)
  local m = self.m
  if self.o.on_logits then self.o.on_logits(m.logits) end
  -- no_repeat_ngram = n: within this answer, a token that would repeat an n-gram already
  -- generated is banned ("每天四个委托每天四个"); the prompt is not affected
  local nr, out = self.o.no_repeat_ngram, self.out
  if nr and #out >= nr then
    local L = #out
    for s = 1, L - nr + 1 do
      local same = true
      for j = 1, nr - 1 do if out[s + j - 1] ~= out[L - nr + 1 + j] then same = false; break end end
      if same then m.logits[out[s + nr - 1] + 1] = -1e30 end
    end
  end
  local id = sample(m.logits, m.vocab, m.history, self.o)
  self.out[#self.out + 1] = id
  if self.stop[id] or #self.out >= self.max_new or m.pos + 1 >= m.ctx then
    self.done = true
  else
    self.feed[#self.feed + 1] = id
    self:begin_pass()
  end
  if not self.stop[id] then
    local s = self.pending .. self.tk:piece(id)
    local k = utf8_complete(s)
    emitted[#emitted + 1] = sub(s, 1, k)
    self.text[#self.text + 1] = sub(s, 1, k)
    self.pending = sub(s, k + 1)
  end
  if self.done and #self.pending > 0 then
    emitted[#emitted + 1] = self.pending; self.text[#self.text + 1] = self.pending; self.pending = ""
  end
  if self.done then
    -- cut by max_new / ctx: the last sampled token was shown but not fed; the next turn feeds it first
    m.turns[#m.turns + 1] = { role = "assistant", content = concat(self.text), stop = m.pos, open = true,
                              tail = not self.stop[id] and id or nil }
  end
end

MM.Model, MM.Gen = Model, Gen
return MM
