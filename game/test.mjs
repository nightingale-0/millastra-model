// Simulator test of the TeyvatAI 派蒙 chat UI: real Lua, real clicks, PNG wireframes and layout checks.
//   node game/test.mjs [--full]       (build first: node game/build.mjs [--full])
// Each step renders game/out/shots/NN-name.png and checks the visible scene:
//   - no Lua errors
//   - visible controls stay inside the 1280x720 canvas (Background excepted)
//   - visible click targets (cursor nodes) do not overlap
//   - every text fits its box (CJK ~1 em, ASCII ~0.55 em per char; lines from the box height)
import fs from 'node:fs'; import path from 'node:path'; import { pathToFileURL } from 'node:url'; import assert from 'node:assert/strict';
const SIM = process.env.QXQY_SIM || path.resolve('../6nimmt/work/simulator');
const { createRequire } = await import('node:module');
const req = createRequire(pathToFileURL(path.join(SIM, 'package.json')));
const { GlobalFonts } = req('@napi-rs/canvas');
for (const f of ['C:/Windows/Fonts/msyh.ttc', '/System/Library/Fonts/STHeiti Medium.ttc']) if (fs.existsSync(f)) { GlobalFonts.registerFromPath(f, 'Microsoft YaHei UI'); break; }
const { createStudio } = await import(pathToFileURL(path.join(SIM, 'studio/index.js')));
const { renderScenePng } = await import(pathToFileURL(path.join(SIM, 'studio/host-png.js')));
const FULL = process.argv.includes('--full');
const SHOTS = 'game/out/shots'; fs.mkdirSync(SHOTS, { recursive: true });
const st = createStudio(JSON.parse(fs.readFileSync(`game/out/paimon-chat-${FULL ? 'full' : 'ui'}.save.json`)));
const problems = [];
let shot = 0;

function boxes(scene) {
  const byId = new Map(scene.nodes.map(n => [n.id, n]));
  // the topmost node's matrix is in screen pixels (origin top-left, y down); every node
  // below it is offset from its parent's centre with y pointing up
  const abs = n => {
    let x = 0, y = 0, m = n;
    for (; byId.get(m.parent); m = byId.get(m.parent)) { x += m.matrix.tx; y -= m.matrix.ty; }
    return { x: m.matrix.tx + x, y: m.matrix.ty + y };
  };
  return scene.nodes.map(n => { const c = abs(n); return { ...n, cx: c.x, cy: c.y, x0: c.x - n.sourceWidth / 2, x1: c.x + n.sourceWidth / 2, y0: c.y - n.sourceHeight / 2, y1: c.y + n.sourceHeight / 2 }; });
}
function textWidth(s, size) { let w = 0; for (const ch of s) w += ch.charCodeAt(0) < 128 ? 0.55 * size : size; return w; }
function check(label, snap) {
  const errs = snap.logs.filter(l => l.level === 'lua-error');
  if (errs.length) problems.push(`${label}: lua error ${errs.map(e => e.text).join(' | ').slice(0, 300)}`);
  // errors caught by chat.lua's guard only reach the log; the injected mock fault is expected
  const caught = snap.logs.filter(l => /\[paimon\] error/.test(l.text || '') && !/mock fault/.test(l.text || ''));
  if (caught.length) problems.push(`${label}: caught error ${caught.map(e => e.text).join(' | ').slice(0, 300)}`);
  const bs = boxes(snap.scene);
  for (const b of bs) {
    if (b.name === 'Background') continue;
    if (b.x0 < -1 || b.y0 < -1 || b.x1 > 1281 || b.y1 > 721) problems.push(`${label}: ${b.name} outside the canvas (${b.x0.toFixed(0)},${b.y0.toFixed(0)})-(${b.x1.toFixed(0)},${b.y1.toFixed(0)})`);
    if (b.kind === 'textbox' && b.text) {
      const lines = Math.max(1, Math.floor((b.sourceHeight / b.fontSize - 0.9) / 1.3 + 0.01));
      const need = b.text.split('\n').reduce((s, part) => s + Math.max(1, Math.ceil(textWidth(part, b.fontSize) / b.sourceWidth)), 0);
      if (need > lines) problems.push(`${label}: ${b.name} text needs ${need} lines, box holds ${lines}: "${b.text.slice(0, 40)}"`);
    }
  }
  const cur = bs.filter(b => b.kind === 'cursor');
  for (let i = 0; i < cur.length; i++) for (let j = i + 1; j < cur.length; j++) {
    const a = cur[i], b = cur[j];
    if (a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1) problems.push(`${label}: click targets overlap ${a.name} / ${b.name}`);
  }
}
const node = (snap, name) => snap.scene.nodes.find(n => n.name === name);
const textOf = (snap, name) => node(snap, name)?.text;
function step(label) {
  const snap = st.playGet({ view: true });
  check(label, snap);
  const file = path.join(SHOTS, String(++shot).padStart(2, '0') + '-' + label + '.png');
  fs.writeFileSync(file, renderScenePng(snap.scene, 1280, 720).data);
  return snap;
}
const frames = (n, dt = 1 / 60) => { for (let i = 0; i < n; i++) st.playStep(dt); };
const type = s => { for (const ch of s) st.playClick('Key' + ch.toUpperCase()); };
// pick a word from the first candidate page, as a player would; asserts it is offered there
function choose(word) {
  const s = st.playGet({ view: true });
  for (let k = 1; k <= 7; k++) if (textOf(s, 'Cand' + k + 'Text') === word) { st.playClick('Cand' + k); return k; }
  const offered = [1, 2, 3, 4, 5, 6, 7].map(k => textOf(s, 'Cand' + k + 'Text')).filter(Boolean).join(' ');
  throw new Error(`"${word}" not on the first page for "${textOf(s, 'Pinyin')}": ${offered}`);
}
function waitAnswer(maxFrames) {
  for (let i = 0; i < maxFrames; i++) { st.playStep(1 / 60); const s = st.playGet({ view: true }); if (textOf(s, 'SendText') === '发送') return i; }
  throw new Error('no answer within ' + maxFrames + ' frames');
}
// visible texts of the conversation slots, top to bottom
function convo(s) {
  const out = [];
  for (let k = 1; k <= 5; k++) for (const [p, who] of [['MsgU', 'u'], ['MsgP', 'p'], ['MsgS', 's']]) { const t = textOf(s, p + k); if (t !== undefined && node(s, p + k)) out.push(who + ':' + t); }
  return out;
}
const MAXF = FULL ? 20000 : 600;
const u32 = v => v >>> 0;

try {
  let s = st.playStart({ view: true });
  assert.equal(s.mountError, null);
  s = step('loading');
  assert.ok(node(s, 'Hero'), 'empty state greeting');
  frames(FULL ? 1 : 60);
  for (let i = 0; i < 100000 && /醒来/.test(textOf(st.playGet({ view: true }), 'Hint') || ''); i++) st.playStep(1 / 60);   // loading starts after a 0.5 s idle warm-up
  s = step('idle');
  assert.match(textOf(s, 'Hint'), /上下文 0 \/ 512/);
  assert.equal(textOf(s, 'InputText'), '问问派蒙…');
  if (!FULL) { assert.match(textOf(s, 'SettleText'), /^结算 \d+s$/); st.playClick('Settle'); assert.doesNotMatch(textOf(st.playGet({ view: true }), 'Stats') || '', /结算/); }

  // IME
  type('nihao');
  s = step('typing-nihao');
  assert.equal(textOf(s, 'Pinyin'), 'nihao');
  assert.equal(textOf(s, 'Cand1Text'), '你好');
  st.playClick('Cand1'); st.playClick('KeyQuestion');
  type('paimeng'); st.playClick('KeySpace');
  type('nizai'); choose('你'); choose('在');
  type('gan'); choose('干'); type('sm');
  s = step('typing-abbrev');
  choose('什么'); st.playClick('KeyQuestion');
  s = step('composed');
  assert.equal(textOf(s, 'InputText'), '你好？派蒙你在干什么？');
  // 中/英: pending pinyin goes in as letters, then plain English with spaces and half-width punctuation
  type('ok'); st.playClick('Lang'); type('hi'); st.playClick('KeySpace'); type('paimon'); st.playClick('KeyExclaim');
  s = step('english');
  assert.equal(textOf(s, 'LangText'), '英'); assert.equal(textOf(s, 'KeySpaceText'), '空格'); assert.equal(textOf(s, 'KeyExclaimText'), '!');
  assert.equal(textOf(s, 'InputText'), '你好？派蒙你在干什么？okhi paimon!');
  st.playClick('Lang');
  for (let i = 0; i < 'okhi paimon!'.length; i++) st.playClick('KeyBack');
  s = st.playGet({ view: true });
  assert.equal(textOf(s, 'LangText'), '中'); assert.equal(textOf(s, 'InputText'), '你好？派蒙你在干什么？');
  st.playClick('Send');
  st.playStep(1 / 60); st.playStep(1 / 60);
  s = step('thinking');
  assert.equal(textOf(s, 'MsgU1'), '你好？派蒙你在干什么？'); assert.ok(node(s, 'MsgU1S'), 'short bubble for an 11-character question');
  assert.equal(textOf(s, 'SendText'), '…');
  assert.ok([1, 2, 3, 4, 5].some(k => /^派蒙(思考中|正在醒来)…（\d+秒）$/.test(textOf(s, 'MsgW' + k) || '')), 'grey waiting status before the first character');
  waitAnswer(MAXF);
  s = step('answered');
  assert.ok(textOf(s, 'MsgP2').length > 3);
  assert.match(textOf(s, 'Stats'), /首字/);
  console.log('answer:', textOf(s, 'MsgP2'), '|', textOf(s, 'Stats'), '|', textOf(s, 'Hint'));

  // suggestion pop-up
  st.playClick('Plus');
  s = step('menu');
  assert.ok(node(s, 'Menu1') && node(s, 'MenuPanel'));
  st.playClick('Menu1');
  s = st.playGet({ view: true }); assert.equal(node(s, 'Menu1'), undefined, 'menu closes after a pick');
  waitAnswer(MAXF);
  s = step('menu-answered');

  // dark theme
  st.playClick('Theme');
  s = step('dark');
  assert.equal(u32(node(s, 'Background').bgColor), 0xff1b1b1a);
  assert.equal(u32(node(s, 'MsgU1').fontColor), 0xffececea);
  st.playClick('Theme');
  s = st.playGet({ view: true }); assert.equal(u32(node(s, 'Background').bgColor), 0xffffffff);

  // filter: type 外挂 and send
  type('wai'); choose('外'); type('gua'); choose('挂'); st.playClick('Send'); waitAnswer(MAXF);   // character by character
  s = step('blocked');
  assert.match(textOf(s, 'Stats'), /已拦截/);
  console.log('blocked:', convo(s).slice(-1)[0]);

  // rewind to the second question: it returns to the input, later messages disappear
  s = st.playGet({ view: true });
  const before = convo(s);
  const backK = [1, 2, 3, 4, 5].find(k => node(s, 'Back' + k) && textOf(s, 'MsgU' + k) === '你是谁？');
  assert.ok(backK, 'a 回退 button next to 你是谁？');
  st.playClick('Back' + backK);
  s = step('rewound');
  assert.equal(textOf(s, 'InputText'), '你是谁？');
  assert.ok(convo(s).length < before.length);
  assert.match(textOf(s, 'Stats'), /回退/);
  st.playClick('KeyClear');

  // context window: keep asking until the oldest exchanges get compressed
  let compressedNote = FULL;                      // with the real model this loop would take hours in fengari
  for (let i = 0; i < 12 && !compressedNote; i++) {
    st.playClick('Plus'); st.playClick('Menu' + (1 + (i % 6))); waitAnswer(MAXF);
    s = st.playGet({ view: true });
    compressedNote = convo(s).some(t => t.startsWith('s:') && /压缩/.test(t));
  }
  if (!FULL) {
    s = step('compressed');
    assert.ok(compressedNote, 'compression note shown');
    assert.match(textOf(s, 'Hint'), /已压缩/);
    console.log('context:', textOf(s, 'Hint'));
    assert.ok(node(s, 'ScrollUp'));
    st.playClick('ScrollUp'); st.playClick('ScrollUp');
    s = step('scrolled');
  }

  // 结算: dead for the first 30 s, then sends PaimonSettle (PaimonWin already went out at 30 s)
  s = st.playGet({ view: true });
  if (/\d+s$/.test(textOf(s, 'SettleText'))) {
    st.playClick('Settle');
    assert.doesNotMatch(textOf(st.playGet({ view: true }), 'Stats') || '', /结算/, 'no reaction before 30 s');
    for (let i = 0; i < 31 * 4; i++) st.playStep(1 / 4);
  }
  s = step('settle-ready');
  assert.equal(textOf(s, 'SettleText'), '结算');
  st.playClick('Settle');
  assert.match(textOf(st.playGet({ view: true }), 'Stats'), /正在结算/);
  st.playClick('NewChat');
  s = step('new-chat');
  assert.ok(node(s, 'Hero')); assert.equal(convo(s).length, 0);

  // crash safety (mock only): an error mid-answer ends the turn with a message, the next question works
  if (!FULL) {
  st.playClick('Lang'); type('crash'); st.playClick('Lang'); st.playClick('Send');
  for (let i = 0; i < 40; i++) st.playStep(1 / 60);
  s = step('crash-recovered');
  assert.equal(textOf(s, 'SendText'), '发送', 'not stuck after an error');
  assert.ok(convo(s).some(t => /脑袋卡住了/.test(t)), 'error message in Paimon slot');
  assert.match(textOf(s, 'Stats'), /出错了/);
  st.playClick('Plus'); st.playClick('Menu1'); waitAnswer(MAXF);
  s = st.playGet({ view: true }); assert.match(textOf(s, 'Stats'), /首字/, 'answers again after the error');
  st.playClick('NewChat');
  }

  // physical keyboard: custom keys 1-26 = a-z, 28 = space (pick), 29 = enter (send)
  for (const ch of 'nihao') st.playKey('KeyboardCraftspersonKey' + (ch.charCodeAt(0) - 96) + 'Down');
  st.playKey('KeyboardCraftspersonKey28Down');
  s = st.playGet({ view: true }); assert.equal(textOf(s, 'InputText'), '你好');
  st.playKey('KeyboardCraftspersonKey29Down');
  waitAnswer(MAXF);
  s = step('physical-keys');
  assert.equal(textOf(s, 'MsgU1'), '你好');
} finally { st.playStop(); }
if (problems.length) { console.log('LAYOUT / RUNTIME PROBLEMS:\n  ' + [...new Set(problems)].join('\n  ')); process.exitCode = 1; }
else console.log(`all checks passed; ${shot} screenshots in ${SHOTS}`);
