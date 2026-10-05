// TeyvatAI 派蒙 — chat UI for 千星奇域: client control template + one client script (+ a tiny server boot script).
//   node game/build.mjs                  -> game/out/paimon-chat-ui.{save.json,gia}   (mock model: UI / IME test)
//   node game/build.mjs --full           -> also game/out/paimon-chat-full.{save.json,gia} (the real 26M model, ~21MB script)
//   node game/build.mjs --full --install -> also copy the GIAs to the editor's BeyondLocal/Beyond_Local_Export
// Uses the miliastra-beyond-simulator checkout of the 6nimmt project (QXQY_SIM overrides the path).
// Authoring rules learnt on the real client (6nimmt):
//  - text hugs the TOP of its box (GIA drops vertical alignment) -> boxes are tall and placed by their top
//  - only buttons are cursor nodes; panels are plain rects (cursor groups swallow clicks)
//  - stacking is applied at runtime (LAYERS + SetAsFirstSibling)
//  - one client script carries everything: the sandbox has no require / exports between scripts
//  - glyphs like ▲▶⌫ are not in every game font: buttons use words
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const SIM = process.env.QXQY_SIM || path.resolve(ROOT, '../6nimmt/work/simulator');
const { createStudio } = await import(pathToFileURL(path.join(SIM, 'studio/index.js')));
const { createNode } = await import(pathToFileURL(path.join(SIM, 'studio/ui/authoring.js')));
const OUT = path.join(HERE, 'out'); fs.mkdirSync(OUT, { recursive: true });
const FULL = process.argv.includes('--full');

// ---------------------------------------------------------------- themes (role -> colour, AARRGGBB)
// light / dark, Claude-like: neutral paper, dark ink, warm orange accent
const THEMES = {
  light: { page: 0xffffffff, ink: 0xff1f1f1d, muted: 0xff8a8a84, line: 0xffe3e2dc, bubble: 0xfff0efea, surface: 0xffffffff,
           key: 0xfff6f5f1, key2: 0xffe8e7e1, accent: 0xffd97757, onAccent: 0xffffffff, btn: 0xfff3f2ee, shade: 0x22000000 },
  dark:  { page: 0xff1b1b1a, ink: 0xffececea, muted: 0xff9a9a94, line: 0xff3a3a37, bubble: 0xff2f2f2d, surface: 0xff242423,
           key: 0xff2c2c2a, key2: 0xff383836, accent: 0xffd97757, onAccent: 0xffffffff, btn: 0xff2a2a28, shade: 0x55000000 },
};
const C = THEMES.light; // authored colours = light theme; THEME_ROLES lets the script recolour at runtime
const AVATAR = 105142;  // 装饰图案-彩色 round face (6nimmt emote button): placeholder for Paimon's portrait
const TEMPLATE_ID = Number(process.env.TEMPLATE_ID || 1073760001); // the editor re-assigns it on import: set boot param templateId

const root = createNode('container', { id: 'paimon-root', guid: TEMPLATE_ID, name: 'PaimonRoot', showCursor: true });
function place(n, x, y, w, h) { for (const t of [...Object.values(n.transformByPlatform), ...Object.values(n.transformByCanvas)]) { t.size = { x: w, y: h }; t.offset = { x: x - 640, y: 360 - y }; } return n; }
place(root, 640, 360, 1280, 720);
const GLYPH_TOP = 0.3, LINE = 1.3;
function textBox(size, cy, lines) { const top = cy - GLYPH_TOP * size - (0.9 * size + (lines - 1) * LINE * size) / 2; const h = Math.ceil((GLYPH_TOP + lines * LINE + 0.6) * size); return { y: top + h / 2, h }; }
const textNode = (name, value, size, fg, align = 'Middle') => createNode('textbox', { id: name, name, text: value, fontSize: size, minimumFontSize: size, bgColor: 0, fontColor: fg, enableOutline: false, horizontalAlignment: align, verticalAlignment: 'Top' });
const rectNode = (name, fill) => createNode('textbox', { id: name, name, text: '', fontSize: 12, minimumFontSize: 12, bgColor: fill, fontColor: 0, enableOutline: false, horizontalAlignment: 'Middle', verticalAlignment: 'Top' });
const imageNode = (name, id) => createNode('image', { id: name, name, imageId: id, imageColor: 0xffffffff });
const add = n => { root.children.push(n); return n; };
const ROLES = [];        // [path, 'bg'|'fg', role]: recoloured by the script when the theme changes
const role = (p, field, r) => ROLES.push([p, field, r]);
function label(name, value, cx, cy, w, size, fg = 'ink', lines = 1, align = 'Middle') {
  const b = textBox(size, cy, lines); add(place(textNode(name, value, size, C[fg], align), cx, b.y, w, b.h)); role(name, 'fg', fg);
}
function rect(name, cx, cy, w, h, fill) { add(place(rectNode(name, C[fill]), cx, cy, w, h)); role(name, 'bg', fill); }
function image(name, id, cx, cy, s) { add(place(imageNode(name, id), cx, cy, s, s)); }
// cursor button: [text, fill, edge] front-to-back, the fill is the click target
function button(name, value, cx, cy, w, h, fill = 'btn', fg = 'ink', size = 16, edge = 'line') {
  const n = place(createNode('cursor', { id: name, name }), cx, cy, w, h);
  const b = textBox(size, 360, 1);
  n.children.push(place(textNode(name + 'Text', value, size, C[fg]), 640, b.y, w - 6, b.h));
  n.children.push(place(rectNode(name + 'Fill', C[fill]), 640, 360, w, h));
  if (edge) n.children.push(place(rectNode(name + 'Edge', C[edge]), 640, 360, w + 2, h + 2));
  role(name + '/' + name + 'Text', 'fg', fg); role(name + '/' + name + 'Fill', 'bg', fill); if (edge) role(name + '/' + name + 'Edge', 'bg', edge);
  add(n); return n;
}

// ---------------------------------------------------------------- layout (1280x720, front-to-back)
// top bar 0-56 | conversation column x 200-1080, y 70-402 | composer 418-548 | keyboard 566-690
const COL = { x0: 200, x1: 1080 }, SLOT = { y0: 70, h: 60, gap: 8 };
const slotY = k => SLOT.y0 + SLOT.h / 2 + (k - 1) * (SLOT.h + SLOT.gap);
// pop-up menu of suggested questions (front-most), opened by "＋ 推荐" in the composer
// keep in sync with distill/build_sft.py SUGGEST (cached first questions: no prefill when picked)
const SUGGEST = ['你是谁？', '派蒙，我们今天去哪儿？', '你饿了吗？', '给我讲个笑话吧。', '璃月的神是谁？', '七种元素都有哪些？'];
SUGGEST.forEach((q, i) => button('Menu' + (i + 1), q, 380, 186 + i * 40, 340, 34, 'surface', 'ink', 15, null));
label('MenuTitle', '推荐问题', 380, 150, 340, 13, 'muted');
rect('MenuPanel', 380, 266, 364, 268, 'surface');
rect('MenuEdge', 380, 266, 366, 270, 'line');
// top bar
label('Title', 'TeyvatAI 派蒙', 150, 32, 220, 22, 'ink', 1, 'Left');
label('Tag', '离线 · 26M · 4bit', 330, 28, 150, 12, 'muted');
label('Stats', '', 760, 28, 420, 13, 'muted');
button('Theme', '深色', 1042, 28, 64, 34);
button('NewChat', '新对话', 1128, 28, 88, 34);
button('Settle', '结算 30s', 1224, 28, 88, 34, 'btn', 'muted', 14);   // usable 30 s after entering (chat.lua SETTLE_AFTER)
rect('TopLine', 640, 56, 1280, 1, 'line');
// empty state
label('Hero', '旅行者，今天想聊什么？', 640, 200, 800, 30, 'ink');
label('HeroSub', '点「＋ 推荐」选一个问题，或者用键盘打拼音', 640, 248, 800, 15, 'muted');
// conversation: 5 slots; Paimon = avatar + plain text, you = right bubble with a 回退 button; system notes centred
button('ScrollUp', '更早', 1140, slotY(1), 60, 30, 'btn', 'muted', 13);
button('ScrollDown', '最新', 1140, slotY(5), 60, 30, 'btn', 'muted', 13);
for (let k = 1; k <= 5; k++) {
  const y = slotY(k);
  button('Back' + k, '回退', 524, y, 60, 28, 'page', 'muted', 13);
  label('MsgP' + k, '', 655, y, 810, 15, 'ink', 3, 'Left');
  label('MsgW' + k, '', 655, y, 810, 13, 'muted', 1, 'Left');   // 派蒙思考中… while waiting for the first character
  image('MsgP' + k + 'Icon', AVATAR, 216, y - 12, 30);
  // your message: one right-aligned line in a short / medium / long bubble, or two lines in a tall one
  label('MsgU' + k, '', 814, y, 508, 15, 'ink', 1, 'Right');
  label('MsgU' + k + 'B', '', 824, y, 492, 15, 'ink', 2, 'Left');
  for (const [v, w] of [['S', 200], ['M', 340], ['L', 512]]) rect('MsgU' + k + v, COL.x1 - w / 2, y, w, 36, 'bubble');
  rect('MsgU' + k + 'T', 824, y, 512, 56, 'bubble');
  label('MsgS' + k, '', 640, y, 700, 13, 'muted');
}
// composer card: input line | pinyin + candidates | 推荐 ... 发送
const COMP = { y: 483, h: 130 };
label('InputText', '', 650, 440, 840, 18, 'ink', 1, 'Left');
label('Pinyin', '', 268, 482, 110, 15, 'muted');
for (let k = 1; k <= 7; k++) button('Cand' + k, '', 344 + (k - 1) * 92, 482, 84, 32, 'key', 'ink', 18, null);
button('CandPrev', '上页', 1000, 482, 52, 32, 'btn', 'muted', 13, null);
button('CandNext', '下页', 1056, 482, 52, 32, 'btn', 'muted', 13, null);
button('Plus', '＋ 推荐', 266, 524, 100, 32, 'btn', 'ink', 15, null);
button('Lang', '中', 382, 524, 52, 32, 'btn', 'ink', 15, null);
label('Hint', '', 680, 524, 440, 13, 'muted');
button('Send', '发送', 1036, 524, 76, 34, 'accent', 'onAccent', 16, null);
rect('Composer', 640, COMP.y, COL.x1 - COL.x0, COMP.h, 'surface');
rect('ComposerEdge', 640, COMP.y, COL.x1 - COL.x0 + 2, COMP.h + 2, 'line');
// on-screen keyboard (phones; on PC the letters are also bound to custom keys 1-29, see chat.lua KEYMAP)
const KEY_Y = [588, 632, 676];
['qwertyuiop', 'asdfghjkl', 'zxcvbnm'].forEach((row, r) => {
  const w = 66, gap = 8, x0 = 640 - (row.length * w + (row.length - 1) * gap) / 2 + w / 2;
  [...row].forEach((ch, i) => button('Key' + ch.toUpperCase(), ch, x0 + i * (w + gap), KEY_Y[r], w, 36, 'key', 'ink', 18, null));
});
button('KeyComma', '，', 200, KEY_Y[0], 80, 36, 'key2', 'ink', 18, null);
button('KeyQuestion', '？', 200, KEY_Y[1], 80, 36, 'key2', 'ink', 18, null);
button('KeyExclaim', '！', 200, KEY_Y[2], 80, 36, 'key2', 'ink', 18, null);
button('KeyBack', '删除', 1080, KEY_Y[0], 110, 36, 'key2', 'ink', 16, null);
button('KeySpace', '空格选字', 1080, KEY_Y[1], 110, 36, 'key2', 'ink', 16, null);
button('KeyClear', '清空', 1080, KEY_Y[2], 110, 36, 'key2', 'ink', 16, null);
// full-screen page (3x the canvas: the root is letterboxed on 16:10 / 21:9 screens)
rect('Background', 640, 360, 3840, 2160, 'page');

// ---------------------------------------------------------------- scripts
function layerGroups(node, p, out) { const kids = node.children || []; if (kids.length > 1) out.push([p, kids.map(c => c.name)]); for (const c of kids) layerGroups(c, p === '' ? c.name : p + '/' + c.name, out); return out; }
const lua = v => JSON.stringify(v);
// same limit as the client sandbox ("table.concat result too large"), so local tests catch it too.
// The real sandbox's libraries are read-only ("attempt to modify a readonly table"): pcall, and
// there the platform enforces the limit itself.
const CONCAT_GUARD = "pcall(function() local c = table.concat; table.concat = function(t, ...) local r = c(t, ...); if #r > 49152 then error('table.concat result too large (' .. #r .. ' bytes)') end; return r end end)\n";
const header = mode =>
  CONCAT_GUARD + `local MODE = '${mode}'\n` +
  `local SUGGEST = {${SUGGEST.map(lua).join(',')}}\n` +
  `local THEMES = {${Object.entries(THEMES).map(([t, p]) => `${t}={${Object.entries(p).map(([k, v]) => `${k}=${v}`).join(',')}}`).join(',')}}\n` +
  `local THEME_ROLES = {${ROLES.map(([p, f, r]) => `{${lua(p)},'${f}','${r}'}`).join(',')}}\n` +
  'local LAYERS = {' + layerGroups(root, '', []).map(([p, names]) => '{' + lua(p) + ',{' + names.map(lua).join(',') + '}}').join(',') + '}\n';
const read = p => fs.readFileSync(path.join(ROOT, p), 'utf8');
function modules(list) {          // modules are local functions inside the one script
  let s = 'local __m = {}\n';
  for (const [name, file] of list) s += `__m['${name}'] = (function()\n${read(file)}\nend)()\n`;
  return s;
}
function clientSource(mode) {
  const mods = [['IME', 'game/ime.lua'], ['IMEDICT', 'game/build/ime_dict.lua'], ['FILTER', 'game/build/filter.lua']];
// retrieval snippets only with LORE=1: the model must be trained on the "（资料：…）" format first
if (process.env.LORE === '1') mods.push(['LORE', 'game/build/lore.lua']);   // game/filter_words.lua is NOT bundled (hashes are in filter.lua)
  if (mode === 'full') mods.push(['BPE', 'lua/bpe.lua'], ['MM', 'lua/minimind.lua'], ['TOKT', 'build/mm_tok.lua'], ['W', 'build/mm_weights.lua']);
  return header(mode) + modules(mods) + read('game/chat.lua');
}
const bootSource = `local TEMPLATE_IDS = {${TEMPLATE_ID}}
function OnStart()
  local ok, p = pcall(function() return script:GetParam('templateId') end)
  if ok and type(p) == 'number' and p > 0 then table.insert(TEMPLATE_IDS, 1, p) end
  local ui, id
  for _, candidate in ipairs(TEMPLATE_IDS) do
    id = candidate; ui = game.InstantiateClientUIControl(id, script.object)
    if ui then break end
  end
  print('[paimon] instantiate template', id, ui ~= nil)
end
`;
const host = place(createNode('container', { id: 'paimon-host', name: 'PaimonHost', showCursor: true }), 640, 360, 1280, 720);
const report = {};
for (const mode of FULL ? ['ui', 'full'] : ['ui']) {
  const name = mode === 'full' ? 'paimon-chat-full' : 'paimon-chat-ui';
  const seed = { format: 'qxqy-simulator-save', version: 3, meta: { name: 'TeyvatAI 派蒙' }, activeAssetType: 'server-control-template', serverLogic: { rules: [] },
    assets: {
      server: { version: 3, meta: { name: 'TeyvatAI 派蒙 · 宿主', assetType: 'server-control-template' }, canvasId: 'mobile-16-9', root: createNode('server-container', { id: 'server-root', name: 'PaimonServerUI', children: [structuredClone(host)] }), selectedId: 'paimon-host' },
      client: { version: 1, meta: { name: 'TeyvatAI 派蒙界面', assetType: 'client-control-template', sourceFormat: 'authoring', sourceFile: '', gameVersion: '7.0.50', giaOwnerUid: 114514, giaFileId: 1073741829, giaFileName: '客户端控件模板列表.gia' }, canvasId: 'mobile-16-9', root: createNode('server-container', { id: 'client-root', name: '客户端控件模板', children: [structuredClone(root)] }), selectedId: 'paimon-root' },
    } };
  const studio = createStudio(seed);
  const src = clientSource(mode === 'full' ? 'full' : 'mock');
  // plain copies of the scripts, for overwriting the editor's mapped files after an import
  fs.mkdirSync(path.join(OUT, 'scripts'), { recursive: true });
  fs.writeFileSync(path.join(OUT, 'scripts', name + '.main.lua'), src);
  fs.writeFileSync(path.join(OUT, 'scripts', 'boot.lua'), bootSource);
  studio.patch({ op: 'addScript', controlId: 'paimon-root', controlAsset: 'client-control-template', path: 'paimon/main', source: src });
  studio.patch({ op: 'addScript', controlId: 'paimon-host', controlAsset: 'server-control-template', path: 'paimon/boot', source: bootSource });
  studio.patch({ op: 'renameSave', name: 'TeyvatAI 派蒙 · ' + (mode === 'full' ? '完整模型' : '界面测试') });
  const save = studio.exportData('save'); fs.writeFileSync(path.join(OUT, name + '.save.json'), Buffer.from(save.data, 'base64'));
  try { const gia = studio.exportData('gia-combined'); fs.writeFileSync(path.join(OUT, name + '.gia'), Buffer.from(gia.data, 'base64')); report[name] = { ...Object.fromEntries(Object.entries(gia).filter(([k]) => k !== 'data')), scriptBytes: Buffer.byteLength(src) }; }
  catch (e) { report[name] = { error: e.message, scriptBytes: Buffer.byteLength(src) }; }
  console.log(name, 'script', (Buffer.byteLength(src) / 1e6).toFixed(2) + 'MB');
}
if (process.argv.includes('--install')) {
  const dir = process.env.LOCALAPPDATA ? process.env.LOCALAPPDATA.replace(/Local$/, 'LocalLow') + '/miHoYo/原神/BeyondLocal/Beyond_Local_Export' : null;
  if (dir && fs.existsSync(dir)) for (const f of fs.readdirSync(OUT).filter(f => f.endsWith('.gia'))) { fs.copyFileSync(path.join(OUT, f), path.join(dir, f)); console.log('installed', f, '->', dir); }
  else console.log('export folder not found:', dir);
}
fs.writeFileSync(path.join(OUT, 'export-report.json'), JSON.stringify(report, null, 2));
console.log('controls', root.children.length, 'theme roles', ROLES.length, '->', OUT);
