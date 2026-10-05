// Render the cover -> game/cover/cover.png (1920x1080) with @napi-rs/canvas (from the simulator):
// cover.svg holds the graphics, TEXTS below are drawn with the Windows CJK fonts.
import fs from 'node:fs'; import path from 'node:path'; import { createRequire } from 'node:module'; import { pathToFileURL } from 'node:url';
const SIM = process.env.QXQY_SIM || path.resolve('../6nimmt/work/simulator');
const req = createRequire(pathToFileURL(path.join(SIM, 'package.json')));
const { createCanvas, loadImage, GlobalFonts } = req('@napi-rs/canvas');
GlobalFonts.registerFromPath('C:/Windows/Fonts/msyh.ttc', 'YaHei');
GlobalFonts.registerFromPath('C:/Windows/Fonts/msyhbd.ttc', 'YaHeiBold');
GlobalFonts.registerFromPath('C:/Windows/Fonts/consola.ttf', 'Mono');
const MUTED = 'rgba(138,138,132,1)', INK = '#1f1f1d', ACCENT = '#d97757';
// [text, x, y, size, font, colour, align, alpha, letterSpacing]
const TEXTS = [
  ['千星奇域 · 离线 AI', 140, 340, 52, 'YaHei', MUTED, 'left', 1, 6],
  ['TeyvatAI', 132, 490, 150, 'YaHeiBold', INK],
  ['派蒙', 136, 658, 128, 'YaHeiBold', ACCENT],
  ['在游戏里本地运行的派蒙聊天模型', 142, 748, 38, 'YaHei', '#4a4a46'],
  ['2600 万参数', 285, 838, 34, 'YaHeiBold', '#ffffff', 'center'],
  ['交给派蒙吧！', 1715, 282, 36, 'YaHeiBold', INK, 'center'],
  ['0110 1001 0011', 640, 80, 26, 'Mono', MUTED, 'left', 0.16],
  ['1010 0101 1100 0110', 80, 1020, 26, 'Mono', MUTED, 'left', 0.16],
  ['0011 1011', 1600, 560, 26, 'Mono', MUTED, 'left', 0.16],
];
const img = await loadImage(fs.readFileSync('game/cover/cover.svg'));
const c = createCanvas(1920, 1080); const g = c.getContext('2d');
g.drawImage(img, 0, 0, 1920, 1080);
// Paimon sticker (cut out by cutout.py); its flat bottom edge sits on the cover's bottom edge
// smaller and centred on the right half; the cut-off bottom edge of the sticker fades out
const PAIMON = { file: 'game/cover/paimon.png', cx: 1370, top: 200, size: 700, fade: 0.18 };
if (fs.existsSync(PAIMON.file)) {
  const p = await loadImage(fs.readFileSync(PAIMON.file));
  const off = createCanvas(PAIMON.size, PAIMON.size), o = off.getContext('2d');
  o.drawImage(p, 0, 0, PAIMON.size, PAIMON.size);
  const grad = o.createLinearGradient(0, PAIMON.size * (1 - PAIMON.fade), 0, PAIMON.size);
  grad.addColorStop(0, 'rgba(0,0,0,1)'); grad.addColorStop(1, 'rgba(0,0,0,0)');
  o.globalCompositeOperation = 'destination-in'; o.fillStyle = grad;
  o.fillRect(0, 0, PAIMON.size, PAIMON.size);
  g.drawImage(off, PAIMON.cx - PAIMON.size / 2, PAIMON.top);
}
for (const [text, x, y, size, font, fill, align = 'left', alpha = 1, ls = 0] of TEXTS) {
  g.font = `${size}px ${font}`; g.fillStyle = fill; g.textAlign = align; g.globalAlpha = alpha; g.letterSpacing = ls + 'px';
  g.fillText(text, x, y);
}
fs.writeFileSync('game/cover/cover.png', c.toBuffer('image/png'));
console.log('game/cover/cover.png');
