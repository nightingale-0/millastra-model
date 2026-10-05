// Ids (varints in 1073741824..1073900000) near control names in a level save: node game/gil_peek.mjs <file.gil>
import fs from 'node:fs';
const buf = fs.readFileSync(process.argv[2]);
const txt = buf.toString('latin1');
function varintAt(i) { let x = 0, s = 0; for (let k = 0; k < 6; k++) { const c = buf[i + k]; x += (c & 127) * 2 ** s; s += 7; if (c < 128) return [x, k + 1]; } return [null, 0]; }
for (const name of (process.argv[3] || 'PaimonRoot,PaimonHost,PaimonServerUI,paimon/main,paimon/boot').split(',')) {
  let at = -1;
  while ((at = txt.indexOf(name, at + 1)) >= 0) {
    const ids = [];
    for (let i = Math.max(0, at - 400); i < Math.min(buf.length, at + 200); i++) {
      const [v, n] = varintAt(i);
      if (v !== null && v >= 1073741824 && v < 1073900000 && n === 5) ids.push(`${v}@${i - at}`);
    }
    console.log(name.padEnd(14), 'offset', at, 'ids', [...new Set(ids)].join(' '));
  }
}
