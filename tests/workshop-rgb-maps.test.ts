// Colour maps of the tool against utils/maps.py of the presentation, and the offline rule of WEB_SPEC §11
// (no dynamic import and no fetch in src/lib/workshop-rgb): run with
// `node --experimental-strip-types tests/workshop-rgb-maps.test.ts`
import { readFileSync, readdirSync } from 'node:fs';
import { table, hex, divHex, index, colorize, rgb } from '../src/lib/workshop-rgb/colormaps.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const fx = (p: string) => new URL(`./fixtures/workshop-rgb/${p}`, import.meta.url);
const exp = JSON.parse(readFileSync(fx('maps.json'), 'utf8'));
const look = JSON.parse(readFileSync(fx('maps_ref.json'), 'utf8'));

// 1. 256 entries of every table equal the export of utils/maps.py (within 1/255, here exact)
const pairs: [string, string[]][] = [
  ['t', table('t')], ['fs_dark', table('fs', 'dark')], ['fs_sala', table('fs', 'light')],
  ['ch_R_dark', table('ch_R', 'dark')], ['ch_G_dark', table('ch_G', 'dark')], ['ch_B_dark', table('ch_B', 'dark')],
  ['ch_R_sala', table('ch_R', 'light')], ['ch_G_sala', table('ch_G', 'light')], ['ch_B_sala', table('ch_B', 'light')],
  ['div_dark', table('div', 'dark')], ['div_sala', table('div', 'light')],
];
for (const [k, t] of pairs) {
  let m = 0;
  for (let i = 0; i < 256; i++) { const a = rgb(t[i]), b = rgb(exp[k][i]); for (let c = 0; c < 3; c++) m = Math.max(m, Math.abs(a[c] - b[c])); }
  check(`${k}, 256 entries equal utils/maps.py`, t.length === 256 && m <= 1, `(max ${m}/255)`);
}
check('f_s more soot is more ink, black to white on dark and white to black on light', table('fs', 'dark')[0] === '#000000' && table('fs', 'dark')[255] === '#FFFFFF' && table('fs', 'light')[0] === '#FFFFFF');

// 2. lookups, including Python's round half to even
const v: number[] = look.values;
check('viridis lookup as t_hex', v.every((x, i) => hex('t', x) === look.t[i]));
check('grey lookup as fs_hex on both themes', v.every((x, i) => hex('fs', x, 0, 1, 'dark') === look.fs_dark[i] && hex('fs', x, 0, 1, 'light') === look.fs_sala[i]));
check('channel ramps as ch_hex', v.every((x, i) => hex('ch_R', x, 0, 1, 'dark') === look.ch_R_dark[i] && hex('ch_B', x, 0, 1, 'light') === look.ch_B_sala[i]));
check('diverging lookup as div_hex', v.every((x, i) => divHex(x - 0.5, 0.5, 'dark') === look.div_dark[i]));
check('T on 1400 to 2100 K, clipped outside', look.t_K_in.every((T: number, i: number) => hex('t', T, 1400, 2100) === look.t_K[i]));
check('index 127,5 rounds to 128 and 0,5 to 0, as Python', index(127.5 / 255, 0, 1) === 128 && index(0.5 / 255, 0, 1) === 0 && index(1.5 / 255, 0, 1) === 2);

// 3. masked or NaN pixels in the background colour, never as zero
{
  const img = colorize([0, NaN, 1, 0.5], 'fs', 0, 1, 'dark', '#202124', [1, 1, 0, 1]);
  check('masked and NaN pixels take the page background', img[4] === 0x20 && img[8] === 0x20 && img[0] === 0 && img[3] === 255);
}

// 4. offline: no dynamic import and no fetch in the tool's numerics (WEB_SPEC §11)
{
  const dir = new URL('../src/lib/workshop-rgb/', import.meta.url);
  const bad: string[] = [];
  for (const f of readdirSync(dir)) {
    if (!f.endsWith('.ts')) continue;
    const s = readFileSync(new URL(f, dir), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
    if (/\bimport\s*\(/.test(s) || /\bfetch\s*\(/.test(s)) bad.push(f);
  }
  check('no import( or fetch( in src/lib/workshop-rgb', bad.length === 0, bad.join(', '));
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
