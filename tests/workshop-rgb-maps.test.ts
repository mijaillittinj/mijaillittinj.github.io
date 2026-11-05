// Colour maps of the tool against utils/maps.py of the presentation, and the offline rule of WEB_SPEC §11
// (no dynamic import and no fetch in src/lib/workshop-rgb): run with
// `node --experimental-strip-types tests/workshop-rgb-maps.test.ts`
import { readFileSync, readdirSync } from 'node:fs';
import { table, hex, divHex, index, colorize, contourf, rgb } from '../src/lib/workshop-rgb/colormaps.ts';
import { rStretch, niceTicks } from '../src/components/workshop-rgb/draw.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
// the factor of the r axis is round, says what the box really does, and leaves a shape that is already wide as it is
{
  const ex = 17.2 / 72;
  check('r factor of the example flame, 3', rStretch(ex) === 3, String(rStretch(ex)));
  check('r factor of a wide shape, 1', rStretch(1.2) === 1 && rStretch(0.6) === 1, `${rStretch(1.2)} ${rStretch(0.6)}`);
  check('r factor of a narrow crop grows', rStretch(6.6 / 72) === 8, String(rStretch(6.6 / 72)));
  check('r factor between 1 and 3 goes in halves', rStretch(0.35) === 2 && rStretch(0.3) === 2.5, `${rStretch(0.35)} ${rStretch(0.3)}`);
  check('r factor never shrinks', [0.01, 0.1, 0.3, 0.56, 0.7, 1, 3, 40].every((a) => rStretch(a) >= 1));
  check('r factor with an empty shape, 1', rStretch(NaN) === 1 && rStretch(Infinity) === 1);
  // the ticks stay symmetric and inside, whichever the box
  const t = niceTicks(-8.6, 8.6, 4);
  check('r ticks are symmetric', t.length === 3 && t[0] === -5 && t[2] === 5, t.join());
}
const fx = (p: string) => new URL(`./fixtures/workshop-rgb/${p}`, import.meta.url);
const exp = JSON.parse(readFileSync(fx('maps.json'), 'utf8'));
const look = JSON.parse(readFileSync(fx('maps_ref.json'), 'utf8'));


// filled contours: bands, values between the cell centres, nothing drawn past a missing corner
{
  const px = (a: Uint8ClampedArray, W: number, x: number, y: number) => [a[4 * (y * W + x)], a[4 * (y * W + x) + 1], a[4 * (y * W + x) + 2]].join();
  const bg = '#102030', bgs = rgb(bg).join();
  const ramp = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], s = 8, W = 4 * s, H = 3 * s;
  const a = contourf(ramp, 4, 3, s, 't', 0, 3, 3, 'dark', bg);
  const band = (b: number) => rgb(table('t')[Math.round(((b + 0.5) / 3) * 255)]).join();
  check('contourf, half a cell around the centres stays blank', px(a, W, 0, 12) === bgs && px(a, W, W - 1, 12) === bgs && px(a, W, 16, 0) === bgs && px(a, W, 16, H - 1) === bgs);
  check('contourf, three bands from left to right', px(a, W, 8, 12) === band(0) && px(a, W, 16, 12) === band(1) && px(a, W, 24, 12) === band(2));
  check('contourf, only band colours and the background', new Set(Array.from({ length: W * H }, (_, q) => px(a, W, q % W, Math.floor(q / W)))).size === 4);
  const hole = ramp.slice(); hole[5] = NaN;
  const b = contourf(hole, 4, 3, s, 't', 0, 3, 3, 'dark', bg);
  check('contourf, a NaN blanks the four squares that touch it', px(b, W, 8, 8) === bgs && px(b, W, 16, 16) === bgs && px(b, W, 24, 8) !== bgs);
  const c = contourf(ramp, 4, 3, s, 't', 0, 3, 3, 'dark', bg, ramp.map((_, q) => (q === 5 ? 0 : 1)));
  check('contourf, a mask acts as a NaN', c.join() === b.join());
  const d = contourf(ramp.map((v) => v * 100 - 100), 4, 3, s, 't', 0, 3, 3, 'dark', bg);
  check('contourf, values past the range take the end bands', px(d, W, 4, 12) === band(0) && px(d, W, 27, 12) === band(2));
}

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
