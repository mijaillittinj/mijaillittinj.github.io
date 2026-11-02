// Saturation, hot pixels, 3 × 3 dilation, pylon 2 × 2 replication phase and Bayer superpixel:
// run with `node --experimental-strip-types tests/workshop-rgb-ceiling.test.ts`
// References in tests/fixtures/workshop-rgb/ceiling_ref.json (numpy and scipy.ndimage, make_web_fixtures.py).
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { readTiff, detectEncoding } from '../src/lib/workshop-rgb/tiff.ts';
import {
  saturationMask, countPerChannel, dilate3x3, hotPixelsFromDark, hotPixelsNoDark, replicationPhase,
  bayerSuperpixel, bin2x2, pylonSites, rgbShiftTest, Accumulator, replaceHot, toDn12,
} from '../src/lib/workshop-rgb/frames.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const fx = (p: string) => new URL(`./fixtures/workshop-rgb/${p}`, import.meta.url);
const ref = JSON.parse(readFileSync(fx('ceiling_ref.json'), 'utf8'));
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);

// 1. sphere dark around its first saturated pixel: saturated in raw counts against the detected ceiling
{
  const r = ref.sphere_dark;
  const img = readTiff(readFileSync(fx(`tiff/${r.file}`)));
  const e = detectEncoding(img);
  const planes = { width: img.width, height: img.height, nc: 3, data: img.data };
  const sat = saturationMask(planes, e.ceiling);
  check('ceiling 65520 detected on the sphere dark', e.ceiling === 65520 && e.encoding === 'u12_shifted');
  check('saturated per channel equals numpy', countPerChannel(sat, 3).join(',') === r.sat.join(','), `(${countPerChannel(sat, 3)})`);
  const dil = dilate3x3(sat, img.width, img.height, 3, true);
  let nd = 0;
  for (let p = 0; p < img.width * img.height; p++) if (dil[3 * p]) nd++;
  check('3 × 3 dilation across channels equals scipy binary_dilation', nd === r.dilated_pixels, `(${nd} pixels)`);
  const dn = { width: img.width, height: img.height, nc: 3, data: toDn12(img.data, e.toDn12) };
  const hot = hotPixelsFromDark(dn, e.ceiling * e.toDn12, 10);
  const ok = ['R', 'G', 'B'].every((c, k) => rel(hot.med[k], r.hot[c].med) < 1e-12 && rel(hot.sigma[k], r.hot[c].sigma) < 1e-12 && hot.count[k] === r.hot[c].count);
  check('hot pixels from the dark in 12 bit DN (median + 10 σ, or saturated) equal the numpy mirror', ok, `(${hot.count})`);
}

// 2. replication phase of the pylon RGB on the oven crop
{
  const r = ref.oven_phase;
  const img = readTiff(readFileSync(fx(`tiff/${r.file}`)));
  const ph = replicationPhase({ width: img.width, height: img.height, nc: 3, data: img.data });
  const ok = ph.every((p, c) => p.phase.join(',') === r.channels[c].phase.join(',') && Math.abs(p.frac - r.channels[c].frac) < 1e-12);
  check('2 × 2 replication phase per channel equals numpy', ok, `(R ${ph[0].phase} ${ph[0].frac.toFixed(3)}, G ${ph[1].frac.toFixed(3)}, B ${ph[2].phase} ${ph[2].frac.toFixed(3)})`);
  check('pylon RGB of the oven, R and B replicated, G interpolated', ph[0].replicated && ph[2].replicated && !ph[1].replicated);
}

// 3. Bayer superpixel RGGB
{
  const r = ref.bayer_superpixel;
  const img = readTiff(readFileSync(fx(`tiff/${r.file}`)));
  const sm = saturationMask({ width: img.width, height: img.height, nc: 1, data: img.data }, 65520);
  const sp = bayerSuperpixel({ width: img.width, height: img.height, data: img.data }, 'RGGB', sm);
  const sums = [0, 1, 2].map((c) => { let s = 0; for (let i = c; i < sp.data.length; i += 3) s += sp.data[i]; return s; });
  const sha = createHash('sha256').update(new Uint8Array(sp.data.buffer)).digest('hex');
  check('superpixel equals numpy (sum, SHA-256 of float32)', sums.every((s, c) => s === r.sum[c]) && sha === r.sha_f32 && sp.width === r.width && sp.height === r.height);
  let ns = 0;
  for (let p = 0; p < sp.width * sp.height; p++) if (sp.sat[3 * p]) ns++;
  check('superpixel saturated if any of its four sites is', ns === r.sat, `(${ns})`);
  // centred sites (GB3r2-adv-03), a linear ramp in the mosaic gives R = G = B at every inner superpixel
  {
    const w = 16, h = 12, ramp = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) ramp[y * w + x] = 3 * x + 7 * y;
    const c = bayerSuperpixel({ width: w, height: h, data: ramp }, 'RGGB', undefined, true);
    let worst = 0;
    for (let y = 1; y < c.height - 1; y++) for (let x = 1; x < c.width - 1; x++) { const o = (y * c.width + x) * 3; worst = Math.max(worst, Math.abs(c.data[o] - c.data[o + 1]), Math.abs(c.data[o + 2] - c.data[o + 1])); }
    const u = bayerSuperpixel({ width: w, height: h, data: ramp }, 'RGGB');
    const off = Math.abs(u.data[3 * 13] - u.data[3 * 13 + 1]);
    check('centred Bayer sites put R and B on the green centre (ramp, inner superpixels)', worst < 1e-4 && off > 1, `(|R − G|, |B − G| ≤ ${worst.toExponential(1)}, uncentred ${off})`);
  }
  const pv = r.probe.map(([y, x]: number[]) => [0, 1, 2].map((c) => sp.data[(y * sp.width + x) * 3 + c]));
  check('superpixel probe values', JSON.stringify(pv) === JSON.stringify(r.probe_vals));
  check('superpixel doubles mm per pixel', sp.scale === 2);
}

// 4. accumulator: Float32 sum of integer counts, exact removal of a frame, saturation OR (WEB_SPEC §7.3)
{
  const W = 32, H = 24, n = W * H * 3;
  const frames = Array.from({ length: 10 }, (_, k) => Uint16Array.from({ length: n }, (_, i) => (((i * 7919 + k * 104729) % 4096) * 16) >>> 0));
  const acc = new Accumulator(W, H, 3);
  for (const f of frames) acc.add(f, saturationMask({ width: W, height: H, nc: 3, data: f }, 65520));
  const before = acc.sum.slice();
  acc.add(frames[3]); acc.remove(frames[3]);
  let same = true;
  for (let i = 0; i < n; i++) if (acc.sum[i] !== before[i]) same = false;
  let exact = true;
  for (let i = 0; i < n; i++) { let s = 0; for (const f of frames) s += f[i]; if (acc.sum[i] !== s) exact = false; }
  check('Float32 sum of 10 frames of 65520 is exact and a removal is exact', same && exact);
  const sat = acc.satMask();
  let anySat = 0, truth = 0;
  for (let i = 0; i < n; i++) { anySat += sat[i]; if (frames.some((f) => f[i] >= 0.995 * 65520)) truth++; }
  check('a pixel saturated in any frame stays marked', anySat === truth && truth > 0, `(${truth})`);
  const dk = new Uint16Array(n).fill(96);
  const m = acc.meanMinus(dk, 1 / 16);
  let ok = true;
  for (let i = 0; i < n; i += 97) { let s = 0; for (const f of frames) s += f[i]; if (Math.abs(m[i] - (s / 10 - 96) / 16) > 1e-3) ok = false; }
  check('mean minus dark in 12 bit DN', ok);
}

// 5. hot pixels without a dark, replacement by the neighbour median, 2 × 2 grouping
{
  const W = 30, H = 20, nc = 3;
  const mk = (gain: number) => {
    const d = new Float64Array(W * H * nc);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < nc; c++) d[(y * W + x) * nc + c] = gain * (100 + 3 * x + 2 * y + c) + ((x * 31 + y * 17 + c) % 5);
    d[(7 * W + 11) * nc + 1] = 4000; // hot in every frame
    return d;
  };
  const fr = [mk(1), mk(2), mk(3)].map((data) => ({ width: W, height: H, nc, data }));
  fr[0].data[(3 * W + 4) * nc] = 3000; // a single bright event only in the shortest exposure
  const hot = hotPixelsNoDark(fr, 5);
  let n = 0;
  for (let i = 0; i < hot.length; i++) n += hot[i];
  check('hot without a dark: only the pixel that stays out in every exposure', n === 1 && hot[(7 * W + 11) * nc + 1] === 1, `(${n})`);
  const img = { width: W, height: H, nc, data: Float64Array.from(fr[1].data) };
  replaceHot(img, hot);
  const v = img.data[(7 * W + 11) * nc + 1];
  check('hot pixel replaced by the median of its neighbours', Math.abs(v - (2 * (100 + 33 + 14 + 1))) < 8, `(${v})`);
  const b = bin2x2({ width: W, height: H, nc, data: fr[1].data });
  check('2 × 2 grouping halves the size and doubles mm per pixel', b.width === 15 && b.height === 10 && b.scale === 2);
  const dn = toDn12(Uint16Array.of(16, 65520), 1 / 16);
  check('raw counts to 12 bit DN', dn[0] === 1 && dn[1] === 4095);
}

// 6. pylon RGB grouped 2 × 2 with R and B centred like the Bayer path (decision after G4 r3, G4r3-conf-24).
// RGGB mosaic with a linear ramp, pylon replication of R in phase (1, 1) and of B in phase (0, 0) (each
// site copied over its block, as in the oven photographs), green interpolated. With the centring every
// inner group has R = G = B, without it R and B sit ¼ group off.
{
  const W = 40, H = 32, nc = 3;
  const ramp = (y: number, x: number) => 1000 + 7 * y + 3 * x;
  const data = new Float64Array(W * H * nc);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const ry = 2 * Math.floor((y + 1) / 2), rx = 2 * Math.floor((x + 1) / 2); // R site of the (1, 1) block
    const by = 2 * Math.floor(y / 2) + 1, bx = 2 * Math.floor(x / 2) + 1; // B site of the (0, 0) block
    const o = (y * W + x) * nc;
    data[o] = ramp(ry, rx); data[o + 1] = ramp(y, x); data[o + 2] = ramp(by, bx);
  }
  const img = { width: W, height: H, nc, data };
  const ph = replicationPhase(img);
  const sites = pylonSites('RGGB', ph[0].phase, ph[2].phase);
  check('pylon phases and sites, R (1, 1) and B (0, 0), both sites in the bottom right half', ph[0].replicated && ph[2].replicated && ph[0].phase.join() === '1,1' && ph[2].phase.join() === '0,0' && sites.R!.join() === '1,1' && sites.B!.join() === '1,1');
  check('channel without replication is not moved', pylonSites('RGGB', null, [0, 0]).R === null);
  const c = bin2x2(img, undefined, 1, 1, sites);
  const u = bin2x2(img, undefined, 1, 1);
  let worst = 0, off = 0;
  for (let y = 1; y < c.height - 1; y++) for (let x = 1; x < c.width - 1; x++) {
    const o = (y * c.width + x) * 3;
    worst = Math.max(worst, Math.abs(c.data[o] - c.data[o + 1]), Math.abs(c.data[o + 2] - c.data[o + 1]));
    off = Math.max(off, Math.abs(u.data[o] - u.data[o + 1]));
  }
  check('pylon RGB grouped 2 × 2 puts R and B on the green centre (ramp, inner groups)', worst < 1e-3 && off > 1, `(|R − G|, |B − G| ≤ ${worst.toExponential(1)}, uncentred ${off})`);
}

// 7. Shift test of R and B on a ruler-like image (decision on closing G4, the centring is off by default and the
// ruler says whether to turn it on). Blurred bars in both directions, pylon replication as in 6 with a little noise.
// Sites in the bottom right half give a shift of −½ sensor pixel (−¼ group) on each axis, k ≈ 1 and `activate`. The
// same image read with the other pattern gives k ≈ −1 and `keep_off`, and R and B without a shift give k ≈ 0.
{
  const W = 400, H = 320, nc = 3;
  const edge = (t: number) => 0.5 + 0.5 * Math.tanh(t / 2.5);
  const bars = (u: number, per: number, wid: number) => { const m = ((u % per) + per) % per; return edge(m - 2) - edge(m - 2 - wid); };
  const scene = (y: number, x: number) => 300 + 1500 * bars(x, 17, 6) + 1000 * bars(y, 23, 8);
  let seed = 7;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
  const make = (shifted: boolean) => {
    const data = new Float64Array(W * H * nc);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const ry = 2 * Math.floor((y + 1) / 2), rx = 2 * Math.floor((x + 1) / 2);
      const by = 2 * Math.floor(y / 2) + 1, bx = 2 * Math.floor(x / 2) + 1;
      const o = (y * W + x) * nc;
      // without a shift the replicated value is the scene at the centre of its block
      data[o] = shifted ? scene(ry, rx) : scene(ry - 0.5, rx - 0.5);
      data[o + 1] = scene(y, x) + 4 * rnd();
      data[o + 2] = shifted ? scene(by, bx) : scene(by - 0.5, bx - 0.5);
    }
    // replicated channels get one noise value per block, as pylon copies one site
    for (let y = 1; y + 1 < H; y += 2) for (let x = 1; x + 1 < W; x += 2) { const n = 4 * rnd(); for (const [dy, dx] of [[0, 0], [0, 1], [1, 0], [1, 1]]) data[((y + dy) * W + x + dx) * nc] += n; }
    for (let y = 0; y + 1 < H; y += 2) for (let x = 0; x + 1 < W; x += 2) { const n = 4 * rnd(); for (const [dy, dx] of [[0, 0], [0, 1], [1, 0], [1, 1]]) data[((y + dy) * W + x + dx) * nc + 2] += n; }
    return { width: W, height: H, nc, data };
  };
  const img = make(true);
  const ph = replicationPhase(img);
  const g = bin2x2(img, undefined, ph[0].phase[0], ph[0].phase[1]);
  const fr = { w: g.width, h: g.height, dn: g.data, sat: g.sat };
  const a = rgbShiftTest(fr, pylonSites('RGGB', ph[0].phase, ph[2].phase));
  check('shift test, sites in the bottom right half measured at −¼ group on both axes, k ≈ 1, activate', a.verdict === 'activate' && Math.abs(a.k - 1) < 0.15 && a.axes[0] && a.axes[1] && Math.abs(a.meas.R![0] + 0.25) < 0.05 && Math.abs(a.meas.B![1] + 0.25) < 0.05, `(k ${a.k.toFixed(3)} ± ${a.kUnc.toFixed(3)}, R ${a.meas.R!.map((v) => v.toFixed(3))}, B ${a.meas.B!.map((v) => v.toFixed(3))}, ${a.nEdge} px, ${a.tiles} tiles)`);
  const b = rgbShiftTest(fr, pylonSites('BGGR', ph[0].phase, ph[2].phase));
  check('shift test with the other pattern, k ≈ −1, keep off', b.verdict === 'keep_off' && Math.abs(b.k + 1) < 0.15, `(k ${b.k.toFixed(3)})`);
  const u = make(false), phu = replicationPhase(u);
  const gu = bin2x2(u, undefined, phu[0].phase[0], phu[0].phase[1]);
  const c = rgbShiftTest({ w: gu.width, h: gu.height, dn: gu.data, sat: gu.sat }, pylonSites('RGGB', phu[0].phase, phu[2].phase));
  check('shift test without a shift, k ≈ 0, keep off', c.verdict === 'keep_off' && Math.abs(c.k) < 0.15, `(k ${c.k.toFixed(3)})`);
  const d = rgbShiftTest(fr, { R: null, B: null });
  check('shift test without replication does not apply', d.verdict === 'not_applicable');
  const flat = new Float32Array(fr.dn.length).fill(1000);
  const e = rgbShiftTest({ ...fr, dn: flat }, pylonSites('RGGB', ph[0].phase, ph[2].phase));
  check('shift test on an image without edges is inconclusive', e.verdict === 'inconclusive' && e.why === 'few_edges');
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
