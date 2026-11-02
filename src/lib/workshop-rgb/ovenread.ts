/**
 * Reader of the oven images for the IR tail calibration, the same as steps 2 to 4 of manim/tools/calibration.py
 * (parity after G5, decision of the director for the closing: the tool and the talk give the same tail figures
 * on the real oven, and calibration.py rules).
 *
 *   2 aperture  G smoothed by a 9 px box (scipy uniform_filter, mode reflect), threshold halfway between the
 *               median of the smoothed frame and its 99,9 percentile, largest 4-connected component with its
 *               holes filled; centre is its centroid, eq. diameter 2 √(area/π)
 *   3 plateau   level = median of the smoothed G within 15 px of the centre; radius = the largest r from 10 px
 *               whose ring [r − 1, r) has 98 % of its pixels within ±3 % of that level; the disc radius is the
 *               smallest over the temperatures, centred on each one
 *   4 signal    mean of every pixel of the disc per channel (no hot pixel replacement, no flags), and the count
 *               of pixels of that channel at the 12 bit ceiling (saturated per channel, not across channels)
 *
 * The reference frame of a temperature is its longest exposure with no pixel at the ceiling in any channel.
 * The frames are read at the sensor grid (pylon RGB not grouped, Bayer as 2 × 2 superpixels, then the pixel
 * lengths are divided by the scale). What the tool did before (median of the 2 × 2 working grid over the
 * plateau of 1a.3, saturation dilated across channels) moved s_R from 0,1165 to 0,127 on the 51 images:
 * the dilated saturation of R took G out of the dark intercept at 800, 1000 and 1050 °C. See WEB_REPORT.md.
 *
 * Images are chunky (H, W, 3) in 12 bit DN. Pixel lengths in px of the frame.
 */
import { median, percentile } from './stats.ts';
import type { Rgb } from './radiometry.ts';

export const READ_SMOOTH = 9;
export const READ_TOL = 0.03;
export const READ_CORE = 15;
export const READ_RING_FROM = 10;
export const READ_RING_OK = 0.98;

export interface ReadFrame { w: number; h: number; dn: ArrayLike<number> }
export interface ReadGeometry { cy: number; cx: number; area: number; eqDiam: number; plateauR: number; level: number; scale: number }
export interface ReadSignal { tC: number; texpUs: number; S: Rgb<number>; nSat: Rgb<number>; n: number }

/** scipy.ndimage.uniform_filter1d, mode 'reflect' (d c b a | a b c d | d c b a), on rows (axis 1) or columns (axis 0). */
function box1d(src: Float64Array, w: number, h: number, size: number, axis: 0 | 1): Float64Array {
  const out = new Float64Array(src.length);
  const lo = size >> 1, hi = size - lo - 1;
  const n = axis === 1 ? w : h, m = axis === 1 ? h : w;
  const refl = (i: number) => { const p = 2 * n; i = ((i % p) + p) % p; return i < n ? i : p - 1 - i; };
  const line = new Float64Array(n);
  for (let k = 0; k < m; k++) {
    for (let i = 0; i < n; i++) line[i] = axis === 1 ? src[k * w + i] : src[i * w + k];
    let s = 0;
    for (let j = -lo; j <= hi; j++) s += line[refl(j)];
    for (let i = 0; i < n; i++) {
      const v = s / size;
      if (axis === 1) out[k * w + i] = v; else out[i * w + k] = v;
      s += line[refl(i + hi + 1)] - line[refl(i - lo)];
    }
  }
  return out;
}

/** G (channel 1) smoothed by a size × size box as scipy.ndimage.uniform_filter. */
export function smoothG(f: ReadFrame, size = READ_SMOOTH): Float64Array {
  const g = new Float64Array(f.w * f.h);
  for (let p = 0; p < g.length; p++) g[p] = f.dn[3 * p + 1];
  return box1d(box1d(g, f.w, f.h, size, 0), f.w, f.h, size, 1);
}

/** Largest 4-connected component of m (first one on ties, as argmax over ndimage.label), holes filled as binary_fill_holes. */
function largestFilled(m: Uint8Array, w: number, h: number): Uint8Array {
  const lab = new Int32Array(m.length);
  const stack = new Int32Array(m.length);
  let best = 0, bestN = 0, k = 0;
  for (let p0 = 0; p0 < m.length; p0++) {
    if (!m[p0] || lab[p0]) continue;
    k++;
    let top = 0, n = 0;
    stack[top++] = p0; lab[p0] = k;
    while (top) {
      const p = stack[--top]; n++;
      const y = (p / w) | 0, x = p - y * w;
      if (x > 0 && m[p - 1] && !lab[p - 1]) { lab[p - 1] = k; stack[top++] = p - 1; }
      if (x < w - 1 && m[p + 1] && !lab[p + 1]) { lab[p + 1] = k; stack[top++] = p + 1; }
      if (y > 0 && m[p - w] && !lab[p - w]) { lab[p - w] = k; stack[top++] = p - w; }
      if (y < h - 1 && m[p + w] && !lab[p + w]) { lab[p + w] = k; stack[top++] = p + w; }
    }
    if (n > bestN) { bestN = n; best = k; }
  }
  // background reachable from the border (4-connected) stays background, the rest is a hole
  const out = new Uint8Array(m.length);
  for (let p = 0; p < m.length; p++) out[p] = lab[p] === best ? 1 : 0;
  const reach = new Uint8Array(m.length);
  let top = 0;
  const seed = (p: number) => { if (!out[p] && !reach[p]) { reach[p] = 1; stack[top++] = p; } };
  for (let x = 0; x < w; x++) { seed(x); seed((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { seed(y * w); seed(y * w + w - 1); }
  while (top) {
    const p = stack[--top];
    const y = (p / w) | 0, x = p - y * w;
    if (x > 0) seed(p - 1);
    if (x < w - 1) seed(p + 1);
    if (y > 0) seed(p - w);
    if (y < h - 1) seed(p + w);
  }
  for (let p = 0; p < m.length; p++) if (!reach[p]) out[p] = 1;
  return out;
}

/** Aperture and plateau radius of one temperature on its reference frame. `scale` is sensor px per frame px. */
export function readGeometry(f: ReadFrame, scale = 1): ReadGeometry {
  const { w, h } = f;
  const s = smoothG(f, Math.max(1, Math.round(READ_SMOOTH / scale)));
  const bg = median(s), top = percentile(s, 99.9);
  const thr = bg + 0.5 * (top - bg);
  const m = new Uint8Array(w * h);
  for (let p = 0; p < m.length; p++) if (s[p] > thr) m[p] = 1;
  const mk = largestFilled(m, w, h);
  let sy = 0, sx = 0, area = 0;
  for (let p = 0; p < mk.length; p++) if (mk[p]) { sy += (p / w) | 0; sx += p % w; area++; }
  if (!area) throw new Error('oven_no_aperture');
  const cy = sy / area, cx = sx / area, eqDiam = 2 * Math.sqrt(area / Math.PI);
  const core: number[] = [];
  const rc = READ_CORE / scale;
  for (let y = Math.max(0, Math.floor(cy - rc)); y <= Math.min(h - 1, Math.ceil(cy + rc)); y++)
    for (let x = Math.max(0, Math.floor(cx - rc)); x <= Math.min(w - 1, Math.ceil(cx + rc)); x++)
      if (Math.hypot(y - cy, x - cx) <= rc) core.push(s[y * w + x]);
  const level = median(core);
  // rings of 1 sensor px, as calibration.py
  const nr = Math.floor(eqDiam / 2 * scale);
  const tot = new Float64Array(nr + 1), ok = new Float64Array(nr + 1);
  const rmaxPx = nr / scale + 1;
  for (let y = Math.max(0, Math.floor(cy - rmaxPx)); y <= Math.min(h - 1, Math.ceil(cy + rmaxPx)); y++)
    for (let x = Math.max(0, Math.floor(cx - rmaxPx)); x <= Math.min(w - 1, Math.ceil(cx + rmaxPx)); x++) {
      const r = Math.hypot(y - cy, x - cx) * scale;
      const k = Math.floor(r) + 1;            // ring [k − 1, k)
      if (k > nr) continue;
      tot[k]++;
      if (Math.abs(s[y * w + x] / level - 1) <= READ_TOL) ok[k]++;
    }
  let rmax = 0;
  for (let rr = READ_RING_FROM; rr < nr; rr++) {
    if (tot[rr] > 0 && ok[rr] / tot[rr] < READ_RING_OK) break;
    rmax = rr;
  }
  return { cy, cx, area, eqDiam, plateauR: rmax / scale, level, scale };
}

/** Index of the reference frame, the longest exposure with no pixel at the ceiling in any channel (the first one if none). */
export function referenceIndex(texpUs: number[], max12: number[][], ceil12: number): number {
  let k = -1;
  texpUs.forEach((t, i) => { if (max12[i].every((v) => v < ceil12) && (k < 0 || t > texpUs[k])) k = i; });
  return k < 0 ? 0 : k;
}

/** Mean per channel over r ≤ R and the pixels of each channel at the ceiling. */
export function readSignal(f: ReadFrame, tC: number, texpUs: number, cy: number, cx: number, R: number, ceil12: number): ReadSignal {
  const sum = [0, 0, 0], sat = [0, 0, 0];
  let n = 0;
  for (let y = Math.max(0, Math.floor(cy - R)); y <= Math.min(f.h - 1, Math.ceil(cy + R)); y++)
    for (let x = Math.max(0, Math.floor(cx - R)); x <= Math.min(f.w - 1, Math.ceil(cx + R)); x++) {
      if (Math.hypot(y - cy, x - cx) > R) continue;
      const p = 3 * (y * f.w + x);
      n++;
      for (let c = 0; c < 3; c++) { const v = f.dn[p + c]; sum[c] += v; if (v >= ceil12) sat[c]++; }
    }
  return { tC, texpUs, n, S: { R: sum[0] / n, G: sum[1] / n, B: sum[2] / n }, nSat: { R: sat[0], G: sat[1], B: sat[2] } };
}

/** The disc radius common to every temperature, the smallest plateau radius. */
export const commonRadius = (g: ReadGeometry[]) => Math.min(...g.map((x) => x.plateauR));
