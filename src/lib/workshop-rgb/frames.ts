/**
 * Frames of the workshop tool. Saturation and hot pixels in raw counts, the 2 × 2 replication
 * phase of the pylon RGB, the Bayer superpixel, 2 × 2 grouping, dark subtraction and the Float32
 * accumulator of WEB_SPEC §7.3. Images are chunky (pixel interleaved), row 0 at the top,
 * index (y·W + x)·nc + c.
 */
import { median, percentile } from './stats.ts';

export interface Planes { width: number; height: number; nc: number; data: ArrayLike<number> }

/** Saturation threshold, a fraction of the ceiling in raw counts (WEB_SPEC §5.3). */
export const SAT_FRAC = 0.995;

/** Per channel saturation mask, 1 where raw ≥ SAT_FRAC × ceiling. NaN ceiling (float) marks nothing. */
export function saturationMask(img: Planes, ceiling: number, frac = SAT_FRAC): Uint8Array {
  const n = img.width * img.height * img.nc;
  const m = new Uint8Array(n);
  if (!Number.isFinite(ceiling)) return m;
  const thr = frac * ceiling;
  for (let i = 0; i < n; i++) if (img.data[i] >= thr) m[i] = 1;
  return m;
}

export function countPerChannel(mask: Uint8Array, nc: number): number[] {
  const out = new Array(nc).fill(0);
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[i % nc]++;
  return out;
}

/**
 * 3 × 3 dilation. With `acrossChannels` a pixel flagged in any channel flags all channels of its
 * 3 × 3 neighbourhood (pylon RGB: the demosaic leaves neighbours with intermediate values).
 */
export function dilate3x3(mask: Uint8Array, width: number, height: number, nc: number, acrossChannels = true): Uint8Array {
  const out = new Uint8Array(mask.length);
  const any = new Uint8Array(width * height);
  if (acrossChannels) for (let p = 0; p < width * height; p++) { for (let c = 0; c < nc; c++) if (mask[p * nc + c]) { any[p] = 1; break; } }
  for (let c = 0; c < (acrossChannels ? 1 : nc); c++) {
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      let hit = 0;
      for (let dy = -1; dy <= 1 && !hit; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= width) continue;
          if (acrossChannels ? any[yy * width + xx] : mask[(yy * width + xx) * nc + c]) { hit = 1; break; }
        }
      }
      if (!hit) continue;
      if (acrossChannels) for (let k = 0; k < nc; k++) out[(y * width + x) * nc + k] = 1;
      else out[(y * width + x) * nc + c] = 1;
    }
  }
  return out;
}

function channel(img: Planes, c: number): Float64Array {
  const n = img.width * img.height;
  const out = new Float64Array(n);
  for (let p = 0; p < n; p++) out[p] = img.data[p * img.nc + c];
  return out;
}

/**
 * Robust σ of a channel, 1,4826 × MAD. When the MAD is zero (quantised or clipped dark frames), the
 * standard deviation (ddof 1) of the values up to the 99,9 percentile is used instead, so a few hot
 * pixels do not set it, and 1 if that is zero too.
 */
export function robustSigma(v: ArrayLike<number>): { med: number; sigma: number } {
  const med = median(v);
  const dev = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) dev[i] = Math.abs(v[i] - med);
  let sigma = 1.4826 * median(dev);
  if (!(sigma > 0)) {
    const thr = percentile(v, 99.9);
    let s = 0, n = 0;
    for (let i = 0; i < v.length; i++) if (v[i] <= thr) { s += v[i]; n++; }
    const m = s / n;
    let q = 0;
    for (let i = 0; i < v.length; i++) if (v[i] <= thr) q += (v[i] - m) ** 2;
    sigma = n > 1 ? Math.sqrt(q / (n - 1)) : 0;
    if (!(sigma > 0)) sigma = 1;
  }
  return { med, sigma };
}

/**
 * Hot pixels from a dark frame (WEB_SPEC §4, 1a.4), per channel. A pixel is hot when the dark is
 * above median + k σ (k = 10 by default) or saturated in the dark. Returns the mask and per channel
 * median, σ and count.
 */
export function hotPixelsFromDark(dark: Planes, ceiling: number, k = 10): { mask: Uint8Array; med: number[]; sigma: number[]; count: number[] } {
  const { width, height, nc } = dark;
  const mask = new Uint8Array(width * height * nc);
  const med: number[] = [], sigma: number[] = [], count: number[] = [];
  const sat = Number.isFinite(ceiling) ? SAT_FRAC * ceiling : Infinity;
  for (let c = 0; c < nc; c++) {
    const v = channel(dark, c);
    const r = robustSigma(v);
    med.push(r.med); sigma.push(r.sigma);
    const thr = r.med + k * r.sigma;
    let n = 0;
    for (let p = 0; p < v.length; p++) if (v[p] > thr || v[p] >= sat) { mask[p * nc + c] = 1; n++; }
    count.push(n);
  }
  return { mask, med, sigma, count };
}

const mirror = (i: number, n: number) => (i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i);

/**
 * Median of the 5 × 5 neighbourhood of every pixel of one channel, mirrored at the borders (scipy
 * 'mirror'), so a smooth ramp has zero residual also at the edges. Images of at least 3 × 3.
 */
export function localMedian5(v: ArrayLike<number>, width: number, height: number): Float64Array {
  const out = new Float64Array(width * height);
  const buf = new Float64Array(25);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let n = 0;
    for (let dy = -2; dy <= 2; dy++) {
      const yy = mirror(y + dy, height);
      for (let dx = -2; dx <= 2; dx++) buf[n++] = v[yy * width + mirror(x + dx, width)];
    }
    out[y * width + x] = median(buf);
  }
  return out;
}

/**
 * Hot pixels without a dark frame: the pixels that depart more than kSigma robust σ from their local
 * 5 × 5 median in the shortest exposure and stay so in every frame (frames sorted by exposure,
 * shortest first). Per channel. The two pixel border is never flagged, because there the 5 × 5 window
 * is mirrored and a strong gradient alone can look like a hot pixel.
 *
 * The scale of each residual is the larger of the global robust σ and the local one (1,4826 times the
 * median absolute residual of the 5 × 5 window), so shot noise inside a bright aperture and the steep
 * edge of the aperture are not taken for hot pixels (GB3r1-estudiante-12, with the global σ alone the
 * real oven example flagged 3 % of its channels, almost all inside the aperture).
 */
export function hotPixelsNoDark(frames: Planes[], kSigma = 5): Uint8Array {
  const { width, height, nc } = frames[0];
  const mask = new Uint8Array(width * height * nc).fill(1);
  for (const f of frames) {
    for (let c = 0; c < nc; c++) {
      const v = channel(f, c);
      const lm = localMedian5(v, width, height);
      const res = new Float64Array(v.length);
      for (let p = 0; p < v.length; p++) res[p] = v[p] - lm[p];
      const { med, sigma } = robustSigma(res);
      const dev = new Float64Array(v.length);
      for (let p = 0; p < v.length; p++) dev[p] = Math.abs(res[p] - med);
      const loc = localMedian5(dev, width, height);
      for (let p = 0; p < v.length; p++) {
        const y = Math.floor(p / width), x = p % width;
        const edge = y < 2 || x < 2 || y >= height - 2 || x >= width - 2;
        const s = Math.max(sigma, 1.4826 * loc[p]);
        if (edge || !(Math.abs(res[p] - med) > kSigma * s)) mask[p * nc + c] = 0;
      }
    }
  }
  return mask;
}

/**
 * Replace flagged pixels by the median of the unflagged pixels of the same channel in their 3 × 3
 * neighbourhood (WEB_SPEC §4, etapa 3). In place on a Float32 or Float64 image.
 */
export function replaceHot(img: { width: number; height: number; nc: number; data: Float32Array | Float64Array }, mask: Uint8Array): number {
  const { width, height, nc, data } = img;
  const src = data.slice();
  const buf = new Float64Array(9);
  let n = 0;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < nc; c++) {
    const i = (y * width + x) * nc + c;
    if (!mask[i]) continue;
    let m = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || yy >= height || xx < 0 || xx >= width || (dx === 0 && dy === 0)) continue;
      const j = (yy * width + xx) * nc + c;
      if (!mask[j]) buf[m++] = src[j];
    }
    if (m) { data[i] = median(buf.subarray(0, m)); n++; }
  }
  return n;
}

/**
 * Phase of the 2 × 2 replication of the pylon RGB (WEB_SPEC §4, etapa 3). For every channel and each
 * phase (py, px) ∈ {0, 1}², the fraction of complete 2 × 2 blocks starting at (py + 2a, px + 2b) whose
 * four values are equal, counting only blocks that are not all zero. The phase with the largest fraction
 * is reported, and `replicated` when it reaches minFrac.
 */
export function replicationPhase(img: Planes, minFrac = 0.9): { phase: [number, number]; frac: number; fracs: number[]; replicated: boolean }[] {
  const { width, height, nc, data } = img;
  const out: { phase: [number, number]; frac: number; fracs: number[]; replicated: boolean }[] = [];
  for (let c = 0; c < nc; c++) {
    const fracs: number[] = [];
    for (const [py, px] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
      let eq = 0, tot = 0;
      for (let y = py; y + 1 < height; y += 2) for (let x = px; x + 1 < width; x += 2) {
        const a = data[(y * width + x) * nc + c], b = data[(y * width + x + 1) * nc + c];
        const d = data[((y + 1) * width + x) * nc + c], e = data[((y + 1) * width + x + 1) * nc + c];
        if (a === 0 && b === 0 && d === 0 && e === 0) continue;
        tot++;
        if (a === b && a === d && a === e) eq++;
      }
      fracs.push(tot ? eq / tot : 0);
    }
    let k = 0;
    for (let q = 1; q < 4; q++) if (fracs[q] > fracs[k]) k = q;
    out.push({ phase: [k >> 1, k & 1], frac: fracs[k], fracs, replicated: fracs[k] >= minFrac });
  }
  return out;
}

export type BayerPattern = 'RGGB' | 'BGGR' | 'GRBG' | 'GBRG';

/**
 * Bayer superpixel 2 × 2 without interpolation (WEB_SPEC §5.3). For RGGB, R = p₀₀, G = (p₀₁ + p₁₀)/2,
 * B = p₁₁, and the permutations. A superpixel is saturated in every channel if any of its four sites
 * is (`satMask` of the mosaic, one channel). Odd trailing rows or columns are dropped. The scale in mm
 * per pixel doubles on both axes (`scale` = 2). The tool calls it with `centre` (below).
 */
export function bayerSuperpixel(mosaic: { width: number; height: number; data: ArrayLike<number> }, pattern: BayerPattern = 'RGGB', satMask?: Uint8Array, centre = false): { width: number; height: number; nc: number; data: Float32Array; sat: Uint8Array; scale: number } {
  const W = mosaic.width >> 1, H = mosaic.height >> 1;
  const data = new Float32Array(W * H * 3);
  const sat = new Uint8Array(W * H * 3);
  const idx: Record<string, number[]> = { R: [], G: [], B: [] };
  for (let k = 0; k < 4; k++) idx[pattern[k]].push(k);
  const off = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const src = mosaic.data, w = mosaic.width;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = off.map(([dy, dx]) => src[(2 * y + dy) * w + 2 * x + dx]);
    const o = (y * W + x) * 3;
    data[o] = v[idx.R[0]];
    data[o + 1] = (v[idx.G[0]] + v[idx.G[1]]) / 2;
    data[o + 2] = v[idx.B[0]];
    if (satMask) {
      let s = 0;
      for (const [dy, dx] of off) if (satMask[(2 * y + dy) * w + 2 * x + dx]) s = 1;
      if (s) sat[o] = sat[o + 1] = sat[o + 2] = 1;
    }
  }
  if (centre) centreSites(data, sat, W, H, off[idx.R[0]], off[idx.B[0]]);
  return { width: W, height: H, nc: 3, data, sat, scale: 2 };
}

/**
 * With `centre` the R and B sites, which sit ¼ superpixel off the centre of the green pair on each
 * axis, are moved to that centre by bilinear interpolation (weights ¾ and ¼ with the neighbouring
 * superpixel, the edge repeats itself). Without it a radial profile of R and B is shifted against G
 * by half a mosaic pixel (GB3r2-adv-03). Saturation spreads to the neighbour used.
 */
function centreSites(data: Float32Array, sat: Uint8Array, W: number, H: number, offR: number[] | null, offB: number[] | null) {
  for (const [c, off] of [[0, offR], [2, offB]] as [number, number[] | null][]) {
    if (!off) continue;
    const [dy, dx] = off;
    const sy = dy === 0 ? 1 : -1, sx = dx === 0 ? 1 : -1;
    const v = new Float32Array(W * H), sv = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { v[y * W + x] = data[(y * W + x) * 3 + c]; sv[y * W + x] = sat[(y * W + x) * 3 + c]; }
    for (let y = 0; y < H; y++) {
      const y1 = Math.min(H - 1, Math.max(0, y + sy));
      for (let x = 0; x < W; x++) {
        const x1 = Math.min(W - 1, Math.max(0, x + sx));
        const a = v[y * W + x], b = v[y * W + x1], d = v[y1 * W + x], e = v[y1 * W + x1];
        data[(y * W + x) * 3 + c] = 0.5625 * a + 0.1875 * b + 0.1875 * d + 0.0625 * e;
        if (sv[y * W + x1] || sv[y1 * W + x] || sv[y1 * W + x1]) sat[(y * W + x) * 3 + c] = 1;
      }
    }
  }
}

/**
 * Mosaic sites of R and B inside a 2 × 2 group of the pylon RGB (decision after G4 r3, G4r3-conf-24). pylon
 * copies each R and B site over a 2 × 2 block (replication phase from `replicationPhase`), and the site sits
 * at offset (p − phase) mod 2 of its block, with p the parity of that colour in the Bayer pattern. On a group
 * grid aligned with the red phase the same offset holds for B, whose four half blocks average to it. Offset
 * 0 is the top or left half of the group, 1 the bottom or right half, as `off` of `bayerSuperpixel`. A
 * channel without replication (null phase) is interpolated on every pixel and needs no centring.
 */
export function pylonSites(pattern: BayerPattern, phaseR: [number, number] | null, phaseB: [number, number] | null): { R: [number, number] | null; B: [number, number] | null } {
  const at = (ch: 'R' | 'B'): [number, number] => { const k = pattern.indexOf(ch); return [k >> 1, k & 1]; };
  const m2 = (v: number) => ((v % 2) + 2) % 2;
  const site = (ch: 'R' | 'B', ph: [number, number] | null): [number, number] | null => {
    if (!ph) return null;
    const p = at(ch);
    return [m2(p[0] - ph[0]), m2(p[1] - ph[1])];
  };
  return { R: site('R', phaseR), B: site('B', phaseB) };
}

/**
 * 2 × 2 grouping of an RGB frame on a grid that starts at (y0, x0), mean of the four values per
 * channel, saturated if any of the four is. Used by default on the pylon RGB (decision after GB1). The
 * scale in mm per pixel doubles. With `centre` (sites from `pylonSites`) the R and B values, which sit
 * ¼ group off the centre of the interpolated green, are moved to that centre by the same bilinear step
 * as the Bayer path (`centreSites`, G4 r3). The tool does it only when asked, off by default (decision on
 * closing G4), and `rgbShiftTest` on the ruler says whether it helps.
 */
export function bin2x2(img: Planes, satMask?: Uint8Array, y0 = 0, x0 = 0, centre?: { R: [number, number] | null; B: [number, number] | null }): { width: number; height: number; nc: number; data: Float32Array; sat: Uint8Array; scale: number } {
  const { width, nc } = img;
  const W = (img.width - x0) >> 1, H = (img.height - y0) >> 1;
  const data = new Float32Array(W * H * nc);
  const sat = new Uint8Array(W * H * nc);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < nc; c++) {
    let s = 0, m = 0;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const i = ((y0 + 2 * y + dy) * width + x0 + 2 * x + dx) * nc + c;
      s += img.data[i];
      if (satMask && satMask[i]) m = 1;
    }
    const o = (y * W + x) * nc + c;
    data[o] = s / 4;
    sat[o] = m;
  }
  if (centre && nc === 3) centreSites(data, sat, W, H, centre.R, centre.B);
  return { width: W, height: H, nc, data, sat, scale: 2 };
}

/** Mean over b × b blocks (b = 4 or 8 for the previews and the asymmetry copy). */
export function binMean(img: Planes, b: number): { width: number; height: number; nc: number; data: Float64Array } {
  const W = Math.floor(img.width / b), H = Math.floor(img.height / b), nc = img.nc;
  const data = new Float64Array(W * H * nc);
  for (let y = 0; y < H * b; y++) for (let x = 0; x < W * b; x++) for (let c = 0; c < nc; c++)
    data[(Math.floor(y / b) * W + Math.floor(x / b)) * nc + c] += img.data[(y * img.width + x) * nc + c];
  for (let i = 0; i < data.length; i++) data[i] /= b * b;
  return { width: W, height: H, nc, data };
}

/** Convert raw file counts to 12 bit DN (common unit of WEB_SPEC §5.3), Float32. */
export function toDn12(data: ArrayLike<number>, factor: number): Float32Array {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = data[i] * factor;
  return out;
}

/**
 * Accumulator of a series (WEB_SPEC §7.3). Sums raw integer counts in Float32 (exact up to 256 frames
 * of 65520), ORs the saturation mask of every frame, and removes an excluded frame exactly.
 */
export class Accumulator {
  width: number;
  height: number;
  nc: number;
  sum: Float32Array;
  satCount: Uint16Array;
  count: number;
  constructor(width: number, height: number, nc: number) {
    this.width = width;
    this.height = height;
    this.nc = nc;
    this.sum = new Float32Array(width * height * nc);
    this.satCount = new Uint16Array(width * height * nc);
    this.count = 0;
  }
  add(data: ArrayLike<number>, sat?: Uint8Array): void {
    if (data.length !== this.sum.length) throw new Error('series_mismatch');
    for (let i = 0; i < data.length; i++) this.sum[i] += data[i];
    if (sat) for (let i = 0; i < sat.length; i++) if (sat[i]) this.satCount[i]++;
    this.count++;
  }
  remove(data: ArrayLike<number>, sat?: Uint8Array): void {
    if (data.length !== this.sum.length || this.count < 1) throw new Error('series_mismatch');
    for (let i = 0; i < data.length; i++) this.sum[i] -= data[i];
    if (sat) for (let i = 0; i < sat.length; i++) if (sat[i]) this.satCount[i]--;
    this.count--;
  }
  /** Saturated in at least one accepted frame. */
  satMask(): Uint8Array {
    const m = new Uint8Array(this.satCount.length);
    for (let i = 0; i < m.length; i++) if (this.satCount[i]) m[i] = 1;
    return m;
  }
  /**
   * Mean of the accepted frames minus the dark (or background) of the same t_exp, both in raw counts,
   * times `toDn12`. By linearity this equals subtracting the dark from every frame.
   */
  meanMinus(dark: ArrayLike<number> | null, toDn12 = 1): Float32Array {
    const out = new Float32Array(this.sum.length);
    const n = this.count || 1;
    for (let i = 0; i < out.length; i++) out[i] = (this.sum[i] / n - (dark ? dark[i] : 0)) * toDn12;
    return out;
  }
}

/** (S − D) / (t_exp · K_j) per channel, K per channel scalar or per pixel (WEB_SPEC §3 step 2). */
export function toRadiance(sig: ArrayLike<number>, nc: number, texpS: number, K: number[] | Float32Array[]): Float64Array {
  const out = new Float64Array(sig.length);
  const per = typeof K[0] !== 'number';
  for (let i = 0; i < sig.length; i++) {
    const c = i % nc;
    const k = per ? (K[c] as Float32Array)[(i - c) / nc] : (K[c] as number);
    out[i] = sig[i] / (texpS * k);
  }
  return out;
}

export type ShiftVerdict = 'activate' | 'keep_off' | 'inconclusive' | 'not_applicable';
export type Pair = [number, number];
export interface ShiftTest {
  verdict: ShiftVerdict;
  why: '' | 'not_replicated' | 'few_edges';
  /** Shift of R and B against G that the centring assumes, [y, x] in groups (± ¼). */
  pred: { R: Pair | null; B: Pair | null };
  /** Measured shift [y, x] in groups, NaN on an axis without edges. */
  meas: { R: Pair | null; B: Pair | null };
  /** Uncertainty of the measured shift from the spread between tiles, [y, x] in groups. */
  unc: { R: Pair | null; B: Pair | null };
  /** Axes with enough edges, [y, x]. */
  axes: [boolean, boolean];
  /** Fraction of the assumed shift that is measured (1 = as assumed, 0 = none, −1 = the other way) and its uncertainty. */
  k: number; kUnc: number;
  nEdge: number; tiles: number;
}

function solveSmall(A: number[][], b: number[]): number[] | null {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-300) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r[n] / r[i]);
}

/**
 * Shift test of R and B against G on an image with edges, the ruler (decision on closing G4). `f` is the
 * pylon RGB grouped 2 × 2 on the red phase **without** centring, `sites` the mosaic sites of `pylonSites`.
 * On the strongest 10 % of the green gradients (no saturated neighbour, 3 pixels from the border) it fits
 * c ≈ a·G(x − s) + d for R and for B by Gauss–Newton on the bilinear G, so s is the shift of the channel's
 * edges against the green, in groups. The centring assumes s = (½ − site)/2 on each axis, ¼ group. The
 * same fit on 3 × 3 tiles gives the spread (lateral chromatic aberration changes sign across the image,
 * the mosaic shift does not). k is the fraction of the assumed shift that is measured, projected on it.
 * The centring reduces the error when k > ½, so the verdict is `activate` if k − 2σ > ½, `keep_off` if
 * k + 2σ < ½ and `inconclusive` otherwise.
 */
export function rgbShiftTest(f: { w: number; h: number; dn: ArrayLike<number>; sat: Uint8Array }, sites: { R: Pair | null; B: Pair | null }): ShiftTest {
  const W = f.w, H = f.h, n = W * H;
  const pred = { R: sites.R ? [(0.5 - sites.R[0]) / 2, (0.5 - sites.R[1]) / 2] as Pair : null, B: sites.B ? [(0.5 - sites.B[0]) / 2, (0.5 - sites.B[1]) / 2] as Pair : null };
  const out: ShiftTest = { verdict: 'not_applicable', why: 'not_replicated', pred, meas: { R: null, B: null }, unc: { R: null, B: null }, axes: [false, false], k: NaN, kUnc: NaN, nEdge: 0, tiles: 0 };
  if (!pred.R && !pred.B) return out;
  const pl = [0, 1, 2].map((c) => { const v = new Float64Array(n); for (let p = 0; p < n; p++) v[p] = f.dn[p * 3 + c]; return v; });
  const g = pl[1];
  const satAny = (p: number) => f.sat[3 * p] || f.sat[3 * p + 1] || f.sat[3 * p + 2];
  const M = 3;
  const cand: number[] = [], mag: number[] = [];
  let sxx = 0, syy = 0;
  for (let y = M; y < H - M; y++) for (let x = M; x < W - M; x++) {
    const p = y * W + x;
    if (satAny(p) || satAny(p - 1) || satAny(p + 1) || satAny(p - W) || satAny(p + W)) continue;
    const gx = (g[p + 1] - g[p - 1]) / 2, gy = (g[p + W] - g[p - W]) / 2;
    cand.push(p); mag.push(gx * gx + gy * gy);
  }
  const sorted = Float64Array.from(mag).sort();
  const thr = sorted.length ? sorted[Math.floor(0.9 * (sorted.length - 1))] : Infinity;
  const sel: number[] = [];
  for (let i = 0; i < cand.length; i++) if (mag[i] >= thr && mag[i] > 0) {
    const p = cand[i]; sel.push(p);
    const gx = (g[p + 1] - g[p - 1]) / 2, gy = (g[p + W] - g[p - W]) / 2;
    sxx += gx * gx; syy += gy * gy;
  }
  out.nEdge = sel.length;
  out.axes = [syy >= 0.05 * (sxx + syy), sxx >= 0.05 * (sxx + syy)];
  if (sel.length < 200 || (!out.axes[0] && !out.axes[1])) { out.verdict = 'inconclusive'; out.why = 'few_edges'; return out; }
  const bil = (y: number, x: number) => {
    const y0 = Math.floor(y), x0 = Math.floor(x), fy = y - y0, fx = x - x0;
    const p = y0 * W + x0;
    return (1 - fy) * ((1 - fx) * g[p] + fx * g[p + 1]) + fy * ((1 - fx) * g[p + W] + fx * g[p + W + 1]);
  };
  const fit = (c: number, idx: number[]): Pair => {
    const v = pl[c];
    let s: Pair = [0, 0];
    for (let it = 0; it < 6; it++) {
      const cols = 2 + (out.axes[0] ? 1 : 0) + (out.axes[1] ? 1 : 0);
      const A = Array.from({ length: cols }, () => new Array(cols).fill(0)), b = new Array(cols).fill(0);
      const row = new Array(cols);
      for (const p of idx) {
        const y = Math.floor(p / W) - s[0], x = (p % W) - s[1];
        let k = 0;
        row[k++] = bil(y, x); row[k++] = 1;
        if (out.axes[0]) row[k++] = (bil(y + 1, x) - bil(y - 1, x)) / 2;
        if (out.axes[1]) row[k++] = (bil(y, x + 1) - bil(y, x - 1)) / 2;
        for (let i = 0; i < cols; i++) { b[i] += row[i] * v[p]; for (let j = 0; j < cols; j++) A[i][j] += row[i] * row[j]; }
      }
      const q = solveSmall(A, b);
      if (!q || !(Math.abs(q[0]) > 0)) return [NaN, NaN];
      let k = 2;
      const dy = out.axes[0] ? -q[k++] / q[0] : 0, dx = out.axes[1] ? -q[k++] / q[0] : 0;
      s = [Math.max(-1, Math.min(1, s[0] + dy)), Math.max(-1, Math.min(1, s[1] + dx))];
      if (Math.abs(dy) < 1e-5 && Math.abs(dx) < 1e-5) break;
    }
    return [out.axes[0] ? s[0] : NaN, out.axes[1] ? s[1] : NaN];
  };
  const tiles: number[][] = Array.from({ length: 9 }, () => []);
  for (const p of sel) tiles[Math.min(2, Math.floor((3 * Math.floor(p / W)) / H)) * 3 + Math.min(2, Math.floor((3 * (p % W)) / W))].push(p);
  const used = tiles.filter((t) => t.length >= 60);
  out.tiles = used.length;
  let num = 0, den = 0, var_ = 0;
  for (const [c, ch] of [[0, 'R'], [2, 'B']] as [number, 'R' | 'B'][]) {
    const pr = pred[ch];
    if (!pr) continue;
    const m = fit(c, sel);
    const per = used.map((t) => fit(c, t));
    const u = [0, 1].map((a) => {
      const vals = per.map((q) => q[a]).filter(Number.isFinite);
      if (vals.length < 3) return NaN;
      const mu = vals.reduce((x, y) => x + y, 0) / vals.length;
      const sd = Math.sqrt(vals.reduce((x, y) => x + (y - mu) ** 2, 0) / (vals.length - 1));
      return Math.max(0.02, sd / Math.sqrt(vals.length));
    }) as Pair;
    out.meas[ch] = m; out.unc[ch] = u;
    for (const a of [0, 1]) if (out.axes[a] && Number.isFinite(m[a])) {
      num += m[a] * pr[a]; den += pr[a] * pr[a]; var_ += pr[a] * pr[a] * (Number.isFinite(u[a]) ? u[a] * u[a] : Infinity);
    }
  }
  out.k = den ? num / den : NaN;
  out.kUnc = den ? Math.sqrt(var_) / den : NaN;
  out.why = '';
  if (!Number.isFinite(out.k) || !Number.isFinite(out.kUnc)) { out.verdict = 'inconclusive'; out.why = 'few_edges'; }
  else if (out.k - 2 * out.kUnc > 0.5) out.verdict = 'activate';
  else if (out.k + 2 * out.kUnc < 0.5) out.verdict = 'keep_off';
  else out.verdict = 'inconclusive';
  return out;
}
