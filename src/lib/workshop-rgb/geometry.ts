/**
 * Flame axis (WEB_SPEC §4, etapa 3, "Centrar el eje"). Port of find_axis of annabel_app/core/preprocess.py
 * in the "boundary" convention: column k covers [k, k + 1), an axis index a puts the axis at x = a, the
 * boundary between columns a − 1 and a, and ray i of each half sits at (i + ½) px from it.
 *
 * Added for GB2r1-numerico-06. ANNAbel crops at the integer axis. A quarter pixel off centre makes the two
 * halves differ by ±4 % on the axis, so the tool also refines the axis to a fraction of a pixel (minimum
 * of the squared half difference with linear interpolation) and samples the halves there (`halvesAt`).
 * Images are (H, W, 3) chunky, oriented, background subtracted.
 */
import { median } from './stats.ts';

export interface AxisOptions {
  /** Half width of the comparison window in px (axis_half_width_px). */
  halfWidth: number;
  /** Search range [lo, hi) of axis indices (axis_search), default [halfWidth, W − halfWidth). */
  search?: [number, number];
  /** Rows used, those whose maximum in the search range passes this fraction of the frame maximum. */
  rowFrac?: number;
  minSnr?: number;
}

export type AxisCode = 'axis_search_reduced' | 'axis_search_too_small' | 'axis_empty' | 'axis_no_rows' | 'axis_low_signal' | 'axis_edge' | 'axis_ambiguous';

export interface AxisResult {
  /** Integer axis index (ANNAbel), null when no axis was found. */
  axis: number | null;
  asym: number;
  positions: number[];
  curve: number[];
  rows: [number, number] | null;
  search: [number, number] | null;
  /** Parabola through the three points around the minimum, as ANNAbel subpixel_pos_px. */
  subpixel: number | null;
  /** Continuous axis position from `refineAxis` (minimum of the interpolated half difference). */
  refined: number | null;
  snr: number;
  codes: AxisCode[];
}

/** Sum over channels, (H, W). */
function channelSum(img: ArrayLike<number>, w: number, h: number): Float64Array {
  const s = new Float64Array(w * h);
  for (let p = 0; p < w * h; p++) s[p] = img[3 * p] + img[3 * p + 1] + img[3 * p + 2];
  return s;
}

/** _asym_curve in the boundary convention on a band of rows (rows × w, clipped at 0). */
export function asymCurve(band: Float64Array, w: number, nRows: number, lo: number, hi: number, hw: number): { pos: number[]; vals: number[] } {
  const pos: number[] = [], vals: number[] = [];
  for (let a = lo; a < hi; a++) {
    let d = 0, tot = 0;
    for (let r = 0; r < nRows; r++) {
      const o = r * w;
      for (let i = 0; i < hw; i++) d += Math.abs(band[o + a - 1 - i] - band[o + a + i]);
      for (let k = a - hw; k < a + hw; k++) tot += band[o + k];
    }
    pos.push(a);
    vals.push(d / (tot + 1e-30));
  }
  return { pos, vals };
}

/** Value at continuous position x (column centres at k + ½), linear interpolation, edge values outside. */
export function sampleAt(row: ArrayLike<number>, x: number): number {
  const u = x - 0.5, n = row.length;
  if (u <= 0) return row[0];
  if (u >= n - 1) return row[n - 1];
  const k = Math.floor(u), t = u - k;
  return row[k] * (1 - t) + row[k + 1] * t;
}

/** Right and mirrored left halves of n rays at axis ± (i + ½), by linear interpolation (exact at an integer axis). */
export function halvesAt(row: ArrayLike<number>, axis: number, n: number): { right: Float64Array; left: Float64Array } {
  const right = new Float64Array(n), left = new Float64Array(n);
  for (let i = 0; i < n; i++) { right[i] = sampleAt(row, axis + i + 0.5); left[i] = sampleAt(row, axis - i - 0.5); }
  return { right, left };
}

/**
 * Continuous axis in [a − 1, a + 1] that minimises Σ (right − left)² over the band rows and hw rays,
 * golden section to 1e-4 px.
 */
export function refineAxis(band: Float64Array, w: number, nRows: number, a: number, hw: number): number {
  const cost = (x: number) => {
    let s = 0;
    for (let r = 0; r < nRows; r++) {
      const row = band.subarray(r * w, (r + 1) * w);
      const h = halvesAt(row, x, hw);
      for (let i = 0; i < hw; i++) s += (h.right[i] - h.left[i]) ** 2;
    }
    return s;
  };
  const g = (Math.sqrt(5) - 1) / 2;
  let lo = a - 1, hi = a + 1;
  let x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo), f1 = cost(x1), f2 = cost(x2);
  while (hi - lo > 1e-4) {
    if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - g * (hi - lo); f1 = cost(x1); }
    else { lo = x1; x1 = x2; f1 = f2; x2 = lo + g * (hi - lo); f2 = cost(x2); }
  }
  return (lo + hi) / 2;
}

/** find_axis of ANNAbel (boundary convention), with codes instead of Spanish flags, plus `refined`. */
export function findAxis(img: ArrayLike<number>, w: number, h: number, opts: AxisOptions): AxisResult {
  const s = channelSum(img, w, h);
  const hw = opts.halfWidth, rowFrac = opts.rowFrac ?? 0.15, minSnr = opts.minSnr ?? 10;
  const codes: AxisCode[] = [];
  const [lo, hi] = opts.search ?? [hw, w - hw];
  const loE = Math.max(lo, hw), hiE = Math.min(hi, w - hw);
  const none = (snr: number): AxisResult => ({ axis: null, asym: NaN, positions: [], curve: [], rows: null, search: null, subpixel: null, refined: null, snr, codes });
  if (loE !== lo || hiE !== hi) codes.push('axis_search_reduced');
  if (hiE - loE < 3) { codes.push('axis_search_too_small'); return none(NaN); }
  const med = median(s);
  const dev = new Float64Array(s.length);
  for (let p = 0; p < s.length; p++) dev[p] = Math.abs(s[p] - med);
  const noise = 1.4826 * median(dev) + 1e-30;
  let peak = -Infinity, smax = -Infinity;
  for (let p = 0; p < s.length; p++) { const x = p % w; if (x >= loE && x < hiE && s[p] > peak) peak = s[p]; if (s[p] > smax) smax = s[p]; }
  const snr = (peak - med) / noise;
  if (!Number.isFinite(snr) || peak <= 0) { codes.push('axis_empty'); return none(snr); }
  const rows: number[] = [];
  for (let y = 0; y < h; y++) {
    let m = -Infinity;
    for (let x = loE; x < hiE; x++) m = Math.max(m, s[y * w + x]);
    if (m > rowFrac * smax) rows.push(y);
  }
  if (!rows.length) { codes.push('axis_no_rows'); return none(snr); }
  const r0 = rows[0], r1 = rows[rows.length - 1], nR = r1 - r0 + 1;
  const band = new Float64Array(nR * w);
  for (let k = 0; k < nR * w; k++) band[k] = Math.max(0, s[r0 * w + k]);
  const { pos, vals } = asymCurve(band, w, nR, loE, hiE, hw);
  let i = 0;
  for (let k = 1; k < vals.length; k++) if (vals[k] < vals[i]) i = k;
  const a = pos[i];
  if (snr < minSnr) codes.push('axis_low_signal');
  if (i <= 0 || i >= pos.length - 1) codes.push('axis_edge');
  const loc: number[] = [];
  for (let j = 1; j < vals.length - 1; j++) if (vals[j] <= vals[j - 1] && vals[j] <= vals[j + 1] && Math.abs(pos[j] - a) > 3) loc.push(j);
  if (loc.length) {
    let j2 = loc[0];
    for (const j of loc) if (vals[j] < vals[j2]) j2 = j;
    if (vals[j2] < vals[i] * 1.05 + 1e-12) codes.push('axis_ambiguous');
  }
  let sub: number | null = null;
  if (i > 0 && i < vals.length - 1) {
    const y0 = vals[i - 1], y1 = vals[i], y2 = vals[i + 1], den = y0 - 2 * y1 + y2;
    if (den > 0) sub = a + Math.min(0.5, Math.max(-0.5, (0.5 * (y0 - y2)) / den));
  }
  const refined = a - 1 - hw >= 0 && a + 1 + hw <= w ? refineAxis(band, w, nR, a, hw) : null;
  return { axis: a, asym: vals[i], positions: pos, curve: vals, rows: [r0, r1], search: [loE, hiE], subpixel: sub, refined, snr, codes };
}
