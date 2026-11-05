/**
 * Display helpers of step 5, pure and without prose. Two independent things.
 *
 * Fine grid. SAT and the two networks are functions of position, so they can be drawn with more points. The
 * fine grid holds every point of the base grid exactly, adds factor − 1 points between neighbours and stops at the
 * last radius and at the first and last height. Towards the axis it goes down to r = 0, which lies between the two
 * mirrored rays ±r₀ of a symmetric field, so it is not an extrapolation. A fine point is valid only when the base
 * cell or cells nearest to it are valid, so the shown zone is the zone of the base grid.
 *
 * Mask by f_s. The maximum is taken over the zone used, the threshold is a percentage of it, and nothing is recalculated.
 */
import { zeros, cholSolveFactored, type Mat } from '../linalg.ts';
import type { SatFactor } from '../sat.ts';

export const REFINE_FACTORS = [1, 2, 4] as const;
export type RefineFactor = (typeof REFINE_FACTORS)[number];
/**
 * Most points offered in one fine view. The laboratory example at factor 4 with a network is 50 375 points, 0,3 to 0,6 s in
 * the Worker and a CSV of 1,2 MB. At this limit a network takes about 11 s (measured in Node) and the CSV is about 50 MB.
 * A factor over the limit is not offered.
 */
export const REFINE_MAX_POINTS = 2_000_000;
/** Largest difference accepted at the points shared with the base grid (the measured one is 0). */
export const REFINE_TOL_T_K = 1e-3, REFINE_TOL_REL = 1e-9;

export const refineFactor = (v: unknown): RefineFactor => (v === 2 || v === 4 ? v : 1);

export interface FineGrid {
  fr: number; fz: number; nr: number; nz: number; rMm: Float64Array; zMm: Float64Array;
  /** Fine points inside the first ray (r₀ − k dr / fr ≥ 0), so the first point is r = 0 when fr > 1. */
  inner: number;
  /** The first radius is r = 0, and the mirrored map shares that column. */
  axisNode: boolean;
  /** Base index of each fine index, −1 when the fine point is not on the base grid. */
  onR: Int32Array; onZ: Int32Array;
  /** Nearest base index of each fine index, the two differ when the fine point is half way. */
  nearR: [Int32Array, Int32Array]; nearZ: [Int32Array, Int32Array];
}

function axisIndex(base: ArrayLike<number>, f: number, inner: number): { v: Float64Array; on: Int32Array; near: [Int32Array, Int32Array] } {
  const n = base.length, m = (n - 1) * f + 1 + inner;
  const v = new Float64Array(m), on = new Int32Array(m).fill(-1), a = new Int32Array(m), b = new Int32Array(m);
  const step0 = n > 1 ? (base[1] - base[0]) / f : 0;
  for (let s = 0; s < inner; s++) { const x = base[0] - (inner - s) * step0; v[s] = Math.abs(x) < 1e-9 * Math.abs(step0) ? 0 : x; }
  for (let s = inner; s < m; s++) {
    const u = s - inner, lo = Math.floor(u / f), rem = u - lo * f;
    if (!rem) { v[s] = base[lo]; on[s] = lo; a[s] = b[s] = lo; continue; }
    v[s] = base[lo] + ((base[lo + 1] - base[lo]) * rem) / f;
    a[s] = 2 * rem > f ? lo + 1 : lo;
    b[s] = 2 * rem >= f ? lo + 1 : lo;
  }
  return { v, on, near: [a, b] };
}

/** rMm and zMm of the base grid, r increasing from (½) dr with constant step, z increasing. */
export function fineGrid(rMm: ArrayLike<number>, zMm: ArrayLike<number>, fr: number, fz: number): FineGrid {
  const dr = rMm.length > 1 ? rMm[1] - rMm[0] : 0;
  const inner = fr > 1 && dr > 0 ? Math.floor((rMm[0] * fr) / dr + 1e-9) : 0;
  const R = axisIndex(rMm, fr, inner), Z = axisIndex(zMm, fz, 0);
  return { fr, fz, nr: R.v.length, nz: Z.v.length, rMm: R.v, zMm: Z.v, inner, axisNode: inner > 0 && R.v[0] === 0, onR: R.on, onZ: Z.on, nearR: R.near, nearZ: Z.near };
}

/** How many points a factor gives. SAT refines only the radius, each height is a fit of its own. */
export function finePoints(nr: number, nz: number, fr: number, fz: number): number {
  return ((nr - 1) * fr + 1 + (fr > 1 ? fr / 2 : 0)) * ((nz - 1) * fz + 1);
}

/** Calls fn with the base cells nearest to each fine point; a point half way has up to four of them. */
export function forNearest(g: FineGrid, baseNr: number, fn: (fineIndex: number, cells: number[]) => void): void {
  const cells: number[] = [];
  for (let l = 0; l < g.nz; l++) for (let s = 0; s < g.nr; s++) {
    cells.length = 0;
    const k0 = g.nearZ[0][l], k1 = g.nearZ[1][l], i0 = g.nearR[0][s], i1 = g.nearR[1][s];
    cells.push(k0 * baseNr + i0);
    if (i1 !== i0) cells.push(k0 * baseNr + i1);
    if (k1 !== k0) { cells.push(k1 * baseNr + i0); if (i1 !== i0) cells.push(k1 * baseNr + i1); }
    fn(l * g.nr + s, cells);
  }
}

/** Fine index of each base point, in the (k, i) order of the base grid. */
export function basePointsOnFine(g: FineGrid, baseNz: number, baseNr: number): Int32Array {
  const out = new Int32Array(baseNz * baseNr);
  for (let k = 0; k < baseNz; k++) for (let i = 0; i < baseNr; i++) out[k * baseNr + i] = k * g.fz * g.nr + g.inner + i * g.fr;
  return out;
}

/** The clamped cubic basis of `splineBasis` (sat.ts) at any x in [0, 1), with the same zero slope on the axis. */
export function splineBasisAt(xs: ArrayLike<number>, segments: number): Mat {
  const k = 3, n = xs.length;
  const knots: number[] = [];
  for (let q = 0; q < k; q++) knots.push(0);
  for (let q = 0; q <= segments; q++) knots.push(q / segments);
  for (let q = 0; q < k; q++) knots.push(1);
  const M = knots.length - k - 1;
  const S = zeros(n, M - 1);
  for (let i = 0; i < n; i++) {
    const x = Math.min(xs[i], 1 - 1e-12);
    let N = new Float64Array(knots.length - 1);
    for (let q = 0; q < N.length; q++) N[q] = x >= knots[q] && x < knots[q + 1] ? 1 : 0;
    for (let d = 1; d <= k; d++) {
      const Nn = new Float64Array(knots.length - 1 - d);
      for (let q = 0; q < Nn.length; q++) {
        const a = knots[q + d] - knots[q], b = knots[q + d + 1] - knots[q + 1];
        Nn[q] = (a > 0 ? ((x - knots[q]) / a) * N[q] : 0) + (b > 0 ? ((knots[q + d + 1] - x) / b) * N[q + 1] : 0);
      }
      N = Nn;
    }
    S.a[i * (M - 1)] = N[0] + N[1];
    for (let q = 2; q < M; q++) S.a[i * (M - 1) + q - 1] = N[q];
  }
  return S;
}

/** x of each fine radius in the normalised radius of `splineBasis`, the base points with the same arithmetic, (i + ½)/n. */
export function splineX(g: FineGrid, baseNr: number): Float64Array {
  const out = new Float64Array(g.nr);
  for (let s = 0; s < g.nr; s++) {
    const on = g.onR[s];
    // in units of dr / fr the radius is s − inner + fr / 2, because the first base ray is at dr / 2
    out[s] = on >= 0 ? (on + 0.5) / baseNr : (s - g.inner + g.fr / 2) / (g.fr * baseNr);
  }
  return out;
}

/** Spline coefficients c of one projection, the same arithmetic as `satApply` before it multiplies by B. */
export function satCoef(F: SatFactor, g: ArrayLike<number>): Float64Array {
  const { M, G } = F;
  const rhs = new Float64Array(M);
  for (let q = 0; q < M; q++) { let t = 0; for (let i = 0; i < G.n; i++) t += G.a[i * M + q] * g[i]; rhs[q] = t; }
  return cholSolveFactored(F.R, M, rhs);
}

/** f = B c on the given basis. */
export function satEval(B: Mat, c: ArrayLike<number>): Float64Array {
  const f = new Float64Array(B.n);
  for (let i = 0; i < B.n; i++) { let t = 0; for (let q = 0; q < B.m; q++) t += B.a[i * B.m + q] * c[q]; f[i] = t; }
  return f;
}

// ---------------------------------------------------------------------------- mask by f_s
export interface FsMask { pct: number; max: number; thr: number; pass: Uint8Array; nZone: number; nPass: number }
export const FS_MASK_DEFAULT_PCT = 5;
export const validMaskPct = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0 && v <= 100;

/**
 * Points with f_s above pct % of its maximum in the zone used (mask value 1). `fs` is read and never written. A point
 * outside the zone with f_s above the threshold passes too, because it is on the map. Null when the zone has no
 * positive f_s, so there is nothing to compare with.
 */
export function fsMaskOf(fs: ArrayLike<number>, mask: ArrayLike<number>, pct: number): FsMask | null {
  let max = -Infinity, nZone = 0;
  for (let q = 0; q < fs.length; q++) if (mask[q] === 1 && Number.isFinite(fs[q])) { nZone++; if (fs[q] > max) max = fs[q]; }
  if (!(max > 0)) return null;
  const thr = (pct / 100) * max;
  const pass = new Uint8Array(fs.length);
  let nPass = 0;
  for (let q = 0; q < fs.length; q++) if (fs[q] > thr) { pass[q] = 1; if (mask[q] === 1) nPass++; }
  return { pct, max, thr, pass, nZone, nPass };
}

/** Contiguous runs of the indices where keep is true, for the profiles under the mask, so a line does not cross a hole. */
export function runsOf(n: number, keep: (i: number) => boolean): [number, number][] {
  const out: [number, number][] = [];
  let a = -1;
  for (let i = 0; i < n; i++) {
    if (keep(i)) { if (a < 0) a = i; } else if (a >= 0) { out.push([a, i]); a = -1; }
  }
  if (a >= 0) out.push([a, n]);
  return out;
}
