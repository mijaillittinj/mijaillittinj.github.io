/**
 * Abel inversion of every height and channel (WEB_SPEC §4, etapa 4) on one mesh and one ray
 * convention, ANNAbel's: ring j on [j·dr, (j+1)·dr), ray i at (i + ½)·dr (`chordMatrixHalf`).
 *   onion     back substitution, no parameter
 *   tikhonov  one Cholesky factor per λ_rel reused for all rows and channels, regulariser 'gamma2'
 *             (second difference, the presentation's, prepare_data.py). The tool does not offer the
 *             'site' regulariser of abel.ts, whose λ_rel means about 6 times less (GB2r1-numerico-09).
 *             λ_rel stays in [LAMBDA_REL_MIN, LAMBDA_REL_MAX], the L-curve grid (GB2r1-numerico-08).
 *   sat       spline-based Abel transform (sat.ts, Littin et al., Fuel 374 (2024) 132365) with κ = 0,
 *             the operator in metres, one factor per λ reused
 * With dr in metres the projection P in W m⁻² sr⁻¹ gives j in W m⁻³ sr⁻¹.
 * Projections are rows × n (one Float64Array per height) per channel.
 * Saturation (WEB_SPEC etapa 3, GB2r1-numerico-07). `excludeRows` lists the heights with saturated pixels
 * in any channel. They leave the inversion in every channel and come out as NaN (J, reprojection and
 * residual), the other rows are untouched. With SAT, `rayMask` keeps the row and drops only the masked
 * rays from the least squares (one factor per masked row).
 */
import { chordMatrixHalf, tikhonovFactor, tikhonovApply, lcurve, regGram, type TikhonovReg } from '../abel.ts';
import { satFactor, satApply, splineBasis } from '../sat.ts';
import { matvec, solveUpper, type Mat } from '../linalg.ts';
import { gradient } from './stats.ts';

export type AbelMethod = 'onion' | 'tikhonov' | 'sat';
export interface AbelParams {
  lambdaRel?: number; segments?: number; satLambda?: number;
  /** Rows (heights) left out in every channel. */
  excludeRows?: ArrayLike<number>;
  /** SAT only, per row index a 0/1 mask of rays to drop (1 = drop), the row stays in. */
  rayMask?: Map<number, ArrayLike<number>>;
}

export const LAMBDA_REL_MIN = 1e-6;
export const LAMBDA_REL_MAX = 1e3;

export interface Inversion { method: AbelMethod; reg: TikhonovReg | null; L: Mat; J: Float64Array[][]; reproj: Float64Array[][]; resid: Float64Array[][]; params: AbelParams; excluded: number[] }

/** SAT of one row with some rays left out of the least squares (κ = 0, the same regulariser as satFactor). */
export function satMasked(L: Mat, B: Mat, lambda: number, p: ArrayLike<number>, drop: ArrayLike<number>): Float64Array {
  const keep: number[] = [];
  for (let i = 0; i < L.n; i++) if (!drop[i]) keep.push(i);
  const Ls: Mat = { n: keep.length, m: L.m, a: new Float64Array(keep.length * L.m) };
  keep.forEach((i, k) => Ls.a.set(L.a.subarray(i * L.m, (i + 1) * L.m), k * L.m));
  return satApply(satFactor(Ls, B, lambda), keep.map((i) => p[i]));
}

/** Invert P[channel][row] (each of length n) with spacing dr (metres for SI). */
export function invertAll(P: Float64Array[][], dr: number, method: AbelMethod, params: AbelParams = {}): Inversion {
  const n = P[0][0].length;
  const L = chordMatrixHalf(n, dr);
  let solve: (p: Float64Array, row: number) => Float64Array;
  if (method === 'onion') solve = (p) => solveUpper(L, p);
  else if (method === 'tikhonov') {
    const lr = params.lambdaRel ?? 1e-3;
    if (!(lr >= LAMBDA_REL_MIN && lr <= LAMBDA_REL_MAX)) throw new Error('lambda_out_of_range');
    const F = tikhonovFactor(L, lr, 'gamma2');
    solve = (p) => tikhonovApply(F, L, p);
  } else {
    const B = splineBasis(n, params.segments ?? 12), lam = params.satLambda ?? 1e-4;
    const F = satFactor(L, B, lam);
    solve = (p, row) => { const m = params.rayMask?.get(row); return m && Array.from(m).some(Boolean) ? satMasked(L, B, lam, p, m) : satApply(F, p); };
  }
  const ex = new Set(Array.from(params.excludeRows ?? []));
  const nan = () => new Float64Array(n).fill(NaN);
  const J = P.map((ch) => ch.map((p, r) => (ex.has(r) ? nan() : solve(p, r))));
  const reproj = J.map((ch) => ch.map((f, r) => (ex.has(r) ? nan() : matvec(L, f))));
  const resid = reproj.map((ch, c) => ch.map((q, r) => q.map((v, i) => P[c][r][i] - v)));
  return { method, reg: method === 'tikhonov' ? 'gamma2' : null, L, J, reproj, resid, params, excluded: [...ex].sort((a, b) => a - b) };
}

/**
 * Over-regularisation check (GB2r1-numerico-08). True when the residual norm of a row exceeds `factor`
 * times what the noise allows, √n σ, with σ the noise of one ray (from the difference between halves
 * or between frames). With λ_rel large Tikhonov with Γ = gamma2 drives j to zero in silence.
 */
export function overRegularised(resid: ArrayLike<number>, sigma: number, factor = 3): boolean {
  let s = 0, n = 0;
  for (let i = 0; i < resid.length; i++) if (Number.isFinite(resid[i])) { s += resid[i] * resid[i]; n++; }
  return Math.sqrt(s) > factor * Math.sqrt(n) * sigma;
}

/** Noise of one ray from the two halves, σ ≈ rms(right − left)/√2. */
export function halvesSigma(right: ArrayLike<number>, left: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < right.length; i++) s += (right[i] - left[i]) ** 2;
  return Math.sqrt(s / right.length / 2);
}

/**
 * L-curve corner as lcurve_corner of prepare_data.py. ρ = ‖L f − P‖, η = ‖Γ f‖ (η² = fᵀΓᵀΓf), on the
 * rows given, curvature of (ln ρ, ln η) against ln λ with numpy.gradient, the maximum over the interior
 * points. One factorisation per λ.
 */
export function lcurveCorner(L: Mat, rows: Float64Array[], lambdas: number[], reg: TikhonovReg = 'gamma2'): { lambda: number; index: number; rho: number[]; eta: number[]; kappa: number[] } {
  const pts = lcurve(L, rows, lambdas, reg);
  const rho = pts.map((p) => p.res), eta = pts.map((p) => p.rough);
  const t = lambdas.map(Math.log), x = rho.map(Math.log), y = eta.map(Math.log);
  const dx = gradient(x, t), dy = gradient(y, t), ddx = gradient(dx, t), ddy = gradient(dy, t);
  const kappa = dx.map((_, i) => (dx[i] * ddy[i] - dy[i] * ddx[i]) / (dx[i] ** 2 + dy[i] ** 2) ** 1.5);
  let k = 1;
  for (let i = 2; i < kappa.length - 1; i++) if (kappa[i] > kappa[k] || Number.isNaN(kappa[k])) k = i;
  return { lambda: lambdas[k], index: k, rho, eta, kappa: Array.from(kappa) };
}

/** logspace(a, b, n) as numpy (10 ** linspace). */
export function logspace(a: number, b: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => 10 ** (n === 1 ? a : a + ((b - a) * i) / (n - 1)));
}

/** Roughness Gram of a Tikhonov regulariser, re-exported for the page. */
export const roughnessGram = regGram;

/** Fold a full-width row (2n columns, axis between columns n−1 and n) into right and mirrored left halves. */
export function splitHalves(row: ArrayLike<number>): { right: Float64Array; left: Float64Array } {
  const nx = row.length;
  if (nx % 2) throw new Error('odd_width');
  const n = nx / 2;
  const right = new Float64Array(n), left = new Float64Array(n);
  for (let i = 0; i < n; i++) { right[i] = row[n + i]; left[i] = row[n - 1 - i]; }
  return { right, left };
}
