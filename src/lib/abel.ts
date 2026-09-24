/**
 * Discrete Abel transform for axisymmetric fields.
 *
 * The field f(r) is piecewise constant on N annuli of width dr (onion-peeling
 * discretisation). The line-of-sight projection at lateral offset y_i = i·dr is
 *     P(y_i) = Σ_j L_ij f_j,
 * where L_ij is the exact chord length of the ray through annulus j:
 *     L_ij = 2 ( sqrt(r_{j+1}² − y_i²) − sqrt(max(r_j, y_i)² − y_i²) ),  j ≥ i.
 * L is upper triangular, so it can be inverted directly by back substitution
 * ("onion peeling"). That direct inversion is exact without noise and amplifies
 * noise near the axis; the regularised variant adds a second-derivative penalty,
 * the same smoothness idea as the spline-based Abel transform (SAT, Littin et al.,
 * Fuel 2024).
 */
import { zeros, matvec, gram, tvec, cholSolve, solveUpper, type Mat } from './linalg.ts';

export function chordMatrix(n: number, R = 1): Mat {
  const dr = R / n;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    const y = i * dr; // ray offset at the inner edge of annulus i
    const y2 = y * y;
    for (let j = i; j < n; j++) {
      const r0 = Math.max(j * dr, y);
      const r1 = (j + 1) * dr;
      L.a[i * n + j] = 2 * (Math.sqrt(Math.max(r1 * r1 - y2, 0)) - Math.sqrt(Math.max(r0 * r0 - y2, 0)));
    }
  }
  return L;
}

export function project(L: Mat, f: ArrayLike<number>): Float64Array {
  return matvec(L, f);
}

/** Direct (onion-peeling) inversion. */
export function onionPeel(L: Mat, p: ArrayLike<number>): Float64Array {
  return solveUpper(L, p);
}

/** Second-difference operator D (n−2 × n) with a symmetry row at the axis. */
function secondDiffGram(n: number): Mat {
  // Build DᵀD directly. Rows: f[k-1] - 2 f[k] + f[k+1] for k=1..n-2, plus
  // an axis-symmetry row f[1] - f[0] (zero slope at r = 0).
  const G = zeros(n, n);
  const add = (i: number, j: number, v: number) => (G.a[i * n + j] += v);
  for (let k = 1; k < n - 1; k++) {
    const idx = [k - 1, k, k + 1];
    const w = [1, -2, 1];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) add(idx[a], idx[b], w[a] * w[b]);
  }
  add(0, 0, 1); add(1, 1, 1); add(0, 1, -1); add(1, 0, -1);
  return G;
}

/**
 * Tikhonov-regularised inversion:  min ‖L f − p‖² + λ ‖D f‖².
 * λ is given relative to trace(LᵀL)/n so the slider is scale-free.
 */
export function tikhonov(L: Mat, p: ArrayLike<number>, lambdaRel: number): Float64Array {
  const n = L.m;
  const A = gram(L);
  let tr = 0;
  for (let i = 0; i < n; i++) tr += A.a[i * n + i];
  const lam = lambdaRel * (tr / n);
  const D = secondDiffGram(n);
  for (let i = 0; i < n * n; i++) A.a[i] += lam * D.a[i];
  const b = tvec(L, p);
  return cholSolve(A, b) ?? new Float64Array(n);
}

/** Residual norm ‖L f − p‖ and roughness ‖D f‖ for L-curve display. */
export function lcurvePoint(L: Mat, f: ArrayLike<number>, p: ArrayLike<number>): [number, number] {
  const q = matvec(L, f);
  let res = 0;
  for (let i = 0; i < q.length; i++) res += (q[i] - p[i]) ** 2;
  let rough = 0;
  for (let k = 1; k < f.length - 1; k++) rough += (f[k - 1] - 2 * f[k] + f[k + 1]) ** 2;
  return [Math.sqrt(res), Math.sqrt(rough)];
}

/** Example radial profiles on r ∈ [0, 1]. Parameters are 0..1 sliders. */
export type ProfileKind = 'gaussian' | 'annular' | 'two-zone';
export function profile(kind: ProfileKind, r: number, a: number, b: number): number {
  switch (kind) {
    case 'gaussian': {
      const w = 0.12 + 0.35 * a;
      return Math.exp(-(r * r) / (2 * w * w));
    }
    case 'annular': {
      // off-axis maximum, typical of the "wings" of a diffusion flame
      const r0 = 0.25 + 0.45 * a;
      const w = 0.05 + 0.15 * b;
      return Math.exp(-((r - r0) ** 2) / (2 * w * w)) + 0.15 * Math.exp(-(r * r) / 0.04);
    }
    case 'two-zone': {
      const edge = 0.3 + 0.4 * a;
      const s = 0.02 + 0.08 * b;
      const core = 1 / (1 + Math.exp((r - edge) / s));
      return 0.35 * core + 0.65 * Math.exp(-((r - edge) ** 2) / (2 * 0.06 ** 2));
    }
  }
}
