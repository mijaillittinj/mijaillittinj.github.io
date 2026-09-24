/**
 * Singular value decomposition for small dense matrices (one-sided Jacobi, Hestenes).
 * Accurate to machine precision relative to each singular value, which matters here:
 * the point of the SVD module is to show singular values spanning many decades.
 *
 * A (n × n, row-major) = U Σ Vᵀ; singular values sorted in decreasing order.
 */
import { zeros, type Mat } from './linalg.ts';

export interface SVD { U: Float64Array; s: Float64Array; V: Float64Array; n: number }

export function svd(A: Mat, sweeps = 60): SVD {
  const n = A.n;
  if (A.m !== n) throw new Error('square matrices only');
  // work on columns: W = A (copy), V = I
  const W = Float64Array.from(A.a);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      let a = 0, b = 0, c = 0;
      for (let k = 0; k < n; k++) {
        const wp = W[k * n + p], wq = W[k * n + q];
        a += wp * wp; b += wq * wq; c += wp * wq;
      }
      if (Math.abs(c) <= 1e-15 * Math.sqrt(a * b) || c === 0) continue;
      off = Math.max(off, Math.abs(c) / Math.sqrt(a * b));
      const zeta = (b - a) / (2 * c);
      const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
      const cs = 1 / Math.sqrt(1 + t * t), sn = cs * t;
      for (let k = 0; k < n; k++) {
        const wp = W[k * n + p], wq = W[k * n + q];
        W[k * n + p] = cs * wp - sn * wq; W[k * n + q] = sn * wp + cs * wq;
        const vp = V[k * n + p], vq = V[k * n + q];
        V[k * n + p] = cs * vp - sn * vq; V[k * n + q] = sn * vp + cs * vq;
      }
    }
    if (off < 1e-14) break;
  }
  // singular values = column norms; U = W / s
  const s = new Float64Array(n);
  for (let j = 0; j < n; j++) { let t = 0; for (let k = 0; k < n; k++) t += W[k * n + j] ** 2; s[j] = Math.sqrt(t); }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => s[j] - s[i]);
  const U = new Float64Array(n * n), Vs = new Float64Array(n * n), ss = new Float64Array(n);
  order.forEach((j, col) => {
    ss[col] = s[j];
    for (let k = 0; k < n; k++) {
      U[k * n + col] = s[j] > 0 ? W[k * n + j] / s[j] : 0;
      Vs[k * n + col] = V[k * n + j];
    }
  });
  return { U, s: ss, V: Vs, n };
}

/** Coefficients uᵢᵀ g (the "Picard" coefficients). */
export function uTg(d: SVD, g: ArrayLike<number>): Float64Array {
  const { n, U } = d, out = new Float64Array(n);
  for (let i = 0; i < n; i++) { let t = 0; for (let k = 0; k < n; k++) t += U[k * n + i] * g[k]; out[i] = t; }
  return out;
}

/** Filtered solution x = Σ φᵢ (uᵢᵀg / σᵢ) vᵢ. */
export function filtered(d: SVD, beta: ArrayLike<number>, phi: (i: number, s: number) => number): Float64Array {
  const { n, V, s } = d, x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const f = phi(i, s[i]);
    if (f === 0 || s[i] === 0) continue;
    const c = (f * beta[i]) / s[i];
    for (let k = 0; k < n; k++) x[k] += c * V[k * n + i];
  }
  return x;
}

export const tsvd = (d: SVD, beta: ArrayLike<number>, k: number) => filtered(d, beta, (i) => (i < k ? 1 : 0));
export const tikhonovSVD = (d: SVD, beta: ArrayLike<number>, lam: number) => filtered(d, beta, (_, s) => (s * s) / (s * s + lam * lam));

/**
 * L-curve for Tikhonov: residual norm ‖A x_λ − g‖ and solution norm ‖x_λ‖ for each λ.
 * With the SVD both are closed-form sums over the filter factors.
 */
export function lcurve(d: SVD, beta: ArrayLike<number>, gNorm2: number, lams: ArrayLike<number>) {
  const res: number[] = [], sol: number[] = [];
  let inRange = 0;
  for (let i = 0; i < d.n; i++) inRange += beta[i] * beta[i];
  const outside = Math.max(gNorm2 - inRange, 0); // part of g outside range(A) (zero for square full-rank A)
  for (const lam of Array.from(lams)) {
    let r = outside, x = 0;
    for (let i = 0; i < d.n; i++) {
      const s = d.s[i], f = (s * s) / (s * s + lam * lam);
      r += ((1 - f) * beta[i]) ** 2;
      if (s > 0) x += ((f * beta[i]) / s) ** 2;
    }
    res.push(Math.sqrt(r)); sol.push(Math.sqrt(x));
  }
  return { res, sol };
}

/** Gaussian blur of width w on n points (a smoothing, severely ill-conditioned operator). */
export function blurOperator(n: number, w: number): Mat {
  const A = zeros(n, n), h = 1 / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const d = (i - j) * h;
    A.a[i * n + j] = (h / (Math.sqrt(2 * Math.PI) * w)) * Math.exp(-(d * d) / (2 * w * w));
  }
  return A;
}

/** Test signal: a box, a ramp and a narrow bump, on [0, 1]. */
export function testSignal(n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, i) => {
    const x = (i + 0.5) / n;
    let v = 0;
    if (x > 0.12 && x < 0.32) v += 1;
    if (x > 0.42 && x < 0.66) v += (x - 0.42) / 0.24 * 0.8;
    v += 0.9 * Math.exp(-(((x - 0.8) / 0.025) ** 2));
    return v;
  });
}
