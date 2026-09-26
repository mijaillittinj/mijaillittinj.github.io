/**
 * Gaussian-process regression in one dimension, small and dependency-free.
 *
 * Prior f ~ GP(0, k), data y_i = f(x_i) + ε_i with ε_i ~ N(0, σ_n²). Kernels:
 *   RBF          k(r) = σ_f² exp(−r² / 2ℓ²)
 *   Matérn 5/2   k(r) = σ_f² (1 + √5 r/ℓ + 5r²/3ℓ²) exp(−√5 r/ℓ)
 * Hyperparameters are handled in log form, θ = (log ℓ, log σ_f², log σ_n²), and fitted by
 * gradient ascent on the log marginal likelihood
 *   log p(y) = −½ yᵀ K⁻¹ y − ½ log|K| − (n/2) log 2π,   K = K_f + σ_n² I,
 * with gradient ½ tr((α αᵀ − K⁻¹) ∂K/∂θ), α = K⁻¹ y.
 */
import { rng } from './linalg.ts';

export type KernelKind = 'rbf' | 'matern52';
export interface Hyper { logEll: number; logSf2: number; logSn2: number }

/** Kernel value and its derivatives with respect to log ℓ and log σ_f². */
export function kernel(kind: KernelKind, a: number, b: number, h: Hyper) {
  const ell = Math.exp(h.logEll), sf2 = Math.exp(h.logSf2);
  const r = Math.abs(a - b);
  if (kind === 'rbf') {
    const e = Math.exp(-(r * r) / (2 * ell * ell));
    const k = sf2 * e;
    return { k, dEll: k * (r * r) / (ell * ell), dSf2: k };
  }
  const s = (Math.sqrt(5) * r) / ell;
  const e = Math.exp(-s);
  const k = sf2 * (1 + s + (s * s) / 3) * e;
  // d k / d log ℓ = σ_f² e (s² (1 + s) / 3)
  return { k, dEll: sf2 * e * (s * s * (1 + s)) / 3, dSf2: k };
}

/** Lower Cholesky factor of a symmetric positive-definite n×n matrix (row-major). */
export function cholesky(A: Float64Array, n: number): Float64Array | null {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) { if (s <= 0) return null; L[i * n + i] = Math.sqrt(s); }
      else L[i * n + j] = s / L[j * n + j];
    }
  }
  return L;
}

/** Solve L Lᵀ x = b. */
export function cholSolveL(L: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}

function gram(kind: KernelKind, xs: ArrayLike<number>, h: Hyper) {
  const n = xs.length, sn2 = Math.exp(h.logSn2);
  const K = new Float64Array(n * n), dEll = new Float64Array(n * n), dSf2 = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const kv = kernel(kind, xs[i], xs[j], h);
    K[i * n + j] = kv.k + (i === j ? sn2 + 1e-9 : 0);
    dEll[i * n + j] = kv.dEll; dSf2[i * n + j] = kv.dSf2;
  }
  return { K, dEll, dSf2, sn2 };
}

/** Posterior mean and standard deviation of f at the query points. */
export function posterior(kind: KernelKind, xs: ArrayLike<number>, ys: ArrayLike<number>, xq: ArrayLike<number>, h: Hyper) {
  const n = xs.length, m = xq.length;
  const mean = new Float64Array(m), sd = new Float64Array(m);
  const sf2 = Math.exp(h.logSf2);
  if (n === 0) { sd.fill(Math.sqrt(sf2)); return { mean, sd }; }
  const { K } = gram(kind, xs, h);
  const L = cholesky(K, n);
  if (!L) { sd.fill(Math.sqrt(sf2)); return { mean, sd }; }
  const alpha = cholSolveL(L, n, ys);
  const ks = new Float64Array(n);
  for (let q = 0; q < m; q++) {
    let mu = 0;
    for (let i = 0; i < n; i++) { ks[i] = kernel(kind, xq[q], xs[i], h).k; mu += ks[i] * alpha[i]; }
    // v = L⁻¹ k*, var = k** − vᵀv
    let vv = 0;
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = ks[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * v[k]; v[i] = s / L[i * n + i]; vv += v[i] * v[i]; }
    mean[q] = mu;
    sd[q] = Math.sqrt(Math.max(sf2 - vv, 1e-12));
  }
  return { mean, sd };
}

/** Full posterior covariance of f on a grid (for drawing samples). */
export function posteriorCov(kind: KernelKind, xs: ArrayLike<number>, ys: ArrayLike<number>, xq: ArrayLike<number>, h: Hyper) {
  void ys; // the posterior covariance does not depend on the measured values
  const n = xs.length, m = xq.length;
  const C = new Float64Array(m * m);
  for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) C[a * m + b] = kernel(kind, xq[a], xq[b], h).k;
  if (n > 0) {
    const { K } = gram(kind, xs, h);
    const L = cholesky(K, n);
    if (L) {
      // V = L⁻¹ K*ᵀ (n × m), C −= Vᵀ V
      const V = new Float64Array(n * m);
      for (let q = 0; q < m; q++) for (let i = 0; i < n; i++) {
        let s = kernel(kind, xq[q], xs[i], h).k;
        for (let k = 0; k < i; k++) s -= L[i * n + k] * V[k * m + q];
        V[i * m + q] = s / L[i * n + i];
      }
      for (let a = 0; a < m; a++) for (let b = a; b < m; b++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += V[i * m + a] * V[i * m + b];
        C[a * m + b] -= s; C[b * m + a] = C[a * m + b];
      }
    }
  }
  return C;
}

/** Log marginal likelihood and its gradient with respect to (log ℓ, log σ_f², log σ_n²). */
export function logMarginal(kind: KernelKind, xs: ArrayLike<number>, ys: ArrayLike<number>, h: Hyper) {
  const n = xs.length;
  if (n === 0) return { lml: 0, grad: [0, 0, 0] as [number, number, number] };
  const { K, dEll, dSf2, sn2 } = gram(kind, xs, h);
  const L = cholesky(K, n);
  if (!L) return { lml: -Infinity, grad: [0, 0, 0] as [number, number, number] };
  const alpha = cholSolveL(L, n, ys);
  let logDet = 0, yAy = 0;
  for (let i = 0; i < n; i++) { logDet += 2 * Math.log(L[i * n + i]); yAy += ys[i] * alpha[i]; }
  const lml = -0.5 * yAy - 0.5 * logDet - 0.5 * n * Math.log(2 * Math.PI);
  // Kinv column by column
  const Kinv = new Float64Array(n * n);
  const e = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    e.fill(0); e[j] = 1;
    const c = cholSolveL(L, n, e);
    for (let i = 0; i < n; i++) Kinv[i * n + j] = c[i];
  }
  let gE = 0, gF = 0, gN = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const W = alpha[i] * alpha[j] - Kinv[i * n + j];
    gE += W * dEll[j * n + i]; gF += W * dSf2[j * n + i];
    if (i === j) gN += W * sn2;
  }
  return { lml, grad: [0.5 * gE, 0.5 * gF, 0.5 * gN] as [number, number, number] };
}

/** Bounds keep the fit away from degenerate corners (ℓ → 0 interpolates the noise). */
export const bounds = { logEll: [Math.log(0.15), Math.log(8)], logSf2: [Math.log(0.05), Math.log(20)], logSn2: [Math.log(1e-4), Math.log(2)] } as const;

/** One Adam step of gradient ascent on the log marginal likelihood (mutates h and state). */
export function fitStep(kind: KernelKind, xs: ArrayLike<number>, ys: ArrayLike<number>, h: Hyper, st: { m: number[]; v: number[]; t: number }, lr = 0.05) {
  const { lml, grad } = logMarginal(kind, xs, ys, h);
  st.t++;
  const keys = ['logEll', 'logSf2', 'logSn2'] as const;
  keys.forEach((key, i) => {
    st.m[i] = 0.9 * st.m[i] + 0.1 * grad[i];
    st.v[i] = 0.999 * st.v[i] + 0.001 * grad[i] * grad[i];
    const mh = st.m[i] / (1 - 0.9 ** st.t), vh = st.v[i] / (1 - 0.999 ** st.t);
    const nv = h[key] + (lr * mh) / (Math.sqrt(vh) + 1e-8);
    h[key] = Math.min(bounds[key][1], Math.max(bounds[key][0], nv));
  });
  return lml;
}

/** The hidden function that generates measurements in the lab, on x ∈ [0, 10]. */
export const truth = (x: number) => Math.sin(0.9 * x) + 0.35 * Math.sin(2.7 * x + 0.4);

/** Initial measurements: a few noisy samples, clustered, leaving gaps. */
export function initialData(seed = 3, noise = 0.1) {
  const R = rng(seed);
  const xs = [0.8, 1.6, 2.2, 5.1, 5.9, 8.4, 9.1];
  return { xs, ys: xs.map((x) => truth(x) + noise * R.normal()) };
}
