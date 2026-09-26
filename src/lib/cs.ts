/**
 * Compressed sensing in the browser.
 *
 * A signal x of length N is sparse in the DCT basis: x = Ψ c with only K non-zero
 * coefficients (a sum of K cosine "tones"). We observe M ≪ N samples at random times,
 * y = S x + ε = Φ c + ε with Φ = S Ψ (M rows of the orthonormal DCT matrix).
 *  - Minimum-norm least squares: c = Φᵀ(ΦΦᵀ)⁻¹ y = Φᵀ y (the rows are orthonormal), which
 *    puts zeros at every unobserved time: the "obvious" linear answer fails.
 *  - L1 minimisation, min ½‖Φc − y‖² + λ‖c‖₁, solved with FISTA; then an optional
 *    least-squares "debias" step on the detected support.
 *  - Orthogonal matching pursuit (greedy), used for the phase-transition map.
 */
import { rng } from './linalg.ts';

/** Orthonormal DCT-II synthesis matrix Ψ (N × N, row-major): x_t = Σ_k Ψ[t][k] c_k. */
export function dctMatrix(N: number): Float64Array {
  const P = new Float64Array(N * N);
  for (let t = 0; t < N; t++) for (let k = 0; k < N; k++) {
    const a = k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
    P[t * N + k] = a * Math.cos((Math.PI * (t + 0.5) * k) / N);
  }
  return P;
}

export interface Problem {
  N: number; K: number; M: number;
  c: Float64Array;      // true coefficients (K-sparse)
  x: Float64Array;      // true signal
  idx: Int32Array;      // sampled times (sorted)
  Phi: Float64Array;    // M × N
  y: Float64Array;      // noisy samples
}

/** Random K-sparse problem with M random samples and noise σ = level × max|x|. */
export function makeProblem(Psi: Float64Array, N: number, K: number, M: number, level: number, seed: number): Problem {
  const R = rng(seed);
  // support: K distinct frequencies in 1..N/2 (avoid the constant term), amplitudes ±(0.5–1)
  const c = new Float64Array(N);
  const pool = Array.from({ length: Math.floor(N / 2) - 1 }, (_, i) => i + 1);
  for (let j = 0; j < K && pool.length; j++) {
    const k = pool.splice(Math.floor(R.uni() * pool.length), 1)[0];
    c[k] = (R.uni() < 0.5 ? -1 : 1) * (0.5 + 0.5 * R.uni());
  }
  const x = new Float64Array(N);
  for (let t = 0; t < N; t++) { let s = 0; for (let k = 0; k < N; k++) s += Psi[t * N + k] * c[k]; x[t] = s; }
  const times = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) { const j = Math.floor(R.uni() * (i + 1)); [times[i], times[j]] = [times[j], times[i]]; }
  const idx = Int32Array.from(times.slice(0, M).sort((a, b) => a - b));
  const Phi = new Float64Array(M * N);
  for (let r = 0; r < M; r++) Phi.set(Psi.subarray(idx[r] * N, (idx[r] + 1) * N), r * N);
  let mx = 0;
  for (const v of x) mx = Math.max(mx, Math.abs(v));
  const y = Float64Array.from(idx, (t) => x[t] + level * mx * R.normal());
  return { N, K, M, c, x, idx, Phi, y };
}

export function synth(Psi: Float64Array, N: number, c: ArrayLike<number>): Float64Array {
  const x = new Float64Array(N);
  for (let t = 0; t < N; t++) { let s = 0; for (let k = 0; k < N; k++) s += Psi[t * N + k] * c[k]; x[t] = s; }
  return x;
}

const PhiT = (Phi: Float64Array, M: number, N: number, r: ArrayLike<number>) => {
  const g = new Float64Array(N);
  for (let i = 0; i < M; i++) { const v = r[i]; if (v) for (let k = 0; k < N; k++) g[k] += Phi[i * N + k] * v; }
  return g;
};
const PhiX = (Phi: Float64Array, M: number, N: number, c: ArrayLike<number>) => {
  const y = new Float64Array(M);
  for (let i = 0; i < M; i++) { let s = 0; for (let k = 0; k < N; k++) s += Phi[i * N + k] * c[k]; y[i] = s; }
  return y;
};

/** Minimum-norm least-squares coefficients (Φ has orthonormal rows, so c = Φᵀ y). */
export function minNorm(p: Problem): Float64Array {
  return PhiT(p.Phi, p.M, p.N, p.y);
}

/** FISTA for min ½‖Φc − y‖² + λ‖c‖₁ (step 1, since ‖Φ‖ ≤ 1). Returns a stepper. */
export function makeFista(p: Problem, lambda: number) {
  const { N, M, Phi, y } = p;
  let c: Float64Array = new Float64Array(N), z: Float64Array = new Float64Array(N), t = 1, iter = 0;
  const soft = (v: number) => (v > lambda ? v - lambda : v < -lambda ? v + lambda : 0);
  return {
    get c() { return c; },
    get iter() { return iter; },
    step() {
      const r = PhiX(Phi, M, N, z);
      for (let i = 0; i < M; i++) r[i] -= y[i];
      const g = PhiT(Phi, M, N, r);
      const cn = new Float64Array(N);
      for (let k = 0; k < N; k++) cn[k] = soft(z[k] - g[k]);
      const tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
      for (let k = 0; k < N; k++) z[k] = cn[k] + ((t - 1) / tn) * (cn[k] - c[k]);
      c = cn; t = tn; iter++;
    },
  };
}

/** Least squares restricted to a support (small normal equations, Gaussian elimination). */
function lsOnSupport(p: Problem, S: number[]): Float64Array {
  const { N, M, Phi, y } = p, k = S.length;
  const A = new Float64Array(k * k), b = new Float64Array(k);
  for (let a = 0; a < k; a++) {
    for (let i = 0; i < M; i++) b[a] += Phi[i * N + S[a]] * y[i];
    for (let c2 = 0; c2 < k; c2++) { let s = 0; for (let i = 0; i < M; i++) s += Phi[i * N + S[a]] * Phi[i * N + S[c2]]; A[a * k + c2] = s; }
    A[a * k + a] += 1e-10;
  }
  // solve A u = b
  for (let col = 0; col < k; col++) {
    let piv = col;
    for (let r = col + 1; r < k; r++) if (Math.abs(A[r * k + col]) > Math.abs(A[piv * k + col])) piv = r;
    if (piv !== col) { for (let j = 0; j < k; j++) [A[col * k + j], A[piv * k + j]] = [A[piv * k + j], A[col * k + j]]; [b[col], b[piv]] = [b[piv], b[col]]; }
    const d = A[col * k + col] || 1e-12;
    for (let r = col + 1; r < k; r++) { const f = A[r * k + col] / d; if (f) { for (let j = col; j < k; j++) A[r * k + j] -= f * A[col * k + j]; b[r] -= f * b[col]; } }
  }
  const u = new Float64Array(k);
  for (let r = k - 1; r >= 0; r--) { let s = b[r]; for (let j = r + 1; j < k; j++) s -= A[r * k + j] * u[j]; u[r] = s / (A[r * k + r] || 1e-12); }
  const c = new Float64Array(N);
  S.forEach((s, a) => (c[s] = u[a]));
  return c;
}

/** Debias: keep the support of c (|c_k| > tol), refit it by least squares. */
export function debias(p: Problem, c: ArrayLike<number>, tol = 1e-3): Float64Array {
  const S: number[] = [];
  for (let k = 0; k < p.N; k++) if (Math.abs(c[k]) > tol) S.push(k);
  if (S.length === 0 || S.length >= p.M) return Float64Array.from(c);
  return lsOnSupport(p, S);
}

/** Orthogonal matching pursuit with K steps. */
export function omp(p: Problem, K: number): Float64Array {
  const { N, M, Phi, y } = p;
  const S: number[] = [];
  let c: Float64Array = new Float64Array(N), r: Float64Array = Float64Array.from(y);
  for (let step = 0; step < Math.min(K, M); step++) {
    const g = PhiT(Phi, M, N, r);
    let best = -1, bv = -1;
    for (let k = 0; k < N; k++) if (!S.includes(k) && Math.abs(g[k]) > bv) { bv = Math.abs(g[k]); best = k; }
    S.push(best);
    c = lsOnSupport(p, S);
    const yh = PhiX(Phi, M, N, c);
    for (let i = 0; i < M; i++) r[i] = y[i] - yh[i];
  }
  return c;
}

export function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let e = 0, n = 0;
  for (let i = 0; i < a.length; i++) { e += (a[i] - b[i]) ** 2; n += b[i] ** 2; }
  return Math.sqrt(e / (n || 1));
}

/**
 * Success rate of OMP (noise-free) on a grid of sparsity K and measurements M:
 * the phase transition of compressed sensing. Returns a stepper computing one cell per call.
 */
export function makePhaseMap(Psi: Float64Array, N: number, Ks: number[], Ms: number[], trials: number) {
  const rate = new Float64Array(Ks.length * Ms.length);
  let cell = 0;
  return {
    rate,
    get done() { return cell >= rate.length; },
    get progress() { return cell / rate.length; },
    step() {
      if (cell >= rate.length) return;
      const ki = Math.floor(cell / Ms.length), mi = cell % Ms.length;
      const K = Ks[ki], M = Ms[mi];
      let ok = 0;
      if (M > K) for (let tr = 0; tr < trials; tr++) {
        const p = makeProblem(Psi, N, K, M, 0, 1000 + 97 * tr + 13 * K + M);
        if (relErr(omp(p, K), p.c) < 1e-3) ok++;
      }
      rate[cell] = ok / trials;
      cell++;
    },
  };
}
