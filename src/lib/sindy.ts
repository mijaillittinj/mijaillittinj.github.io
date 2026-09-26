/**
 * Sparse identification of nonlinear dynamics (SINDy; Brunton, Proctor & Kutz, PNAS 113, 2016).
 *
 * From a sampled trajectory X(t) of a dynamical system dx/dt = f(x):
 *   1. estimate the derivatives dX/dt from the (noisy) samples;
 *   2. evaluate a library Θ(X) of candidate terms (1, x, y, x², xy, sin x, …);
 *   3. find a sparse coefficient matrix Ξ with dX/dt ≈ Θ(X) Ξ by sequentially
 *      thresholded least squares (STLSQ): fit, set |ξ| < λ to zero, refit, repeat.
 * The surviving terms are the discovered equations.
 */
import { rng } from './linalg.ts';

export type SystemId = 'oscillator' | 'pendulum' | 'lotka' | 'lorenz';

export interface Term { name: string; html: string; f: (s: ArrayLike<number>) => number }

export interface SystemDef {
  id: SystemId;
  vars: string[];                 // state names, e.g. ['x', 'y']
  rhs: (s: ArrayLike<number>) => number[];
  x0: number[];                   // initial state at amplitude 1
  dt: number;                     // sampling interval
  tMax: number;                   // longest record
  library: Term[];
  truth: number[][];              // true coefficients, [equation][term]
}

const v = (name: string) => `<i>${name}</i>`;

/** Library of polynomials up to degree 2 in the given variables, plus optionally sin of the first. */
function polyLibrary(vars: string[], trig: boolean): Term[] {
  const n = vars.length;
  const terms: Term[] = [{ name: '1', html: '1', f: () => 1 }];
  for (let i = 0; i < n; i++) terms.push({ name: vars[i], html: v(vars[i]), f: (s) => s[i] });
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
    const name = i === j ? `${vars[i]}²` : `${vars[i]}${vars[j]}`;
    const html = i === j ? `${v(vars[i])}<sup>2</sup>` : `${v(vars[i])}${v(vars[j])}`;
    terms.push({ name, html, f: (s) => s[i] * s[j] });
  }
  if (trig) terms.push({ name: `sin ${vars[0]}`, html: `sin ${v(vars[0])}`, f: (s) => Math.sin(s[0]) });
  return terms;
}

/** Place the coefficients {term name: value} of one equation in library order. */
function row(lib: Term[], coef: Record<string, number>): number[] {
  return lib.map((t) => coef[t.name] ?? 0);
}

export function system(id: SystemId): SystemDef {
  switch (id) {
    case 'oscillator': {
      const lib = polyLibrary(['x', 'y'], false);
      return {
        id, vars: ['x', 'y'], library: lib,
        rhs: (s) => [s[1], -4 * s[0] - 0.4 * s[1]],
        x0: [2, 0], dt: 0.05, tMax: 20,
        truth: [row(lib, { y: 1 }), row(lib, { x: -4, y: -0.4 })],
      };
    }
    case 'pendulum': {
      const lib = polyLibrary(['x', 'y'], true);
      return {
        id, vars: ['x', 'y'], library: lib,
        rhs: (s) => [s[1], -4 * Math.sin(s[0]) - 0.3 * s[1]],
        x0: [2.6, 0], dt: 0.05, tMax: 20,
        truth: [row(lib, { y: 1 }), row(lib, { 'sin x': -4, y: -0.3 })],
      };
    }
    case 'lotka': {
      const lib = polyLibrary(['x', 'y'], false);
      return {
        id, vars: ['x', 'y'], library: lib,
        rhs: (s) => [1.0 * s[0] - 0.5 * s[0] * s[1], 0.2 * s[0] * s[1] - 0.6 * s[1]],
        x0: [4, 1], dt: 0.1, tMax: 40,
        truth: [row(lib, { x: 1, xy: -0.5 }), row(lib, { xy: 0.2, y: -0.6 })],
      };
    }
    case 'lorenz': {
      const lib = polyLibrary(['x', 'y', 'z'], false);
      return {
        id, vars: ['x', 'y', 'z'], library: lib,
        rhs: (s) => [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - (8 / 3) * s[2]],
        x0: [-8, 7, 27], dt: 0.01, tMax: 10,
        truth: [row(lib, { x: -10, y: 10 }), row(lib, { x: 28, y: -1, xz: -1 }), row(lib, { xy: 1, z: -8 / 3 })],
      };
    }
  }
}

/** Classical RK4 step. */
function rk4(f: (s: number[]) => number[], s: number[], h: number): number[] {
  const k1 = f(s);
  const k2 = f(s.map((x, i) => x + (h / 2) * k1[i]));
  const k3 = f(s.map((x, i) => x + (h / 2) * k2[i]));
  const k4 = f(s.map((x, i) => x + h * k3[i]));
  return s.map((x, i) => x + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

/**
 * Integrate ds/dt = f(s) from s0 and return nSamples states spaced by dt (sub-stepped for accuracy).
 * Stops early (NaN-padded) if the state leaves a large box: a wrong model can blow up.
 */
export function simulate(f: (s: number[]) => number[], s0: number[], dt: number, nSamples: number, sub = 10): number[][] {
  const out: number[][] = [];
  let s = s0.slice();
  const h = dt / sub;
  for (let k = 0; k < nSamples; k++) {
    out.push(s.slice());
    for (let q = 0; q < sub; q++) s = rk4(f, s, h);
    if (!s.every((x) => Number.isFinite(x) && Math.abs(x) < 1e4)) {
      while (out.length < nSamples) out.push(s0.map(() => NaN));
      break;
    }
  }
  return out;
}

export interface Dataset { t: number[]; clean: number[][]; noisy: number[][] }

/** Trajectory of the true system with Gaussian noise of relative size `noise` (fraction of each state's std). */
export function makeData(sys: SystemDef, duration: number, noise: number, amplitude = 1, seed = 1): Dataset {
  const n = Math.max(20, Math.round(duration / sys.dt));
  const x0 = sys.x0.map((x) => (sys.id === 'lotka' || sys.id === 'lorenz' ? x : x * amplitude));
  const clean = simulate(sys.rhs, x0, sys.dt, n);
  const d = sys.vars.length, R = rng(seed);
  const std = Array.from({ length: d }, (_, j) => {
    const col = clean.map((s) => s[j]);
    const m = col.reduce((a, b) => a + b, 0) / n;
    return Math.sqrt(col.reduce((a, b) => a + (b - m) ** 2, 0) / n) || 1;
  });
  const noisy = clean.map((s) => s.map((x, j) => x + noise * std[j] * R.normal()));
  return { t: Array.from({ length: n }, (_, k) => k * sys.dt), clean, noisy };
}

/**
 * Savitzky–Golay smoothing (local quadratic fit over 2m+1 points) and first derivative
 * (local linear slope, which equals the quadratic-fit derivative for symmetric windows).
 * m = 1 without smoothing gives the plain central difference. Edge points are dropped.
 */
export function derivatives(X: number[][], dt: number, m: number): { X: number[][]; dX: number[][] } {
  const n = X.length, d = X[0].length;
  const denomD = (m * (m + 1) * (2 * m + 1)) / 3; // Σ k², k = −m..m
  const cS = Array.from({ length: 2 * m + 1 }, (_, q) => {
    const k = q - m;
    return (3 * (3 * m * m + 3 * m - 1) - 15 * k * k) / ((2 * m - 1) * (2 * m + 1) * (2 * m + 3));
  });
  const Xs: number[][] = [], dX: number[][] = [];
  for (let i = m; i < n - m; i++) {
    const s = new Array(d).fill(0), ds = new Array(d).fill(0);
    for (let k = -m; k <= m; k++) for (let j = 0; j < d; j++) {
      s[j] += (m === 1 ? (k === 0 ? 1 : 0) : cS[k + m]) * X[i + k][j];
      ds[j] += (k * X[i + k][j]) / (denomD * dt);
    }
    Xs.push(s); dX.push(ds);
  }
  return { X: Xs, dX };
}

/** Least squares with column scaling and a tiny ridge; returns coefficients for the active columns. */
function lstsq(Th: number[][], y: number[], active: number[]): number[] {
  const p = active.length, n = Th.length;
  if (!p) return [];
  const sc = active.map((c) => Math.sqrt(Th.reduce((a, r) => a + r[c] * r[c], 0) / n) || 1);
  const A = Array.from({ length: p }, () => new Float64Array(p)), b = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    const r = Th[i];
    for (let a = 0; a < p; a++) {
      const ra = r[active[a]] / sc[a];
      b[a] += ra * y[i];
      for (let c = a; c < p; c++) A[a][c] += ra * (r[active[c]] / sc[c]);
    }
  }
  for (let a = 0; a < p; a++) { for (let c = 0; c < a; c++) A[a][c] = A[c][a]; A[a][a] += 1e-10 * n; }
  // Gaussian elimination with partial pivoting
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < p; c++) {
    let piv = c;
    for (let r = c + 1; r < p; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < p; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= p; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(p).fill(0);
  for (let r = p - 1; r >= 0; r--) {
    let s = M[r][p];
    for (let k = r + 1; k < p; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x.map((xi, a) => xi / sc[a]);
}

/** Sequentially thresholded least squares. Returns Ξ as [equation][term]. */
export function stlsq(Th: number[][], dX: number[][], lambda: number, iters = 10): number[][] {
  const p = Th[0].length, d = dX[0].length;
  const Xi: number[][] = [];
  for (let j = 0; j < d; j++) {
    const y = dX.map((r) => r[j]);
    let active = Array.from({ length: p }, (_, c) => c);
    let coef = new Array(p).fill(0);
    for (let it = 0; it < iters; it++) {
      const sol = lstsq(Th, y, active);
      coef = new Array(p).fill(0);
      active.forEach((c, a) => (coef[c] = sol[a]));
      const next = active.filter((c) => Math.abs(coef[c]) >= lambda);
      if (next.length === active.length) break;
      active = next;
    }
    Xi.push(coef.map((c, k) => (active.includes(k) ? c : 0)));
  }
  return Xi;
}

export interface Discovery {
  Xi: number[][];
  nTerms: number;
  nTrue: number;
  missing: number;   // true terms not found
  spurious: number;  // terms found that are not in the true model
  maxRelErr: number; // on the correctly identified terms
}

/** Full pipeline: derivatives → library → STLSQ, plus a comparison with the true coefficients. */
export function discover(sys: SystemDef, data: Dataset, lambda: number, smooth: boolean): Discovery {
  const { X, dX } = derivatives(data.noisy, sys.dt, smooth ? 2 : 1);
  const Th = X.map((s) => sys.library.map((t) => t.f(s)));
  const Xi = stlsq(Th, dX, lambda);
  let nTerms = 0, nTrue = 0, missing = 0, spurious = 0, maxRelErr = 0;
  Xi.forEach((rowXi, j) => rowXi.forEach((c, k) => {
    const tr = sys.truth[j][k];
    if (c !== 0) nTerms++;
    if (tr !== 0) nTrue++;
    if (tr !== 0 && c === 0) missing++;
    if (tr === 0 && c !== 0) spurious++;
    if (tr !== 0 && c !== 0) maxRelErr = Math.max(maxRelErr, Math.abs(c - tr) / Math.abs(tr));
  }));
  return { Xi, nTerms, nTrue, missing, spurious, maxRelErr };
}

/** Right-hand side of a discovered model. */
export function modelRhs(sys: SystemDef, Xi: number[][]): (s: number[]) => number[] {
  return (s) => Xi.map((rowXi) => rowXi.reduce((a, c, k) => (c ? a + c * sys.library[k].f(s) : a), 0));
}

/** HTML for one equation, e.g. "dx/dt = 1.00 x − 0.50 xy". */
export function equationHtml(sys: SystemDef, coefs: number[], j: number, digits = 2): string {
  const parts: string[] = [];
  coefs.forEach((c, k) => {
    if (!c) return;
    const mag = Math.abs(c).toFixed(digits);
    const term = sys.library[k].name === '1' ? '' : ` ${sys.library[k].html}`;
    const sign = c < 0 ? '−' : '+';
    parts.push(parts.length ? ` ${sign} ${mag}${term}` : `${c < 0 ? '−' : ''}${mag}${term}`);
  });
  const lhs = `d<i>${sys.vars[j]}</i>/d<i>t</i>`;
  return `${lhs} = ${parts.length ? parts.join('') : '0'}`;
}
