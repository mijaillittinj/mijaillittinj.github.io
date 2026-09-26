/**
 * The Lorenz-63 system and an ensemble Kalman filter, for the chaos and data-assimilation lab.
 *
 *   dx/dt = σ (y − x),   dy/dt = x (ρ − z) − y,   dz/dt = x y − β z,
 * with the classic parameters σ = 10, ρ = 28, β = 8/3 (chaotic regime; largest Lyapunov
 * exponent ≈ 0.9 per time unit). Integrated with fourth-order Runge–Kutta.
 *
 * Data assimilation: a stochastic ensemble Kalman filter (EnKF) with perturbed observations.
 * Observations are noisy measurements of a chosen subset of (x, y, z) every few steps.
 */
import { rng } from './linalg.ts';

export type Vec3 = [number, number, number];
export const SIGMA = 10, RHO = 28, BETA = 8 / 3;

export function rhs(s: Vec3): Vec3 {
  const [x, y, z] = s;
  return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z];
}

export function rk4(s: Vec3, dt: number): Vec3 {
  const k1 = rhs(s);
  const k2 = rhs([s[0] + 0.5 * dt * k1[0], s[1] + 0.5 * dt * k1[1], s[2] + 0.5 * dt * k1[2]]);
  const k3 = rhs([s[0] + 0.5 * dt * k2[0], s[1] + 0.5 * dt * k2[1], s[2] + 0.5 * dt * k2[2]]);
  const k4 = rhs([s[0] + dt * k3[0], s[1] + dt * k3[1], s[2] + dt * k3[2]]);
  return [
    s[0] + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
    s[1] + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    s[2] + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
  ];
}

/** A point on the attractor (after discarding a transient). */
export function onAttractor(seed = 1, dt = 0.01): Vec3 {
  const R = rng(seed);
  let s: Vec3 = [1 + R.normal(), 1 + R.normal(), 20 + R.normal()];
  for (let i = 0; i < 2000; i++) s = rk4(s, dt);
  return s;
}

export function trajectory(s0: Vec3, n: number, dt: number): Vec3[] {
  const out: Vec3[] = [s0];
  let s = s0;
  for (let i = 1; i < n; i++) { s = rk4(s, dt); out.push(s); }
  return out;
}

export const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

export function mean(ens: Vec3[]): Vec3 {
  const m: Vec3 = [0, 0, 0];
  for (const e of ens) { m[0] += e[0]; m[1] += e[1]; m[2] += e[2]; }
  return [m[0] / ens.length, m[1] / ens.length, m[2] / ens.length];
}

/** Ensemble around s0 with isotropic Gaussian spread. */
export function makeEnsemble(s0: Vec3, n: number, spread: number, seed: number): Vec3[] {
  const R = rng(seed);
  return Array.from({ length: n }, () => [s0[0] + spread * R.normal(), s0[1] + spread * R.normal(), s0[2] + spread * R.normal()] as Vec3);
}

/**
 * Stochastic EnKF analysis step (in place). `obs` are measured values of the components in
 * `idx` (e.g. [0] for x only), with noise standard deviation r. Multiplicative inflation
 * counteracts the spread loss of small ensembles.
 */
export function enkfUpdate(ens: Vec3[], obs: number[], idx: number[], r: number, R: ReturnType<typeof rng>, inflation = 1.05) {
  const N = ens.length, p = idx.length;
  const m = mean(ens);
  // inflate
  for (const e of ens) for (let c = 0; c < 3; c++) e[c] = m[c] + inflation * (e[c] - m[c]);
  const mi = mean(ens);
  // anomalies
  const A = ens.map((e) => [e[0] - mi[0], e[1] - mi[1], e[2] - mi[2]]);
  // P Hᵀ (3 × p) and H P Hᵀ + R (p × p)
  const PHt = Array.from({ length: 3 }, () => new Array(p).fill(0));
  const S = Array.from({ length: p }, () => new Array(p).fill(0));
  for (const a of A) {
    for (let c = 0; c < 3; c++) for (let j = 0; j < p; j++) PHt[c][j] += a[c] * a[idx[j]];
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) S[i][j] += a[idx[i]] * a[idx[j]];
  }
  for (let c = 0; c < 3; c++) for (let j = 0; j < p; j++) PHt[c][j] /= N - 1;
  for (let i = 0; i < p; i++) { for (let j = 0; j < p; j++) S[i][j] /= N - 1; S[i][i] += r * r; }
  const Sinv = invert(S);
  // K = P Hᵀ S⁻¹ (3 × p)
  const K = PHt.map((row) => Sinv[0].map((_, j) => row.reduce((s, v, k) => s + v * Sinv[k][j], 0)));
  for (const e of ens) {
    const innov = idx.map((c, j) => obs[j] + r * R.normal() - e[c]);
    for (let c = 0; c < 3; c++) e[c] += K[c].reduce((s, v, j) => s + v * innov[j], 0);
  }
}

/** Inverse of a small symmetric positive-definite matrix (Gauss–Jordan). */
function invert(M: number[][]): number[][] {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    const d = A[c][c];
    for (let k = 0; k < 2 * n; k++) A[c][k] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c]; for (let k = 0; k < 2 * n; k++) A[r][k] -= f * A[c][k]; }
  }
  return A.map((row) => row.slice(n));
}

/**
 * Twin experiment (used by the tests and by the lab's readout logic): the truth and two
 * ensembles started from the same perturbed members, one free-running and one assimilating
 * observations every `every` steps. Returns time-averaged RMSE of the ensemble means.
 */
export function twin(opts: { steps: number; dt: number; every: number; r: number; idx: number[]; members: number; seed: number; spread?: number }) {
  const { steps, dt, every, r, idx, members, seed } = opts;
  const R = rng(seed + 100);
  let truth = onAttractor(seed, dt);
  const first = makeEnsemble(truth, members, opts.spread ?? 2, seed + 1);
  const free = first.map((e) => [...e] as Vec3), da = first.map((e) => [...e] as Vec3);
  let eFree = 0, eDa = 0, count = 0;
  for (let k = 1; k <= steps; k++) {
    truth = rk4(truth, dt);
    for (let i = 0; i < members; i++) { free[i] = rk4(free[i], dt); da[i] = rk4(da[i], dt); }
    if (k % every === 0) enkfUpdate(da, idx.map((c) => truth[c] + r * R.normal()), idx, r, R);
    if (k > steps / 4) { eFree += dist(mean(free), truth) ** 2; eDa += dist(mean(da), truth) ** 2; count++; }
  }
  return { rmseFree: Math.sqrt(eFree / count), rmseDa: Math.sqrt(eDa / count) };
}
