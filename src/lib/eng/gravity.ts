/**
 * Gravity exploration with a 2D body: a horizontal cylinder of excess mass per unit
 * length λ = π R² Δρ (kg/m), centred at (x₀, depth z₀). Its vertical anomaly at the
 * surface is
 *     Δg(x) = 2 G λ z₀ / ((x − x₀)² + z₀²),
 * reported in mGal (1 mGal = 10⁻⁵ m/s²). The half-width at half-maximum equals z₀, and
 * the peak is 2Gλ/z₀, so the data determine λ but not R and Δρ separately.
 */
import { rng } from '../linalg.ts';

export const G = 6.674e-11;
const MGAL = 1e5;
/** λ is handled in units of 10⁶ kg/m to keep the fit well scaled. */
export const LSCALE = 1e6;
const K = 2 * G * LSCALE * MGAL;

export interface Body { x0: number; z0: number; m: number } // m = λ / 10⁶ kg/m

export const lambdaOf = (R: number, drho: number) => (Math.PI * R * R * drho) / LSCALE;
export const radiusOf = (m: number, drho: number) => Math.sqrt((m * LSCALE) / (Math.PI * drho));

export function anomaly(x: number, b: Body): number {
  const d = x - b.x0;
  return (K * b.m * b.z0) / (d * d + b.z0 * b.z0);
}

export interface Survey { x: number[]; g: number[]; sigma: number }

export function makeSurvey(b: Body, sigma: number, seed: number, n = 25, half = 500): Survey {
  const R = rng(seed);
  const x = Array.from({ length: n }, (_, i) => -half + (2 * half * i) / (n - 1));
  return { x, g: x.map((xi) => anomaly(xi, b) + sigma * R.normal()), sigma };
}

export function rmsMisfit(b: Body, s: Survey): number {
  let e = 0;
  for (let i = 0; i < s.x.length; i++) e += (s.g[i] - anomaly(s.x[i], b)) ** 2;
  return Math.sqrt(e / s.x.length);
}

function solve3(A: number[][], y: number[]): number[] | null {
  const M = A.map((r, i) => [...r, y[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-300) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 3; r++) if (r !== c) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k];
    }
  }
  return [0, 1, 2].map((i) => M[i][3] / M[i][i]);
}

/**
 * Levenberg–Marquardt fit of (x₀, z₀, m) with analytic Jacobian. Returns every
 * accepted iterate (for animation); the last one is the estimate.
 */
export function fitLM(s: Survey, start: Body, maxIter = 60, zmin = 5): Body[] {
  let b = { ...start };
  let mu = 1e-2;
  let cost = rmsMisfit(b, s);
  const path: Body[] = [{ ...b }];
  for (let it = 0; it < maxIter; it++) {
    const JtJ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], Jtr = [0, 0, 0];
    for (let i = 0; i < s.x.length; i++) {
      const d = s.x[i] - b.x0, q = d * d + b.z0 * b.z0;
      const J = [(K * b.m * b.z0 * 2 * d) / (q * q), (K * b.m * (d * d - b.z0 * b.z0)) / (q * q), (K * b.z0) / q];
      const r = s.g[i] - (K * b.m * b.z0) / q;
      for (let a = 0; a < 3; a++) { Jtr[a] += J[a] * r; for (let c = 0; c < 3; c++) JtJ[a][c] += J[a] * J[c]; }
    }
    let accepted = false;
    for (let tries = 0; tries < 12 && !accepted; tries++) {
      const A = JtJ.map((row, a) => row.map((v, c) => v + (a === c ? mu * (v + 1e-12) : 0)));
      const dlt = solve3(A, Jtr);
      if (!dlt) { mu *= 10; continue; }
      const nb = { x0: b.x0 + dlt[0], z0: Math.max(zmin, b.z0 + dlt[1]), m: Math.max(1e-6, b.m + dlt[2]) };
      const nc = rmsMisfit(nb, s);
      if (nc < cost) { b = nb; cost = nc; mu = Math.max(1e-7, mu / 3); accepted = true; path.push({ ...b }); }
      else mu *= 4;
    }
    if (!accepted) break;
    const last = path[path.length - 2];
    if (Math.abs(last.x0 - b.x0) < 1e-3 && Math.abs(last.z0 - b.z0) < 1e-3 && Math.abs(last.m - b.m) < 1e-6) break;
  }
  return path;
}
