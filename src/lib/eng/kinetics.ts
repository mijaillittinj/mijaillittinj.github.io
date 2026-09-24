/**
 * Consecutive first-order reactions A → B → C in a batch reactor (A₀ = 1, B₀ = C₀ = 0):
 *     C_A = e^{−k₁t},  C_B = k₁/(k₂ − k₁) (e^{−k₁t} − e^{−k₂t}),  C_C = 1 − C_A − C_B.
 * The inverse problem recovers (k₁, k₂) from noisy samples of the intermediate B
 * (and optionally A). With an unknown detector calibration factor s, B-only data are
 * exactly symmetric under k₁ ↔ k₂: C_B(k₂, k₁) = (k₂/k₁) C_B(k₁, k₂), and the factor is
 * absorbed by s, so the misfit has two equally good minima.
 */
import { rng } from '../linalg.ts';

export interface Conc { A: number; B: number; C: number }

export function conc(t: number, k1: number, k2: number): Conc {
  const A = Math.exp(-k1 * t);
  const d = k2 - k1;
  // near k1 = k2 use the limit k t e^{−k t} (with a first-order correction)
  const B = Math.abs(d) < 1e-7 * Math.max(k1, k2)
    ? k1 * t * Math.exp(-k1 * t) * (1 - (d * t) / 2)
    : (k1 / d) * (Math.exp(-k1 * t) - Math.exp(-k2 * t));
  return { A, B, C: 1 - A - B };
}

export interface Sample { t: number; B: number; A: number }
export interface KData { samples: Sample[]; sigma: number }

/** Noisy samples of B (and A) at n times in [tmin, tmax]; σ = noise × max C_B. */
export function makeData(k1: number, k2: number, noise: number, seed: number, n = 12, tmin = 0.4, tmax = 10): KData {
  const R = rng(seed);
  const ts = Array.from({ length: n }, (_, i) => tmin + ((tmax - tmin) * i) / (n - 1));
  let bmax = 0;
  for (let i = 0; i <= 400; i++) bmax = Math.max(bmax, conc((tmax * i) / 400, k1, k2).B);
  const sigma = noise * bmax;
  return {
    sigma,
    samples: ts.map((t) => { const c = conc(t, k1, k2); return { t, B: c.B + sigma * R.normal(), A: c.A + sigma * R.normal() }; }),
  };
}

export interface MisfitOpts { scaleUnknown: boolean; measureA: boolean }

/**
 * Mean squared misfit at (log₁₀k₁, log₁₀k₂). If the calibration factor is unknown,
 * B is modelled as s·C_B with s from linear least squares (closed form).
 */
export function misfit(l1: number, l2: number, data: KData, o: MisfitOpts): { m: number; s: number } {
  const k1 = 10 ** l1, k2 = 10 ** l2;
  const S = data.samples;
  const b = S.map((p) => conc(p.t, k1, k2));
  let s = 1;
  if (o.scaleUnknown) {
    let num = 0, den = 0;
    for (let i = 0; i < S.length; i++) { num += S[i].B * b[i].B; den += b[i].B * b[i].B; }
    s = den > 0 ? Math.max(0, num / den) : 1;
  }
  let m = 0, n = 0;
  for (let i = 0; i < S.length; i++) {
    m += (S[i].B - s * b[i].B) ** 2; n++;
    if (o.measureA) { m += (S[i].A - b[i].A) ** 2; n++; }
  }
  return { m: m / n, s };
}

/** Nelder–Mead in 2D. Returns the path of the best vertex (one entry per iteration). */
export function nelderMead(f: (x: [number, number]) => number, x0: [number, number], step = 0.3, maxIter = 120, tol = 1e-10): [number, number][] {
  let simplex: [number, number][] = [x0, [x0[0] + step, x0[1]], [x0[0], x0[1] + step]];
  let fv = simplex.map(f);
  const path: [number, number][] = [[...x0]];
  for (let it = 0; it < maxIter; it++) {
    const idx = [0, 1, 2].sort((a, b) => fv[a] - fv[b]);
    simplex = idx.map((i) => simplex[i]); fv = idx.map((i) => fv[i]);
    path.push([...simplex[0]]);
    if (Math.abs(fv[2] - fv[0]) < tol * (1 + Math.abs(fv[0])) && Math.hypot(simplex[2][0] - simplex[0][0], simplex[2][1] - simplex[0][1]) < 1e-4) break;
    const c: [number, number] = [(simplex[0][0] + simplex[1][0]) / 2, (simplex[0][1] + simplex[1][1]) / 2];
    const at = (t: number): [number, number] => [c[0] + t * (simplex[2][0] - c[0]), c[1] + t * (simplex[2][1] - c[1])];
    const xr = at(-1), fr = f(xr);
    if (fr < fv[0]) {
      const xe = at(-2), fe = f(xe);
      if (fe < fr) { simplex[2] = xe; fv[2] = fe; } else { simplex[2] = xr; fv[2] = fr; }
    } else if (fr < fv[1]) { simplex[2] = xr; fv[2] = fr; }
    else {
      const xc = fr < fv[2] ? at(-0.5) : at(0.5), fc = f(xc);
      if (fc < Math.min(fr, fv[2])) { simplex[2] = xc; fv[2] = fc; }
      else {
        for (const i of [1, 2]) {
          simplex[i] = [(simplex[0][0] + simplex[i][0]) / 2, (simplex[0][1] + simplex[i][1]) / 2];
          fv[i] = f(simplex[i]);
        }
      }
    }
  }
  return path;
}
