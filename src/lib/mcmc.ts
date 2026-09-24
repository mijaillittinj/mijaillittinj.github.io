/**
 * Bayesian calibration of Newton's law of cooling with a Metropolis sampler.
 *
 *   T(t) = T∞ + (T₀ − T∞) exp(−t/τ),   T₀ known, (τ, T∞) unknown,
 *   data dᵢ = T(tᵢ) + εᵢ,  εᵢ ~ N(0, σ²) with σ known,
 *   uniform prior on a box.  The posterior is sampled by random-walk Metropolis.
 */
import { rng } from './linalg.ts';

export const T0 = 90; // °C, initial temperature (known)
export const BOX = { tau: [1, 40] as [number, number], Tinf: [0, 60] as [number, number] };

export const cooling = (t: number, tau: number, Tinf: number) => Tinf + (T0 - Tinf) * Math.exp(-t / tau);

export interface Data { t: number[]; d: number[]; sigma: number }

export function makeData(tau: number, Tinf: number, n: number, tmax: number, sigma: number, seed: number): Data {
  const R = rng(seed), t: number[] = [], d: number[] = [];
  for (let i = 0; i < n; i++) {
    const ti = (tmax * (i + 1)) / n;
    t.push(ti); d.push(cooling(ti, tau, Tinf) + sigma * R.normal());
  }
  return { t, d, sigma };
}

export function logPost(data: Data, tau: number, Tinf: number): number {
  if (tau < BOX.tau[0] || tau > BOX.tau[1] || Tinf < BOX.Tinf[0] || Tinf > BOX.Tinf[1]) return -Infinity;
  let s = 0;
  for (let i = 0; i < data.t.length; i++) s += (data.d[i] - cooling(data.t[i], tau, Tinf)) ** 2;
  return -s / (2 * data.sigma * data.sigma);
}

export interface Chain {
  tau: number[]; Tinf: number[]; accepted: number; proposed: number;
  cur: [number, number]; lp: number; R: ReturnType<typeof rng>;
}

export function startChain(data: Data, start: [number, number], seed: number): Chain {
  return { tau: [start[0]], Tinf: [start[1]], accepted: 0, proposed: 0, cur: [...start] as [number, number], lp: logPost(data, start[0], start[1]), R: rng(seed) };
}

/** n Metropolis steps with Gaussian proposal widths (wTau, wTinf). */
export function step(ch: Chain, data: Data, n: number, wTau: number, wTinf: number) {
  for (let k = 0; k < n; k++) {
    const p: [number, number] = [ch.cur[0] + wTau * ch.R.normal(), ch.cur[1] + wTinf * ch.R.normal()];
    const lp = logPost(data, p[0], p[1]);
    ch.proposed++;
    if (Math.log(Math.max(ch.R.uni(), 1e-300)) < lp - ch.lp) { ch.cur = p; ch.lp = lp; ch.accepted++; }
    ch.tau.push(ch.cur[0]); ch.Tinf.push(ch.cur[1]);
  }
}

export function quantile(xs: ArrayLike<number>, q: number): number {
  const a = Array.from(xs).sort((u, v) => u - v);
  if (!a.length) return NaN;
  const pos = (a.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

export function histogram(xs: ArrayLike<number>, lo: number, hi: number, bins: number) {
  const c = new Float64Array(bins);
  for (let i = 0; i < xs.length; i++) {
    const b = Math.floor(((xs[i] - lo) / (hi - lo)) * bins);
    if (b >= 0 && b < bins) c[b]++;
  }
  const w = (hi - lo) / bins, tot = xs.length * w || 1;
  return { edges: Array.from({ length: bins }, (_, i) => lo + i * w), dens: Array.from(c, (v) => v / tot), w };
}
