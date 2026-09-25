/**
 * PINN for PV soiling and degradation (electrical tab).
 *
 * A network N_θ(u), u = 2t/(DAYS − 1) − 1, carries the soiling history; cleanings from
 * the maintenance log are hard resets:
 *     log SR_θ(t) = N_θ(t) − N_θ(t_k),     t_k = last cleaning ≤ t,
 * so SR_θ = 1 on every cleaning day. Trainable soiling rate r̃ (%/day) and degradation
 * d̃ (%/year). Loss (both terms in percent units):
 *     L = mean_i [100 (SR_θ(tᵢ) (1 − d̃ tᵢ / 36500) − PIᵢ)]²          data (monitoring)
 *       + λ mean_c [100 dN_θ/dt(t_c) + r̃]²                          soiling ODE dSR/dt = −r SR
 * The network is never told that soiling is exponential: the ODE residual is what makes
 * it so. Without cleanings, r̃ and d̃ enter only through their sum (confounding).
 * Network: N_θ(u) = Σ_j a_j tanh(w_j u + b_j); analytic gradients; Adam.
 */
import { rng } from '../linalg.ts';
import { DAYS, lastCleaning } from './pv.ts';

export interface PvPinn { th: Float64Array; m: Float64Array; v: Float64Array; t: number; H: number }

const ix = (H: number) => ({ a: 0, w: H, b: 2 * H, r: 3 * H, d: 3 * H + 1, P: 3 * H + 2 });
const DU = 2 / (DAYS - 1); // du/dt

export function initPvPinn(H: number, seed: number, r0 = 0.1, d0 = 0.3): PvPinn {
  const R = rng(seed), I = ix(H), th = new Float64Array(I.P);
  for (let j = 0; j < H; j++) {
    th[I.a + j] = R.normal() * 0.02;
    th[I.w + j] = R.normal() * 2;
    th[I.b + j] = R.normal() * 1;
  }
  th[I.r] = r0; th[I.d] = d0;
  return { th, m: new Float64Array(I.P), v: new Float64Array(I.P), t: 0, H };
}

export const rateOf = (p: PvPinn) => p.th[ix(p.H).r];        // %/day
export const degradationOf = (p: PvPinn) => p.th[ix(p.H).d]; // %/year

function netAt(p: PvPinn, u: number): number {
  const { th, H } = p, I = ix(H);
  let o = 0;
  for (let j = 0; j < H; j++) o += th[I.a + j] * Math.tanh(th[I.w + j] * u + th[I.b + j]);
  return o;
}
const uOf = (t: number) => t * DU - 1;

/** Soiling ratio SR_θ(t) for every day. */
export function pinnSoiling(p: PvPinn, cleanings: number[]): Float64Array {
  const last = lastCleaning(cleanings), out = new Float64Array(DAYS);
  const N = new Float64Array(DAYS);
  for (let t = 0; t < DAYS; t++) N[t] = netAt(p, uOf(t));
  for (let t = 0; t < DAYS; t++) out[t] = Math.exp(N[t] - N[last[t]]);
  return out;
}

const COL = Array.from({ length: 160 }, (_, i) => ((i + 0.5) / 160) * (DAYS - 1));

export function pvLossGrad(p: PvPinn, PI: ArrayLike<number>, cleanings: number[], lambda = 10) {
  const { th, H } = p, I = ix(H), g = new Float64Array(I.P);
  const last = lastCleaning(cleanings);
  // hidden activations for every day
  const T = new Float64Array(DAYS * H), N = new Float64Array(DAYS);
  for (let t = 0; t < DAYS; t++) {
    const u = uOf(t);
    let o = 0;
    for (let j = 0; j < H; j++) { const h = Math.tanh(th[I.w + j] * u + th[I.b + j]); T[t * H + j] = h; o += th[I.a + j] * h; }
    N[t] = o;
  }
  const dt = th[I.d];
  // data term
  let Ld = 0;
  const coef = new Float64Array(DAYS); // ∂L/∂N[t] accumulated (with signs for resets)
  for (let t = 0; t < DAYS; t++) {
    const sr = Math.exp(N[t] - N[last[t]]), D = 1 - (dt * t) / 36500;
    const e = 100 * (sr * D - PI[t]);
    Ld += e * e;
    const w = (2 / DAYS) * e * 100 * sr * D; // ∂(e²/n)/∂logSR
    coef[t] += w; coef[last[t]] -= w;
    g[I.d] += (2 / DAYS) * e * 100 * sr * (-t / 36500);
  }
  Ld /= DAYS;
  for (let t = 0; t < DAYS; t++) {
    const c = coef[t];
    if (c === 0) continue;
    const u = uOf(t);
    for (let j = 0; j < H; j++) {
      const h = T[t * H + j], dd = 1 - h * h, a = th[I.a + j];
      g[I.a + j] += c * h;
      g[I.w + j] += c * a * dd * u;
      g[I.b + j] += c * a * dd;
    }
  }
  // soiling ODE residual: R = 100 · N_u · du/dt + r̃
  let Lp = 0;
  const S = 100 * DU;
  for (const tc of COL) {
    const u = uOf(tc);
    let Nu = 0;
    const hs = new Float64Array(H);
    for (let j = 0; j < H; j++) { const h = Math.tanh(th[I.w + j] * u + th[I.b + j]); hs[j] = h; Nu += th[I.a + j] * th[I.w + j] * (1 - h * h); }
    const R = S * Nu + th[I.r];
    Lp += R * R;
    const w = (2 * lambda * R) / COL.length;
    g[I.r] += w;
    for (let j = 0; j < H; j++) {
      const h = hs[j], dd = 1 - h * h, a = th[I.a + j], wj = th[I.w + j], d2 = -2 * h * dd;
      g[I.a + j] += w * S * wj * dd;
      g[I.w + j] += w * S * a * (dd + wj * d2 * u);
      g[I.b + j] += w * S * a * wj * d2;
    }
  }
  Lp /= COL.length;
  return { loss: Ld + lambda * Lp, Ld, Lp, grad: g };
}

export function pvAdam(p: PvPinn, grad: Float64Array, lr: number, b1 = 0.9, b2 = 0.999) {
  p.t++;
  const c1 = 1 - b1 ** p.t, c2 = 1 - b2 ** p.t;
  for (let q = 0; q < p.th.length; q++) {
    p.m[q] = b1 * p.m[q] + (1 - b1) * grad[q];
    p.v[q] = b2 * p.v[q] + (1 - b2) * grad[q] * grad[q];
    p.th[q] -= (lr * (p.m[q] / c1)) / (Math.sqrt(p.v[q] / c2) + 1e-8);
  }
}
