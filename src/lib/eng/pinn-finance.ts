/**
 * PINN for Black–Scholes calibration (finance tab).
 *
 * Coordinates: x = S/K ∈ [0, 2] and τ ∈ [0, 1] years; price in units of K, V̂ = V/K.
 * Black–Scholes is homogeneous of degree one in (S, K), so V̂(x, τ) obeys the same PDE
 * with K = 1 and a single strike gives a well-posed 2D problem. The data are quotes of
 * that one strike recorded on several days (different spots and times to maturity).
 *
 * Network: V̂_θ(x, τ) = c + Σ_j a_j tanh(w1_j u + w2_j v + b_j),  u = x − 1, v = 2τ − 1.
 * Trainable log-volatility ℓ = log σ. Loss:
 *     L = mean (V̂_θ(xᵢ, τᵢ) − V̂ᵢ)²                         data
 *       + λ_p mean R²,  R = V̂_τ − ½σ²x²V̂_xx − r x V̂_x + r V̂   Black–Scholes residual
 *       + λ_0 mean (V̂_θ(x, 0) − max(x − 1, 0))²             payoff at maturity
 *       + λ_b [mean V̂_θ(0, τ)² + mean (V̂_θ(2, τ) − (2 − e^{−rτ}))²]   boundaries
 * The network is never given the Black–Scholes formula. Derivatives in x, τ and in the
 * parameters are analytic (single tanh layer).
 */
import { rng } from '../linalg.ts';
import type { PinnQuote } from './finance.ts';

export interface FinPinn { th: Float64Array; m: Float64Array; v: Float64Array; t: number; H: number }
export interface FinCfg { r: number; K: number; lambdaP: number; lambdaIC: number; lambdaBC: number }

/** Parameter layout: a | w1 | w2 | b (H each) | c | ℓ */
const ix = (H: number) => ({ a: 0, w1: H, w2: 2 * H, b: 3 * H, c: 4 * H, l: 4 * H + 1, P: 4 * H + 2 });

export function initFinPinn(H: number, seed: number, sigma0: number): FinPinn {
  const R = rng(seed), I = ix(H), th = new Float64Array(I.P);
  for (let j = 0; j < H; j++) {
    th[I.a + j] = R.normal() * 0.1;
    th[I.w1 + j] = R.normal() * 2.5;
    th[I.w2 + j] = R.normal() * 1;
    th[I.b + j] = R.normal() * 1;
  }
  th[I.c] = 0.1;
  th[I.l] = Math.log(sigma0);
  return { th, m: new Float64Array(I.P), v: new Float64Array(I.P), t: 0, H };
}

export const sigmaOfNet = (p: FinPinn) => Math.exp(p.th[ix(p.H).l]);

/** Price in currency units at spots S (array) and one τ. */
export function evalFin(p: FinPinn, S: ArrayLike<number>, tau: number, K: number): number[] {
  const { th, H } = p, I = ix(H), v = 2 * tau - 1, out: number[] = [];
  for (let i = 0; i < S.length; i++) {
    const u = S[i] / K - 1;
    let o = th[I.c];
    for (let j = 0; j < H; j++) o += th[I.a + j] * Math.tanh(th[I.w1 + j] * u + th[I.w2 + j] * v + th[I.b + j]);
    out.push(o * K);
  }
  return out;
}

// fixed training points
const COL: [number, number][] = [];
{
  const R = rng(77);
  for (let q = 0; q < 140; q++) COL.push([2 * R.uni(), 0.02 + 0.98 * R.uni()]);
}
const ICX = Array.from({ length: 41 }, (_, i) => (2 * i) / 40);
const BCT = Array.from({ length: 11 }, (_, i) => i / 10);

export function finLossGrad(p: FinPinn, data: PinnQuote[], cfg: FinCfg) {
  const { th, H } = p, I = ix(H), g = new Float64Array(I.P);
  const sig = Math.exp(th[I.l]), s2 = sig * sig, r = cfg.r;
  const h = new Float64Array(H), d = new Float64Array(H), d2 = new Float64Array(H), d3 = new Float64Array(H);
  const hidden = (u: number, v: number) => {
    for (let j = 0; j < H; j++) {
      const z = th[I.w1 + j] * u + th[I.w2 + j] * v + th[I.b + j];
      const t = Math.tanh(z), dd = 1 - t * t;
      h[j] = t; d[j] = dd; d2[j] = -2 * t * dd; d3[j] = -2 * dd * dd + 4 * t * t * dd;
    }
  };
  const value = () => { let o = th[I.c]; for (let j = 0; j < H; j++) o += th[I.a + j] * h[j]; return o; };
  /** Accumulate w · ∂o/∂θ into g at (u, v). */
  const addO = (w: number, u: number, v: number) => {
    for (let j = 0; j < H; j++) {
      const a = th[I.a + j];
      g[I.a + j] += w * h[j];
      g[I.w1 + j] += w * a * d[j] * u;
      g[I.w2 + j] += w * a * d[j] * v;
      g[I.b + j] += w * a * d[j];
    }
    g[I.c] += w;
  };

  // data
  let Ld = 0;
  for (const q of data) {
    const u = q.S / cfg.K - 1, v = 2 * q.tau - 1;
    hidden(u, v);
    const e = value() - q.price / cfg.K;
    Ld += e * e;
    addO((2 * e) / data.length, u, v);
  }
  Ld /= data.length;

  // PDE residual
  let Lp = 0;
  const wp = cfg.lambdaP / COL.length;
  for (const [x, tau] of COL) {
    const u = x - 1, v = 2 * tau - 1;
    hidden(u, v);
    let o = th[I.c], ox = 0, oxx = 0, ov = 0;
    for (let j = 0; j < H; j++) {
      const a = th[I.a + j], w1 = th[I.w1 + j], w2 = th[I.w2 + j];
      o += a * h[j]; ox += a * d[j] * w1; oxx += a * d2[j] * w1 * w1; ov += a * d[j] * w2;
    }
    const Vt = 2 * ov;
    const R = Vt - 0.5 * s2 * x * x * oxx - r * x * ox + r * o;
    Lp += R * R;
    const c = 2 * R * wp;
    // ∂R/∂θ = 2 ∂o_v − ½σ²x² ∂o_xx − r x ∂o_x + r ∂o
    const cxx = -0.5 * s2 * x * x, cx = -r * x;
    for (let j = 0; j < H; j++) {
      const a = th[I.a + j], w1 = th[I.w1 + j], w2 = th[I.w2 + j];
      const dA = 2 * d[j] * w2 + cxx * d2[j] * w1 * w1 + cx * d[j] * w1 + r * h[j];
      const dW1 = 2 * a * d2[j] * u * w2 + cxx * a * (d3[j] * u * w1 * w1 + 2 * d2[j] * w1) + cx * a * (d2[j] * u * w1 + d[j]) + r * a * d[j] * u;
      const dW2 = 2 * a * (d2[j] * v * w2 + d[j]) + cxx * a * d3[j] * v * w1 * w1 + cx * a * d2[j] * v * w1 + r * a * d[j] * v;
      const dB = 2 * a * d2[j] * w2 + cxx * a * d3[j] * w1 * w1 + cx * a * d2[j] * w1 + r * a * d[j];
      g[I.a + j] += c * dA; g[I.w1 + j] += c * dW1; g[I.w2 + j] += c * dW2; g[I.b + j] += c * dB;
    }
    g[I.c] += c * r;
    g[I.l] += c * (-s2 * x * x * oxx); // ∂(−½σ²)/∂ℓ = −σ²
  }
  Lp /= COL.length;

  // payoff at τ = 0
  let L0 = 0;
  for (const x of ICX) {
    const u = x - 1, v = -1;
    hidden(u, v);
    const e = value() - Math.max(x - 1, 0);
    L0 += e * e;
    addO((2 * e * cfg.lambdaIC) / ICX.length, u, v);
  }
  L0 /= ICX.length;

  // boundaries x = 0 and x = 2
  let Lb = 0;
  for (const tau of BCT) {
    const v = 2 * tau - 1;
    hidden(-1, v);
    const e0 = value();
    addO((2 * e0 * cfg.lambdaBC) / (2 * BCT.length), -1, v);
    hidden(1, v);
    const e1 = value() - (2 - Math.exp(-r * tau));
    addO((2 * e1 * cfg.lambdaBC) / (2 * BCT.length), 1, v);
    Lb += e0 * e0 + e1 * e1;
  }
  Lb /= 2 * BCT.length;

  const loss = Ld + cfg.lambdaP * Lp + cfg.lambdaIC * L0 + cfg.lambdaBC * Lb;
  return { loss, Ld, Lp, L0, Lb, grad: g };
}

export function finAdam(p: FinPinn, grad: Float64Array, lr = 0.01, b1 = 0.9, b2 = 0.999) {
  p.t++;
  const c1 = 1 - b1 ** p.t, c2 = 1 - b2 ** p.t;
  for (let q = 0; q < p.th.length; q++) {
    p.m[q] = b1 * p.m[q] + (1 - b1) * grad[q];
    p.v[q] = b2 * p.v[q] + (1 - b2) * grad[q] * grad[q];
    p.th[q] -= (lr * (p.m[q] / c1)) / (Math.sqrt(p.v[q] / c2) + 1e-8);
  }
  const l = ix(p.H).l;
  p.th[l] = Math.max(Math.log(0.03), Math.min(Math.log(1.5), p.th[l]));
}
