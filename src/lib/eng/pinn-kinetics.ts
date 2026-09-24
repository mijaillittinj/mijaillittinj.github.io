/**
 * Physics-informed neural network for consecutive reactions A → B → C (chemical tab).
 *
 * One small network maps time to the three concentrations, with the initial condition
 * built in (A(0) = 1, B(0) = C(0) = 0):
 *     A = 1 − φ(t) σ(z_A),  B = φ(t) σ(z_B),  C = φ(t) σ(z_C),  φ(t) = 1 − e^{−10 t},
 *     z_k(u) = b_k + Σ_j W_kj tanh(w_j u + v_j),  u = t/5 − 1,
 * so every concentration stays in [0, 1]. The rate constants k₁ = 10^{ℓ₁}, k₂ = 10^{ℓ₂}
 * (and, if the detector is uncalibrated, the factor s = e^{ℓ_s}) are trained together
 * with the weights. The loss is
 *     L = L_data + λ_ode L_ode,
 *     L_data = mean (s B(tᵢ) − Bᵢ)²  [+ (A(tᵢ) − Aᵢ)² if A is measured],
 *     L_ode  = mean over collocation points of
 *              (A' + k₁A)² + s²(B' − k₁A + k₂B)² + (C' − k₂B)²,
 * All derivatives (time and parameters) are analytic.
 */
import { rng } from '../linalg.ts';
import type { KData } from './kinetics.ts';

export interface KinCfg {
  hidden: number;
  lambdaOde: number;
  scaleUnknown: boolean;
  measureA: boolean;
  nCol: number;
}

export interface KinPinn { th: Float64Array; m: Float64Array; v: Float64Array; t: number; H: number }

const DUDT = 0.2; // u = t/5 − 1
const PHI = 10;  // φ(t) = 1 − e^{−PHI t}
const SIGN = [-1, 1, 1], BASE = [1, 0, 0];
const toU = (t: number) => t / 5 - 1;

/** Parameter layout: W (3H) | b (3) | w (H) | v (H) | ℓ₁ | ℓ₂ | ℓ_s */
export const idx = (H: number) => ({ W: 0, b: 3 * H, w: 3 * H + 3, v: 4 * H + 3, l1: 5 * H + 3, l2: 5 * H + 4, ls: 5 * H + 5, P: 5 * H + 6 });

export function initKinPinn(H: number, seed: number, l1: number, l2: number): KinPinn {
  const R = rng(seed), I = idx(H), th = new Float64Array(I.P);
  for (let q = 0; q < 3 * H; q++) th[I.W + q] = R.normal() * 0.8;
  for (let j = 0; j < H; j++) { th[I.w + j] = R.normal() * 2; th[I.v + j] = R.normal(); }
  th[I.l1] = l1; th[I.l2] = l2; th[I.ls] = 0;
  return { th, m: new Float64Array(I.P), v: new Float64Array(I.P), t: 0, H };
}

/** Concentrations at times ts (for plotting). */
export function evalKin(p: KinPinn, ts: ArrayLike<number>): { A: number[]; B: number[]; C: number[] } {
  const { th, H } = p, I = idx(H);
  const out = { A: [] as number[], B: [] as number[], C: [] as number[] };
  for (let i = 0; i < ts.length; i++) {
    const u = toU(ts[i]);
    const z = [th[I.b], th[I.b + 1], th[I.b + 2]];
    for (let j = 0; j < H; j++) {
      const s = Math.tanh(th[I.w + j] * u + th[I.v + j]);
      for (let k = 0; k < 3; k++) z[k] += th[I.W + k * H + j] * s;
    }
    const ph = 1 - Math.exp(-PHI * ts[i]), sg = z.map((v) => 1 / (1 + Math.exp(-v)));
    out.A.push(1 - ph * sg[0]); out.B.push(ph * sg[1]); out.C.push(ph * sg[2]);
  }
  return out;
}

export const kOf = (p: KinPinn) => { const I = idx(p.H); return { l1: p.th[I.l1], l2: p.th[I.l2], s: Math.exp(p.th[I.ls]) }; };

/** Loss terms and gradient. */
export function kinLossGrad(p: KinPinn, data: KData, cfg: KinCfg) {
  const { th, H } = p, I = idx(H);
  const grad = new Float64Array(I.P);
  const k1 = 10 ** th[I.l1], k2 = 10 ** th[I.l2], s = cfg.scaleUnknown ? Math.exp(th[I.ls]) : 1;
  const S = new Float64Array(H), D = new Float64Array(H);

  // forward pass at time t: fills S, D; returns c, ct (dc/dt) and the local factors
  const fwd = (t: number) => {
    const u = toU(t);
    const z = [th[I.b], th[I.b + 1], th[I.b + 2]], zu = [0, 0, 0];
    for (let j = 0; j < H; j++) {
      const sj = Math.tanh(th[I.w + j] * u + th[I.v + j]), dj = 1 - sj * sj;
      S[j] = sj; D[j] = dj;
      for (let k = 0; k < 3; k++) { const W = th[I.W + k * H + j]; z[k] += W * sj; zu[k] += W * th[I.w + j] * dj; }
    }
    const ph = 1 - Math.exp(-PHI * t), dph = PHI * Math.exp(-PHI * t);
    const sg = z.map((v) => 1 / (1 + Math.exp(-v)));
    const s1 = sg.map((v) => v * (1 - v)), s2 = s1.map((v, k) => v * (1 - 2 * sg[k]));
    const zt = zu.map((v) => DUDT * v);
    const c = sg.map((v, k) => BASE[k] + SIGN[k] * ph * v);
    const ct = sg.map((v, k) => SIGN[k] * (dph * v + ph * s1[k] * zt[k]));
    return { u, c, ct, ph, dph, s1, s2, zt };
  };
  // backward pass: given ∂L/∂c and ∂L/∂ct at the point of the last fwd call
  const back = (f: ReturnType<typeof fwd>, gc: number[], gct: number[]) => {
    const u = f.u;
    for (let k = 0; k < 3; k++) {
      const gz = SIGN[k] * (gc[k] * f.ph * f.s1[k] + gct[k] * (f.dph * f.s1[k] + f.ph * f.s2[k] * f.zt[k]));
      const gzu = SIGN[k] * gct[k] * f.ph * f.s1[k] * DUDT;
      if (gz === 0 && gzu === 0) continue;
      grad[I.b + k] += gz;
      for (let j = 0; j < H; j++) {
        const W = th[I.W + k * H + j], wj = th[I.w + j], sj = S[j], dj = D[j];
        grad[I.W + k * H + j] += gz * sj + gzu * wj * dj;
        grad[I.w + j] += gz * W * u * dj + gzu * W * dj * (1 - 2 * wj * u * sj);
        grad[I.v + j] += gz * W * dj - gzu * W * wj * 2 * sj * dj;
      }
    }
  };

  // data
  const smp = data.samples;
  const nd = smp.length * (cfg.measureA ? 2 : 1);
  let Ld = 0;
  for (const q of smp) {
    const f = fwd(q.t);
    const rB = s * f.c[1] - q.B;
    Ld += rB * rB;
    const gc = [0, (2 / nd) * rB * s, 0];
    if (cfg.scaleUnknown) grad[I.ls] += (2 / nd) * rB * f.c[1] * s;
    if (cfg.measureA) { const rA = f.c[0] - q.A; Ld += rA * rA; gc[0] = (2 / nd) * rA; }
    back(f, gc, [0, 0, 0]);
  }
  Ld /= nd;

  // ODE residuals at collocation points
  let Lp = 0;
  const nc = cfg.nCol, wP = cfg.lambdaOde / nc;
  let gk1 = 0, gk2 = 0, gls = 0;
  const s2 = s * s;
  for (let i = 0; i < nc; i++) {
    const t = 10 * (i / (nc - 1)) ** 1.5, f = fwd(t); // denser early, where A changes fastest
    const [A, B] = f.c, [At, Bt, Ct] = f.ct;
    const rA = At + k1 * A, rB = Bt - k1 * A + k2 * B, rC = Ct - k2 * B;
    // the B residual is weighted by s², so shrinking B while inflating s cannot make it cheap
    Lp += rA * rA + s2 * rB * rB + rC * rC;
    const gc = [2 * wP * (rA * k1 - s2 * rB * k1), 2 * wP * (s2 * rB * k2 - rC * k2), 0];
    const gct = [2 * wP * rA, 2 * wP * s2 * rB, 2 * wP * rC];
    gk1 += 2 * wP * (rA * A - s2 * rB * A);
    gk2 += 2 * wP * (s2 * rB * B - rC * B);
    if (cfg.scaleUnknown) gls += wP * 2 * s2 * rB * rB; // ∂(s² rB²)/∂ℓ_s = 2 s² rB²
    back(f, gc, gct);
  }
  Lp /= nc;
  grad[I.l1] += gk1 * k1 * Math.LN10;
  grad[I.l2] += gk2 * k2 * Math.LN10;
  grad[I.ls] += gls;

  const loss = Ld + cfg.lambdaOde * Lp;
  return { loss, Ld, Lp, grad };
}

export function kinAdam(p: KinPinn, grad: Float64Array, lr = 0.01, b1 = 0.9, b2 = 0.999) {
  p.t++;
  const c1 = 1 - b1 ** p.t, c2 = 1 - b2 ** p.t;
  for (let q = 0; q < p.th.length; q++) {
    p.m[q] = b1 * p.m[q] + (1 - b1) * grad[q];
    p.v[q] = b2 * p.v[q] + (1 - b2) * grad[q] * grad[q];
    p.th[q] -= (lr * (p.m[q] / c1)) / (Math.sqrt(p.v[q] / c2) + 1e-8);
  }
  // keep the rate constants inside the plotted range
  const I = idx(p.H);
  p.th[I.l1] = Math.max(-2, Math.min(1.5, p.th[I.l1]));
  p.th[I.l2] = Math.max(-2, Math.min(1.5, p.th[I.l2]));
  p.th[I.ls] = Math.max(Math.log(0.1), Math.min(Math.log(10), p.th[I.ls]));
}
