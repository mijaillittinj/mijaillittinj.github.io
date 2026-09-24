/**
 * Physics-informed network for earthquake location (civil tab).
 *
 * The travel time from the unknown source is represented in factored form,
 *     τ(x) = ‖x − x_s‖ / v · (1 + c_θ(x)),
 * where c_θ is a small coordinate network (one tanh layer). The factor ‖x − x_s‖/v puts
 * the point-source singularity in by construction (τ(x_s) = 0), which keeps training well
 * posed and fast; the network only has to learn the correction.
 * Trainable: the network weights, the epicentre x_s and the origin time t0.
 *
 * Loss = data term  (1/N) Σ_i (t0 + τ(x_i) − t_i^obs)² / σ_d²
 *      + λ · eikonal residual (1/M) Σ_k (v² |∇τ(x_k)|² − 1)²   at M collocation points.
 * In a homogeneous medium the eikonal equation forces c_θ → 0, so the PINN converges to
 * the same answer as the classical misfit search; the same formulation carries over to
 * heterogeneous media, where no closed-form travel time exists.
 * All gradients are analytic; the optimiser is Adam with per-group learning rates.
 */
import { rng } from '../linalg.ts';
import { V_P, type Pt } from './quake.ts';

export interface QuakeNet {
  H: number;
  a: Float64Array; w: Float64Array; b: Float64Array; // w: H × 2
  a0: number; xs: number; ys: number; t0: number;
}
export interface QuakePinnData { stations: Pt[]; times: number[]; col: Pt[]; L: number; v?: number; lambda?: number; sigmaD?: number }

/** Number of trainable parameters and their packing order: a, w, b, a0, xs, ys, t0. */
export const nParams = (H: number) => 4 * H + 4;

/** The correction network starts at zero (c ≡ 0), i.e. at the homogeneous travel time. */
export function initQuakeNet(H: number, seed: number, start: Pt, t0: number): QuakeNet {
  const R = rng(seed);
  const a = new Float64Array(H), w = new Float64Array(2 * H), b = new Float64Array(H);
  for (let j = 0; j < H; j++) { a[j] = 0; w[2 * j] = R.normal(); w[2 * j + 1] = R.normal(); b[j] = R.normal(); }
  return { H, a, w, b, a0: 0, xs: start.x, ys: start.y, t0 };
}

export function collocation(n: number, L: number, seed: number): Pt[] {
  const R = rng(seed);
  return Array.from({ length: n }, () => ({ x: L * R.uni(), y: L * R.uni() }));
}

interface Eval { c: number; g0: number; g1: number; s: Float64Array; sp: Float64Array; spp: Float64Array; u0: number; u1: number }

function evalC(net: QuakeNet, x: number, y: number, L: number): Eval {
  const H = net.H, k = 2 / L, u0 = 2 * x / L - 1, u1 = 2 * y / L - 1;
  const s = new Float64Array(H), sp = new Float64Array(H), spp = new Float64Array(H);
  let c = net.a0, g0 = 0, g1 = 0;
  for (let j = 0; j < H; j++) {
    const t = Math.tanh(net.w[2 * j] * u0 + net.w[2 * j + 1] * u1 + net.b[j]);
    s[j] = t; sp[j] = 1 - t * t; spp[j] = -2 * t * sp[j];
    c += net.a[j] * t;
    g0 += k * net.a[j] * sp[j] * net.w[2 * j];
    g1 += k * net.a[j] * sp[j] * net.w[2 * j + 1];
  }
  return { c, g0, g1, s, sp, spp, u0, u1 };
}

/** Travel time predicted by the network at a point. */
export function travelTime(net: QuakeNet, p: Pt, L: number, v = V_P): number {
  const d = Math.hypot(p.x - net.xs, p.y - net.ys);
  return (d / v) * (1 + evalC(net, p.x, p.y, L).c);
}

/** Loss terms and the gradient packed as [a, w, b, a0, xs, ys, t0]. */
export function quakeLossGrad(net: QuakeNet, D: QuakePinnData) {
  const H = net.H, L = D.L, v = D.v ?? V_P, lam = D.lambda ?? 1000, sd = D.sigmaD ?? 0.1, k = 2 / L;
  const g = new Float64Array(nParams(H));
  const iA = 0, iW = H, iB = 3 * H, iA0 = 4 * H, iXs = 4 * H + 1, iYs = 4 * H + 2, iT0 = 4 * H + 3;

  // data term
  let Ld = 0;
  const Ns = D.stations.length;
  for (let i = 0; i < Ns; i++) {
    const st = D.stations[i];
    const E = evalC(net, st.x, st.y, L);
    const px = st.x - net.xs, py = st.y - net.ys, d = Math.max(Math.hypot(px, py), 1e-9);
    const tau = (d / v) * (1 + E.c);
    const r = net.t0 + tau - D.times[i];
    Ld += (r * r) / (Ns * sd * sd);
    const wr = (2 * r) / (Ns * sd * sd), dt = (wr * d) / v;
    g[iT0] += wr;
    for (let j = 0; j < H; j++) {
      g[iA + j] += dt * E.s[j];
      const as = net.a[j] * E.sp[j];
      g[iB + j] += dt * as;
      g[iW + 2 * j] += dt * as * E.u0;
      g[iW + 2 * j + 1] += dt * as * E.u1;
    }
    g[iA0] += dt;
    const f = (wr * (1 + E.c)) / v;
    g[iXs] -= f * (px / d);
    g[iYs] -= f * (py / d);
  }

  // eikonal residual: q = v ∇τ = e (1 + c) + d ∇c,  R = |q|² − 1
  let Le = 0;
  const M = D.col.length;
  for (const p of D.col) {
    const E = evalC(net, p.x, p.y, L);
    const px = p.x - net.xs, py = p.y - net.ys, d = Math.max(Math.hypot(px, py), 1e-6);
    const e0 = px / d, e1 = py / d, cc = 1 + E.c;
    const q0 = e0 * cc + d * E.g0, q1 = e1 * cc + d * E.g1;
    const R = q0 * q0 + q1 * q1 - 1;
    Le += (R * R) / M;
    const wR = (2 * lam * R) / M, Q0 = 2 * q0 * wR, Q1 = 2 * q1 * wR;
    const Qe = Q0 * e0 + Q1 * e1;
    for (let j = 0; j < H; j++) {
      const w0 = net.w[2 * j], w1 = net.w[2 * j + 1], aj = net.a[j];
      const Qw = Q0 * w0 + Q1 * w1;
      g[iA + j] += Qe * E.s[j] + d * k * E.sp[j] * Qw;
      g[iB + j] += Qe * aj * E.sp[j] + d * k * aj * E.spp[j] * Qw;
      g[iW + 2 * j] += Qe * aj * E.sp[j] * E.u0 + d * k * aj * (E.spp[j] * E.u0 * Qw + E.sp[j] * Q0);
      g[iW + 2 * j + 1] += Qe * aj * E.sp[j] * E.u1 + d * k * aj * (E.spp[j] * E.u1 * Qw + E.sp[j] * Q1);
    }
    g[iA0] += Qe;
    // dq/dp = (1+c)(I − e eᵀ)/d + ∇c eᵀ ;  p = x − x_s
    const Qg = Q0 * E.g0 + Q1 * E.g1;
    const dp0 = (cc * (Q0 - Qe * e0)) / d + Qg * e0;
    const dp1 = (cc * (Q1 - Qe * e1)) / d + Qg * e1;
    g[iXs] -= dp0;
    g[iYs] -= dp1;
  }
  return { loss: Ld + lam * Le, data: Ld, eik: Le, grad: g };
}

/** Flat views for Adam and finite-difference checks. */
export function getParams(net: QuakeNet): Float64Array {
  const H = net.H, p = new Float64Array(nParams(H));
  p.set(net.a, 0); p.set(net.w, H); p.set(net.b, 3 * H);
  p[4 * H] = net.a0; p[4 * H + 1] = net.xs; p[4 * H + 2] = net.ys; p[4 * H + 3] = net.t0;
  return p;
}
export function setParams(net: QuakeNet, p: ArrayLike<number>) {
  const H = net.H;
  for (let j = 0; j < H; j++) { net.a[j] = p[j]; net.b[j] = p[3 * H + j]; }
  for (let j = 0; j < 2 * H; j++) net.w[j] = p[H + j];
  net.a0 = p[4 * H]; net.xs = p[4 * H + 1]; net.ys = p[4 * H + 2]; net.t0 = p[4 * H + 3];
}

/**
 * Adam trainer. Learning rates per group: network 1e-3, epicentre `lrSrc` km per step
 * (cosine-decayed over `total` steps), origin time 0.05 s.
 */
export function makeQuakeTrainer(net: QuakeNet, D: QuakePinnData, total = 1500, lrSrc = 2, lrNet = 1e-3) {
  const P = nParams(net.H), m = new Float64Array(P), s = new Float64Array(P);
  let t = 0;
  const lr = new Float64Array(P).fill(lrNet);
  lr[4 * net.H + 3] = 0.05;
  return {
    get iter() { return t; },
    step() {
      const r = quakeLossGrad(net, D);
      t++;
      const decay = 0.5 * (1 + Math.cos(Math.PI * Math.min(t / total, 1)));
      lr[4 * net.H + 1] = lr[4 * net.H + 2] = 0.05 + (lrSrc - 0.05) * decay;
      const p = getParams(net);
      const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
      for (let k = 0; k < P; k++) {
        m[k] = 0.9 * m[k] + 0.1 * r.grad[k];
        s[k] = 0.999 * s[k] + 0.001 * r.grad[k] * r.grad[k];
        p[k] -= (lr[k] * (m[k] / c1)) / (Math.sqrt(s[k] / c2) + 1e-12);
      }
      p[4 * net.H + 1] = Math.max(0, Math.min(D.L, p[4 * net.H + 1]));
      p[4 * net.H + 2] = Math.max(0, Math.min(D.L, p[4 * net.H + 2]));
      setParams(net, p);
      return r;
    },
  };
}

/** Network travel times on an n × n grid of cell centres (row 0 at the bottom), for isochrones. */
export function travelTimeGrid(net: QuakeNet, n: number, L: number, v = V_P): Float32Array {
  const out = new Float32Array(n * n), h = L / n;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) out[j * n + i] = travelTime(net, { x: (i + 0.5) * h, y: (j + 0.5) * h }, L, v);
  return out;
}
