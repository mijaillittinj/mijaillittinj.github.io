/**
 * Trustworthy AI example (computing tab): deep ensembles, split conformal prediction
 * and physics-informed ensembles on a damped oscillator.
 *
 * Process (synthetic): y'' + 2ζω y' + ω² y = 0, y(0) = 1, y'(0) = 0, observed with
 * Gaussian noise on part of [0, L] only: a gap in the middle and nothing beyond X_END.
 *
 * Model: one-hidden-layer tanh network u(x) = c + Σ_j a_j tanh(w_j s + b_j), s = 2x/L − 1.
 * All derivatives needed by the physics residual (u', u'') and all parameter gradients
 * are analytic (same pattern as src/lib/pinn.ts), trained with Adam.
 * Plain ensemble: loss = mean data misfit (+ tiny weight decay).
 * PINN ensemble: + λ · mean ODE residual² at collocation points over the whole domain.
 */
import { rng } from '../linalg.ts';

export const L = 10;
export const ZETA = 0.08;
export const OMEGA = 1.3;
export const GAP: [number, number] = [3.0, 6.4];
export const X_END = 7.4;

const WD = OMEGA * Math.sqrt(1 - ZETA * ZETA);
/** Exact solution of the oscillator with y(0) = 1, y'(0) = 0. */
export function truth(x: number): number {
  return Math.exp(-ZETA * OMEGA * x) * (Math.cos(WD * x) + ((ZETA * OMEGA) / WD) * Math.sin(WD * x));
}

export type Region = 'data' | 'gap' | 'extra';
export function region(x: number): Region {
  if (x > X_END) return 'extra';
  if (x > GAP[0] && x < GAP[1]) return 'gap';
  return 'data';
}

export interface Sample { x: number; y: number }
export interface Dataset { train: Sample[]; calib: Sample[]; test: Sample[] }

/**
 * Observations on the observed region only, split at random into training (≈ 70 %)
 * and calibration (≈ 30 %) sets; a separate test set covers the whole domain.
 */
export function makeData(seed: number, noise: number, nObs = 150, nTest = 400): Dataset {
  const R = rng(seed);
  const obs: Sample[] = [];
  while (obs.length < nObs) {
    const x = R.uni() * X_END;
    if (region(x) !== 'data') continue;
    obs.push({ x, y: truth(x) + noise * R.normal() });
  }
  const train: Sample[] = [], calib: Sample[] = [];
  obs.forEach((s, i) => ((i % 10) < 7 ? train : calib).push(s));
  const test: Sample[] = [];
  for (let i = 0; i < nTest; i++) {
    const x = ((i + 0.5) / nTest) * L;
    test.push({ x, y: truth(x) + noise * R.normal() });
  }
  return { train, calib, test };
}

/* --------------------------------------------------------------- network */

export interface TNet { a: Float64Array; w: Float64Array; b: Float64Array; c: Float64Array; m: Float64Array; v: Float64Array; t: number }

export function initTNet(hidden: number, seed: number): TNet {
  const R = rng(seed), P = 3 * hidden + 1;
  const net: TNet = { a: new Float64Array(hidden), w: new Float64Array(hidden), b: new Float64Array(hidden), c: new Float64Array(1), m: new Float64Array(P), v: new Float64Array(P), t: 0 };
  for (let j = 0; j < hidden; j++) {
    net.w[j] = R.normal() * 3;
    net.b[j] = R.normal() * 2;
    net.a[j] = R.normal() * 0.3;
  }
  return net;
}

const K = 2 / L; // ds/dx

export function evalT(net: TNet, x: number): { u: number; du: number; d2u: number } {
  const s = K * x - 1;
  let u = net.c[0], g1 = 0, g2 = 0;
  for (let j = 0; j < net.a.length; j++) {
    const t = Math.tanh(net.w[j] * s + net.b[j]), d = 1 - t * t;
    u += net.a[j] * t;
    g1 += net.a[j] * net.w[j] * d;
    g2 += net.a[j] * net.w[j] * net.w[j] * (-2 * t * d);
  }
  return { u, du: K * g1, d2u: K * K * g2 };
}

export interface TrainCfg { physics: number; decay: number; collocation: Float64Array }

/** Loss and gradient (flat [a, w, b, c]). physics = 0 gives the plain network. */
export function lossGradT(net: TNet, data: Sample[], cfg: TrainCfg) {
  const H = net.a.length, g = new Float64Array(3 * H + 1);
  let lData = 0, lPhys = 0;
  const nd = data.length;
  for (const { x, y } of data) {
    const s = K * x - 1;
    let u = net.c[0];
    const ts = new Float64Array(H);
    for (let j = 0; j < H; j++) { ts[j] = Math.tanh(net.w[j] * s + net.b[j]); u += net.a[j] * ts[j]; }
    const r = u - y;
    lData += r * r / nd;
    const q = (2 * r) / nd;
    for (let j = 0; j < H; j++) {
      const t = ts[j], d = 1 - t * t;
      g[j] += q * t;
      g[H + j] += q * net.a[j] * d * s;
      g[2 * H + j] += q * net.a[j] * d;
    }
    g[3 * H] += q;
  }
  if (cfg.physics > 0) {
    const nc = cfg.collocation.length, A = 2 * ZETA * OMEGA, B = OMEGA * OMEGA;
    for (let ci = 0; ci < nc; ci++) {
      const s = K * cfg.collocation[ci] - 1;
      let u = net.c[0], g1 = 0, g2 = 0;
      const T = new Float64Array(H);
      for (let j = 0; j < H; j++) {
        const t = Math.tanh(net.w[j] * s + net.b[j]), d = 1 - t * t;
        T[j] = t;
        u += net.a[j] * t; g1 += net.a[j] * net.w[j] * d; g2 += net.a[j] * net.w[j] * net.w[j] * (-2 * t * d);
      }
      // residual R = u'' + A u' + B u, with u' = K g1, u'' = K² g2
      const R = K * K * g2 + A * K * g1 + B * u;
      lPhys += (R * R) / nc;
      const q = (cfg.physics * 2 * R) / nc;
      const c2 = K * K, c1 = A * K;
      for (let j = 0; j < H; j++) {
        const a = net.a[j], w = net.w[j], t = T[j], d = 1 - t * t, e = -2 * t * d, de = -2 * d * d + 4 * t * t * d;
        // ∂u, ∂g1, ∂g2 with respect to a_j, w_j, b_j
        const ua = t, uw = a * d * s, ub = a * d;
        const g1a = w * d, g1w = a * d + a * w * e * s, g1b = a * w * e;
        const g2a = w * w * e, g2w = 2 * a * w * e + a * w * w * de * s, g2b = a * w * w * de;
        g[j] += q * (c2 * g2a + c1 * g1a + B * ua);
        g[H + j] += q * (c2 * g2w + c1 * g1w + B * uw);
        g[2 * H + j] += q * (c2 * g2b + c1 * g1b + B * ub);
      }
      g[3 * H] += q * B;
    }
  }
  let lDecay = 0;
  if (cfg.decay > 0) {
    for (let j = 0; j < H; j++) {
      lDecay += cfg.decay * (net.a[j] ** 2 + net.w[j] ** 2);
      g[j] += 2 * cfg.decay * net.a[j];
      g[H + j] += 2 * cfg.decay * net.w[j];
    }
  }
  return { loss: lData + cfg.physics * lPhys + lDecay, data: lData, phys: lPhys, g };
}

export function adamT(net: TNet, g: Float64Array, lr: number, b1 = 0.9, b2 = 0.999) {
  net.t++;
  const c1 = 1 - b1 ** net.t, c2 = 1 - b2 ** net.t;
  let p = 0;
  for (const arr of [net.a, net.w, net.b, net.c]) for (let j = 0; j < arr.length; j++, p++) {
    net.m[p] = b1 * net.m[p] + (1 - b1) * g[p];
    net.v[p] = b2 * net.v[p] + (1 - b2) * g[p] * g[p];
    arr[j] -= (lr * (net.m[p] / c1)) / (Math.sqrt(net.v[p] / c2) + 1e-8);
  }
}

/* --------------------------------------------------------------- ensemble */

export const HIDDEN = 20;
export const MEMBERS = 5;
export const PHYS_WEIGHT = 0.5;

export function collocation(n = 80): Float64Array {
  return Float64Array.from({ length: n }, (_, i) => ((i + 0.5) / n) * L);
}

export function makeEnsemble(physics: boolean, seed = 11) {
  const nets = Array.from({ length: MEMBERS }, (_, m) => initTNet(HIDDEN, seed + 97 * m));
  const cfg: TrainCfg = { physics: physics ? PHYS_WEIGHT : 0, decay: 1e-5, collocation: collocation() };
  return { nets, cfg };
}

/** Learning-rate schedule shared by the browser and the tests. */
export const lrAt = (it: number, total: number) => 0.01 * (0.1 + 0.9 * 0.5 * (1 + Math.cos((Math.PI * Math.min(it, total)) / total)));

export function ensembleStats(nets: TNet[], xs: ArrayLike<number>) {
  const n = xs.length, mean = new Float64Array(n), sd = new Float64Array(n);
  const members = nets.map((net) => Float64Array.from(xs as ArrayLike<number>, (x) => evalT(net, x).u));
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const m of members) s += m[i];
    const mu = s / members.length;
    let v = 0;
    for (const m of members) v += (m[i] - mu) ** 2;
    mean[i] = mu; sd[i] = Math.sqrt(v / (members.length - 1));
  }
  return { mean, sd, members };
}

/** Split conformal half-width: the ⌈(n+1)(1−α)⌉-th smallest calibration residual. */
export function conformalQuantile(scores: number[], coverage: number): number {
  const s = [...scores].sort((p, q) => p - q), n = s.length;
  const k = Math.ceil((n + 1) * coverage);
  return k > n ? Infinity : s[k - 1];
}

/** Empirical coverage of the band mean ± q per region of the test set. */
export function coverageByRegion(test: Sample[], mu: (x: number) => number, q: number) {
  const out: Record<Region, { hit: number; n: number }> = { data: { hit: 0, n: 0 }, gap: { hit: 0, n: 0 }, extra: { hit: 0, n: 0 } };
  for (const { x, y } of test) {
    const r = out[region(x)];
    r.n++;
    if (Math.abs(y - mu(x)) <= q) r.hit++;
  }
  return { data: out.data.hit / (out.data.n || 1), gap: out.gap.hit / (out.gap.n || 1), extra: out.extra.hit / (out.extra.n || 1) };
}

/** RMS error of a prediction against the noise-free truth, on the points of one region. */
export function rmsInRegion(mu: (x: number) => number, which: Region, n = 200): number {
  let e = 0, k = 0;
  for (let i = 0; i < n; i++) {
    const x = ((i + 0.5) / n) * L;
    if (region(x) !== which) continue;
    e += (mu(x) - truth(x)) ** 2; k++;
  }
  return Math.sqrt(e / (k || 1));
}

export const NN_STEPS = 4000;
export const PINN_STEPS = 6000;

/**
 * Out-of-distribution flag: the members disagree much more than they do where data
 * exist (3 × the median spread over the observed region, with a small floor).
 */
export function oodThreshold(xs: ArrayLike<number>, sd: ArrayLike<number>): number {
  const inData: number[] = [];
  for (let i = 0; i < xs.length; i++) if (region(xs[i]) === 'data') inData.push(sd[i]);
  inData.sort((p, q) => p - q);
  const med = inData.length ? inData[Math.floor(inData.length / 2)] : 0;
  return Math.max(3 * med, 0.02);
}
