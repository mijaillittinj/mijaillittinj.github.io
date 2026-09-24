/**
 * A deliberately small physics-informed network, trained in the browser.
 *
 * Problem: steady heat conduction in a cooling fin (non-dimensional),
 *     θ''(x) − m² θ(x) = 0,   x ∈ [0, 1],   θ(0) = 1,   θ'(1) = 0,
 * with exact solution θ(x) = cosh(m(1 − x)) / cosh(m).
 * A handful of noisy "thermocouple" readings are available on part of the fin.
 *
 * Model: one hidden layer, u(x) = Σ a_j tanh(w_j x + b_j) + c.
 * All derivatives needed by the physics residual are analytic, and so are the
 * parameter gradients, which keeps the code short and fast (no autodiff).
 *
 * Loss = mean data misfit + λ_phys · mean residual² at collocation points
 *        + λ_bc · (boundary conditions).
 * With λ_phys = 0 this is plain data fitting. m can be fixed (known physics)
 * or learned from the data (the inverse problem).
 */
import { rng } from './linalg.ts';

export interface PinnConfig {
  hidden: number;
  lambdaPhys: number; // 0 = data only
  learnM: boolean;
  mInit: number;
  lr: number;
  seed: number;
}

export interface Net { a: Float64Array; w: Float64Array; b: Float64Array; c: number; logm: number }

export function initNet(h: number, seed: number, mInit: number): Net {
  const r = rng(seed);
  const a = new Float64Array(h), w = new Float64Array(h), b = new Float64Array(h);
  for (let j = 0; j < h; j++) {
    w[j] = r.normal() * 2.5;
    b[j] = r.normal() * 1.0;
    a[j] = r.normal() * 0.3;
  }
  return { a, w, b, c: 0.5, logm: Math.log(mInit) };
}

export function evalNet(net: Net, x: number): { u: number; du: number; d2u: number } {
  let u = net.c, du = 0, d2u = 0;
  for (let j = 0; j < net.a.length; j++) {
    const s = Math.tanh(net.w[j] * x + net.b[j]);
    const d = 1 - s * s;
    u += net.a[j] * s;
    du += net.a[j] * net.w[j] * d;
    d2u += net.a[j] * net.w[j] * net.w[j] * (-2 * s * d);
  }
  return { u, du, d2u };
}

export function exact(x: number, m: number) {
  return Math.cosh(m * (1 - x)) / Math.cosh(m);
}

interface Grad { a: Float64Array; w: Float64Array; b: Float64Array; c: number; logm: number }

/** Loss and gradient. data: [x, θ] pairs; col: collocation points. */
export function lossGrad(net: Net, data: [number, number][], col: Float64Array, cfg: PinnConfig) {
  const h = net.a.length;
  const g: Grad = { a: new Float64Array(h), w: new Float64Array(h), b: new Float64Array(h), c: 0, logm: 0 };
  const m = Math.exp(net.logm);
  let Ld = 0, Lp = 0, Lb = 0;

  // helper: accumulate coef * ∂(quantity)/∂params where quantity ∈ {u, du, d2u}
  const acc = (x: number, cu: number, cdu: number, cd2u: number) => {
    g.c += cu;
    for (let j = 0; j < h; j++) {
      const a = net.a[j], w = net.w[j];
      const z = w * x + net.b[j];
      const s = Math.tanh(z);
      const d = 1 - s * s;
      const dd = -2 * s * d;               // d/dz of d
      const g2 = -2 * s * d;               // u'' kernel
      const g2p = -2 * d * (1 - 3 * s * s); // d/dz of g2
      // u = a s
      g.a[j] += cu * s;
      g.w[j] += cu * a * d * x;
      g.b[j] += cu * a * d;
      // u' = a w d
      g.a[j] += cdu * w * d;
      g.w[j] += cdu * a * (d + w * dd * x);
      g.b[j] += cdu * a * w * dd;
      // u'' = a w² g2
      g.a[j] += cd2u * w * w * g2;
      g.w[j] += cd2u * a * (2 * w * g2 + w * w * g2p * x);
      g.b[j] += cd2u * a * w * w * g2p;
    }
  };

  // data misfit
  const nd = data.length;
  for (const [x, t] of data) {
    const { u } = evalNet(net, x);
    const r = u - t;
    Ld += (r * r) / nd;
    acc(x, (2 * r) / nd, 0, 0);
  }

  // physics residual
  if (cfg.lambdaPhys > 0) {
    const nc = col.length;
    for (let i = 0; i < nc; i++) {
      const x = col[i];
      const { u, d2u } = evalNet(net, x);
      const R = d2u - m * m * u;
      Lp += (R * R) / nc;
      const k = (2 * cfg.lambdaPhys * R) / nc;
      acc(x, -m * m * k, 0, k);
      if (cfg.learnM) g.logm += k * (-2 * m * m * u);
    }
    // boundary conditions (part of the physical model)
    const e0 = evalNet(net, 0), e1 = evalNet(net, 1);
    const r0 = e0.u - 1, r1 = e1.du;
    Lb = r0 * r0 + r1 * r1;
    acc(0, 2 * cfg.lambdaPhys * r0, 0, 0);
    acc(1, 0, 2 * cfg.lambdaPhys * r1, 0);
  }
  return { loss: Ld + cfg.lambdaPhys * (Lp + Lb), Ld, Lp, Lb, g };
}

/** Adam state and step. */
export function makeTrainer(net: Net, cfg: PinnConfig) {
  const h = net.a.length;
  const mom = { a: new Float64Array(h), w: new Float64Array(h), b: new Float64Array(h), c: 0, logm: 0 };
  const vel = { a: new Float64Array(h), w: new Float64Array(h), b: new Float64Array(h), c: 0, logm: 0 };
  let t = 0;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  return function step(g: Grad) {
    t++;
    const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
    const upd = (p: Float64Array, gm: Float64Array, mm: Float64Array, vv: Float64Array) => {
      for (let j = 0; j < p.length; j++) {
        mm[j] = b1 * mm[j] + (1 - b1) * gm[j];
        vv[j] = b2 * vv[j] + (1 - b2) * gm[j] * gm[j];
        p[j] -= (cfg.lr * (mm[j] / c1)) / (Math.sqrt(vv[j] / c2) + eps);
      }
    };
    upd(net.a, g.a, mom.a, vel.a);
    upd(net.w, g.w, mom.w, vel.w);
    upd(net.b, g.b, mom.b, vel.b);
    mom.c = b1 * mom.c + (1 - b1) * g.c; vel.c = b2 * vel.c + (1 - b2) * g.c * g.c;
    net.c -= (cfg.lr * (mom.c / c1)) / (Math.sqrt(vel.c / c2) + eps);
    if (cfg.learnM) {
      mom.logm = b1 * mom.logm + (1 - b1) * g.logm; vel.logm = b2 * vel.logm + (1 - b2) * g.logm * g.logm;
      net.logm -= (cfg.lr * (mom.logm / c1)) / (Math.sqrt(vel.logm / c2) + eps);
    }
  };
}

/** Synthetic sensor data on a sub-interval of the fin. */
export function makeData(mTrue: number, n: number, xmax: number, noise: number, seed: number): [number, number][] {
  const r = rng(seed);
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const x = 0.04 + ((xmax - 0.04) * i) / Math.max(n - 1, 1);
    out.push([x, exact(x, mTrue) + noise * r.normal()]);
  }
  return out;
}
